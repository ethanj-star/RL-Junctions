import os
import random
import torch
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import VecMonitor, VecNormalize, VecEnvWrapper
from sumo_rl import parallel_env
import supersuit as ss

# 路径动态获取 (Dynamic path acquisition)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CURRENT_DIR)
# 在获取的路径上加上文件名，且不用+可以自动处理跨平台操作系统的路径斜杠问题。
# (Append filenames using os.path.join to handle cross-platform slash issues automatically instead of using '+')
net_path = os.path.join(ROOT_DIR, 'SUMOroutes.net.xml')
route_path = os.path.join(ROOT_DIR, 'traffic.rou.rou.xml')

# 自动编号：寻找下一个可用的 Run 编号（Auto-numbering tool）
def get_next_run_number(base_dir, prefix="marl_run_"):
    """扫描目录，找到最大的 run 编号并 +1"""
    if not os.path.exists(base_dir):
        return 1
    existing_runs = []
    for folder in os.listdir(base_dir):
        if folder.startswith(prefix):
            try:
                # 提取数字部分，比如 'marl_run_3' 提取出 3
                # (Extract the numeric part, e.g., get 3 from 'marl_run_3')
                num = int(folder.replace(prefix, ""))
                existing_runs.append(num)
            except ValueError:
                continue
    return max(existing_runs) + 1 if existing_runs else 1


# 补丁：解决 5 个返回值与 4 个返回值的 API 世纪冲突  （solve API conflict of 5 return or 4 return）
class SB3CompatibilityWrapper(VecEnvWrapper):
    def __init__(self, venv):
        super().__init__(venv)

    def reset(self):
        # 如果新版环境返回 (obs, info) 两个值，我们只取 obs 喂给老实的 SB3
        obs = self.venv.reset()
        if isinstance(obs, tuple) and len(obs) == 2:
            return obs[0]
        return obs

    def step_async(self, actions):
        self.venv.step_async(actions)

    #这是最关键的一步。在旧标准里，AI 每走一步，环境会返回 4 个值：状态、奖励、是否结束 (done)、额外信息。
    #后来学术界觉得“结束”这个词太笼统了，于是新版把它硬生生拆成了两个词：“被动死掉的结束”和“主动超时的结束”。这样返回值就变成了 5 个。
    #我们的做法很简单：只要发生了其中一个，我们就把它俩重新揉成一个统一的“结束”信号，把 5 个值重新打包成 4 个值，喂给 SB3。
    def step_wait(self):
        results = self.venv.step_wait()
        # 如果是新版的 5 个返回值 (obs, reward, terminated, truncated, info)
        if len(results) == 5:
            obs, rews, terms, truncs, infos = results
            # 把 terminated 和 truncated 合并回 done
            dones = np.logical_or(terms, truncs)
            return obs, rews, dones, infos
        # 如果已经是 4 个返回值了，原样返回
        return results


if __name__ == '__main__':
    print("正在初始化多智能体 SUMO 环境...")

    #固定全局随机种子!!!!每次更改seed Fix global random seed
    seed = 868
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # 动态分配本次实验的专属路径(Dynamically allocate paths for this training run)
    saved_models_base = os.path.join(ROOT_DIR, 'saved_models')
    logs_base = os.path.join(ROOT_DIR, 'logs')
    os.makedirs(saved_models_base, exist_ok=True)
    os.makedirs(logs_base, exist_ok=True)

    # 自动计算本次是 MARL 的第几次运行(Automatically calculate the current run number)
    run_idx = get_next_run_number(saved_models_base, "marl_run_")
    print(f"\n 自动检测到历史记录，本次 MARL 分配为: [ 第 {run_idx} 次运行 ]")

    # 创建本次运行的专属模型保存文件夹(Create an exclusive model save folder for this run based on the ID)
    run_save_dir = os.path.join(saved_models_base, f'marl_run_{run_idx}')
    os.makedirs(run_save_dir, exist_ok=True)

    # 创建本次运行的专属 CSV 日志文件夹(Create an exclusive CSV log folder for this run based on the ID)
    run_csv_dir = os.path.join(logs_base, f'marl_run_{run_idx}')
    os.makedirs(run_csv_dir, exist_ok=True)
    csv_base_path = os.path.join(run_csv_dir, 'marl_output')

    # Tensorboard 总目录依然保持不变，localhost方便把多条曲线画在同一张图里对比
    #(Keep the main Tensorboard directory unchanged to easily plot multiple curves on the same graph on localhost)
    tensorboard_log_path = os.path.join(logs_base, 'ppo_marl_tb')

    # 1. 创建原生的 PettingZoo 多智能体并行环境
    #`sumo-rl` 库的底层逻辑是：它会去解析你传入的 `SUMOroutes.net.xml` 文件。
    # 地图里有几个配置了动态红绿灯的交叉路口，它就会自动生成几个 Agent。
    env = parallel_env(
        net_file=net_path,
        route_file=route_path,
        out_csv_name=csv_base_path,  # <--- 使用动态生成的CSV日志路径
        use_gui=False,
        num_seconds=3600,
        reward_fn='queue'
        #Pressure = 驶入车道的车辆数-驶出车道的车辆数
        #不写默认调用diff-waiting-time（等待时间差）的函数。但是不适合MARL，因为不同agent会互相干扰
    )

    #PettingZoo 确实不懂 SB3，但我们用 SuperSuit 把它强行“翻译”成了 SB3 的形状。既然它已经变成了 SB3 的形状，
    # 我们后续给它打补丁、加护甲（VecNormalize）、做监控（VecMonitor），就全都要依赖 stable_baselines3 提供的工具了。

    # 2. SuperSuit 魔法转换  1 个包含 3 个 Agent 的多智能体环境，伪装成 3 个并行的单智能体环境。
    env = ss.pettingzoo_env_to_vec_env_v1(env)

    # 3. 拼接成向量环境 (单核，但自带3路口并行)
    env = ss.concat_vec_envs_v1(
        env,
        num_vec_envs=1,  #目前多核不可用，似乎是云端wrapper服务器问题，多核可用的话把1改成4或8就可以了
        num_cpus=1,
        base_class='stable_baselines3'
    )

    # 4. 套上魔法补丁，把5个输入变成4个
    env = SB3CompatibilityWrapper(env)

    # 获取并打印底层真实存在的 Agent ID  # Get and print the actual underlying Agent IDs
    #print("当前环境中的 Agent 列表 (List of Agents in current env)：", env.venv.venv.venv.possible_agents)

    # 5. 包装 Monitor
    #负责“记账”的监控器。它盯着你的环境，每当一个回合（Episode）结束时，它把这回合走了多少步、拿了多少分记录下来，然后汇报给SB3。
    env = VecMonitor(env)

    # 6. 包装 VecNormalize (防崩溃护甲)
    #norm_obs=True（状态归一化） 路况数据（比如 0~150 的排队长度，0~1 的红绿灯相位）全部按比例压缩到均值为 0，方差为 1 的小范围区间内。
    #norm_reward=True（奖励归一化）把极其夸张的得分（比如发生死锁时扣 9000 分，通畅时扣 10 分）同样压缩成平缓的小数（比如 -2.5 到 -0.1）。
    #clip_obs=10.（状态极值裁剪）即使归一化之后，如果遇到罕见的变态数据，计算出来的值超过了 10 或者低于 -10，强行把它一刀切，最大只准是 10
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)

    # 确保环境的动作空间也被固定种子
    #env.seed(seed)

    # 7. 参数共享的PPO 模型  create PPO model, shared with parameters
    model = PPO(
        "MlpPolicy",  # 策略网络类型：采用多层感知机（Multi-Layer Perceptron）提取低维状态特征
        env,
        learning_rate=3e-5,  # 学习率：优化器的更新步长。采用较小的值（3e-5）以确保网络权重在非平稳交通流环境中的稳定收敛。
        n_steps=1024,  # 截断步数（Rollout Steps）：每次策略更新前，每个并行环境收集的交互步数。总缓冲区大小等于该值乘以环境总数。
        batch_size=256,  # 微批次大小（Mini-batch Size）：每次执行随机梯度下降（SGD）计算策略梯度时的样本量。
        n_epochs=5,  # 优化轮数：对单次收集的经验回放缓冲区（Buffer）数据进行反复复习的遍数。降低此值可防止模型对当前批次经验产生过拟合。
        clip_range=0.2,  # 裁剪参数（Clipping Parameter）：PPO算法的核心超参数，限制新旧策略比值在 [0.8, 1.2] 区间内，保证信任域（Trust Region）内的策略更新单调性。
        ent_coef=0.01,  # 熵系数（Entropy Coefficient）：AI 的好奇心。0.01 表示保留一丁点尝试新鲜动作的欲望，防止模型过早收敛至局部最优策略。
        target_kl=0.05, # 目标KL散度（Target KL Divergence）：如果这一轮学习导致 AI 的认知变化极大（KL散度超过0.05），强制终止这轮学习，防止策略灾难性崩塌。
        verbose=1,  # 日志级别：设置为 1 就是让它把 FPS、ep_rew_mean 这些表格数据打印
        device="cpu",# 计算设备：指定底层张量运算使用 CPU。由于本研究的 MLP 网络参数量较小，且 SUMO 仿真高度依赖 CPU 单线程计算，
        # 避免 GPU/CPU 之间的数据拷贝可降低通信延迟，提升整体采样效率（FPS）。
        tensorboard_log=tensorboard_log_path  # <--- 统一定位到 TB 日志主目录
    )

    # 第二道保险：定时自动存档 (Checkpoint)
    # 每隔 50000 步，自动把模型备份到当前专属的 run 文件夹里
    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path=os.path.join(run_save_dir, 'checkpoints'),
        name_prefix='rl_model'
    )

    print("环境就绪！SB3 正在接收多智能体数据流。开始训练...")
    # tb_log_name 会在 Tensorboard 目录下自动新建 "run_1", "run_2" 子文件夹
    model.learn(total_timesteps=500000, callback=checkpoint_callback, tb_log_name=f"run_{run_idx}")

    # 8. 保存模型和归一化参数到专属文件夹  (save models)
    model.save(os.path.join(run_save_dir, "ppo_marl_model"))
    #把所有传给 AI 的状态和奖励，都强行缩放到了一个非常小且标准的范围内（通常是 均值为 0，方差为 1）。
    #和训练时候一样，测试也要归一化
    env.save(os.path.join(run_save_dir, "vec_normalize_marl.pkl"))

    env.close()
    print(f"\n 多智能体训练完成！所有产出均已安全保存至专属目录:")
    print(f"模型与断点: {run_save_dir}")
    print(f"日志文件: {run_csv_dir}")