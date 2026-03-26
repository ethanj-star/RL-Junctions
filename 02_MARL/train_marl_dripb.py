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



# 新增：非线性平方协同惩罚 (Quadratic Collaborative Reward)

def drip_reward(traffic_signal):
    """
    终极融合：全局差分协同 (Difference) + 局部潜力引导 (PBRS)
    既防自私陷阱，又带老交警直觉，并使用静态缩放保护 PPO。
    """

    # 模块 1：计算差分/协同全局基础奖励 (替代原来的单点 base_reward)

    global_queue = 0
    # 动态遍历全网路口  get num of global queue
    for ts_id in traffic_signal.env.traffic_signals:
        global_queue += traffic_signal.env.traffic_signals[ts_id].get_total_queued()

    # 获取自己路口的排队  for each agent, get self queue
    local_queue = traffic_signal.get_total_queued()

    alpha = 0.8  # 自身路口的权重   self queue weight
    beta_global = 0.2  # 全网大局的权重   global queue weight

    # 用包含全局视野的差分惩罚，作为新的  diff Reward
    collaborative_base_reward = -(alpha * local_queue) - (beta_global * global_queue)


    # 模块 2：计算当前势能 Phi(s')  calculate potential
    # get traffic light state
    current_light_state = traffic_signal.sumo.trafficlight.getRedYellowGreenState(traffic_signal.id)
    active_phase_queue = 0
    for i, lane in enumerate(traffic_signal.lanes):
        # 如果这个车道当前是绿灯或黄灯   if the light == green and yellow
        if current_light_state[i] in ('G', 'g', 'y', 'Y'):
            active_phase_queue += traffic_signal.sumo.lane.getLastStepHaltingNumber(lane)  #get that lane car number

    # 计算排队占比，这里分母用本路口的 local_queue 即可  QLa / ∑QL
    phi_current = active_phase_queue / (local_queue + 1e-6)


    # 模块 3：提取上一步势能并计算差分 F

    if not hasattr(traffic_signal, 'last_potential'):
        traffic_signal.last_potential = 0.0

    gamma = 0.99  # 折扣因子
    #势能差计算，排队的车越多，说明本次绿灯行动越有意义，减去上一步的势能得到势能差
    shaping_reward = (gamma * phi_current) - traffic_signal.last_potential

    # 更新记忆，为下一步计算做准备
    traffic_signal.last_potential = phi_current


    # 模块 4：终极组合与静态缩放

    # 注意：因为 Base Reward 现在加入了全网排队，惩罚数值变得比以前更巨大了！
    # 为了防止势能奖金再次被淹没，我们把势能放大系数调高到 100.0 (你可以根据实际情况微调)
    beta_shaping = 100.0

    # R' = R (大局观底薪) + (Beta * F) (老交警绩效)
    final_reward = collaborative_base_reward + (beta_shaping * shaping_reward)

    # 加上静态缩放，拯救 Critic 网络，避免 PPO 梯度锁死！
    return final_reward / 100.0


if __name__ == '__main__':
    print("正在初始化多智能体 SUMO 环境...")

    # 固定全局随机种子                              !!!!每次更改seed Fix global random seed
    seed = 8848
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    saved_models_base = os.path.join(ROOT_DIR, 'saved_models')
    logs_base = os.path.join(ROOT_DIR, 'logs')
    os.makedirs(saved_models_base, exist_ok=True)
    os.makedirs(logs_base, exist_ok=True)

    run_idx = get_next_run_number(saved_models_base, "marl_run_")
    print(f"\n 自动检测到历史记录，本次 MARL 分配为: [ 第 {run_idx} 次运行 ]")

    run_save_dir = os.path.join(saved_models_base, f'marl_run_{run_idx}')
    os.makedirs(run_save_dir, exist_ok=True)

    run_csv_dir = os.path.join(logs_base, f'marl_run_{run_idx}')
    os.makedirs(run_csv_dir, exist_ok=True)
    csv_base_path = os.path.join(run_csv_dir, 'marl_output')

    tensorboard_log_path = os.path.join(logs_base, 'ppo_marl_tb')

    env = parallel_env(
        net_file=net_path,
        route_file=route_path,
        out_csv_name=csv_base_path,
        use_gui=False,
        num_seconds=3600,
        reward_fn=drip_reward  # <==== 【修改点1】：调用强效非线性奖励函数
    )

    env = ss.pettingzoo_env_to_vec_env_v1(env)

    env = ss.concat_vec_envs_v1(
        env,
        num_vec_envs=1,
        num_cpus=1,
        base_class='stable_baselines3'
    )

    env = SB3CompatibilityWrapper(env)

    env = VecMonitor(env)

    # ：关闭 norm_reward！极其关键！绝对不能让 SB3 洗白我们精心计算的惩罚量级！
    env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.)

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-5,
        n_steps=1024,
        batch_size=256,
        n_epochs=5,
        clip_range=0.2,
        ent_coef=0.01,
        target_kl=0.05,
        verbose=1,
        device="cpu",
        tensorboard_log=tensorboard_log_path
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path=os.path.join(run_save_dir, 'checkpoints'),
        name_prefix='rl_model'
    )

    print("环境就绪！SB3 正在接收多智能体数据流。开始训练...")
    model.learn(total_timesteps=300000, callback=checkpoint_callback, tb_log_name=f"run_{run_idx}")

    model.save(os.path.join(run_save_dir, "ppo_marl_model"))
    env.save(os.path.join(run_save_dir, "vec_normalize_marl.pkl"))

    env.close()
    print(f"\n 多智能体训练完成！所有产出均已安全保存至专属目录:")
    print(f"模型与断点: {run_save_dir}")
    print(f"日志文件: {run_csv_dir}")