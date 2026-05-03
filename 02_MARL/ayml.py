import os
import random
import torch
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import VecMonitor, VecNormalize, VecEnvWrapper
from sumo_rl import parallel_env
import supersuit as ss
from typing import Callable, Dict, Any

# 直接导入模块，拒绝 os.system 的静默失败
from generate_Random_Traffic import generate_route_file

# 路径动态获取 (Dynamic path acquisition)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CURRENT_DIR)
# 在获取的路径上加上文件名，且不用+可以自动处理跨平台操作系统的路径斜杠问题。
# (Append filenames using os.path.join to handle cross-platform slash issues automatically instead of using '+')
net_path = os.path.join(ROOT_DIR, 'SUMOroutes.net.xml')
# 【修改2：换交通流】对接 8 向泊松随机车流
route_path = os.path.join(ROOT_DIR, 'traffic.random.rou.xml')


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


# 补丁：解决 5 个返回值与 4 个返回值的 API 世纪冲突 (恢复至最干净版本)
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
        if len(results) == 5:
            obs, rews, terms, truncs, infos = results
            dones = np.logical_or(terms, truncs)
            return obs, rews, dones, infos
        return results


# 基于潜力的奖励塑形 (Potential-Based Reward)

def pbrs_reward(traffic_signal):
    """
    修正后的 PBRS 奖励函数
    核心思想：势能 Φ(s) = -总排队数。排队越少，势能越高 (负得越少)。
    """

    # 1. 计算基础奖励 (Base Reward)
    # 获取当前状态下的总排队车辆数
    total_queue = traffic_signal.get_total_queued()
    base_reward = -total_queue

    # ---------------------------------------------------------
    # 【已删除】获取信号灯状态和计算 active_phase_queue 的臃肿循环
    # ---------------------------------------------------------

    # 2. 计算当前状态的势能 Φ(s)
    # 直接使用负的总排队数作为势能。状态越通畅，势能值越大（越接近0）。
    phi_current = -total_queue

    # 3. 势能污染防御 (处理跨回合清零，防御第一步污染)
    current_step = getattr(traffic_signal.env, "sim_step", 0)
    delta_time = getattr(traffic_signal.env, "delta_time", 5)
    is_new_episode = current_step <= delta_time

    if not hasattr(traffic_signal, 'last_potential') or is_new_episode:
        # 如果是第一步，让记忆直接等于当前状态（不产生任何差值）
        traffic_signal.last_potential = phi_current
        shaping_reward = 0.0  # 第一步没有状态转移，强行把势能奖金归零
    else:
        # 4. 计算势能差 (Shaping Reward) F = γ * Φ(s') - Φ(s)
        gamma = 0.99  # 折扣因子

        # 如果排队减少了（比如从 -10 变成了 -5），shaping_reward 将是一个正数，给予奖励！
        # 如果排队增加了（比如从 -5 变成了 -10），shaping_reward 将是一个负数，给予惩罚！
        shaping_reward = (gamma * phi_current) - traffic_signal.last_potential

        # 更新记忆，为下一步计算做准备
        traffic_signal.last_potential = phi_current

    # 5. 组合最终奖励: R' = R + (Beta * F)
    # 【重要修改】：因为现在的 phi_current 量级已经和 base_reward 相同（都是车辆数），
    # 所以 Beta 没必要再乘以 100 了，否则势能差会完全淹没基础奖励。改为 1.0 即可。
    beta = 1.0
    final_reward = base_reward + (beta * shaping_reward)

    # 加上静态缩放，弥补关闭奖励归一化以后的梯度更新锁死
    return final_reward / 100.0


if __name__ == '__main__':
    print("正在初始化多智能体 SUMO 环境...")

    # 固定全局随机种子             !!!!每次更改seed Fix global random seed
    seed = 8848
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
    # (Keep the main Tensorboard directory unchanged to easily plot multiple curves on the same graph on localhost)
    tensorboard_log_path = os.path.join(logs_base, 'ppo_marl_tb')

    # 【前置保障】：先生成一次兜底文件，防止首次启动找不到文件报错
    print(" [系统启动] 正在生成初始的泊松随机交通流...")
    generate_route_file()

    # 1. 创建原生的 PettingZoo 多智能体并行环境
    # `sumo-rl` 库的底层逻辑是：它会去解析你传入的 `SUMOroutes.net.xml` 文件。
    env = parallel_env(
        net_file=net_path,
        route_file=route_path,
        out_csv_name=csv_base_path,
        use_gui=False,
        num_seconds=3600,
        reward_fn=pbrs_reward
    )

    # 核心拦截器：猴子补丁 (a Monkey Patch)
    # 替换底层的 reset 方法，保证交通流生成在 SUMO 读取之前完成！ dynamic change reset function，add new traffic flow .xml
    original_reset = env.reset


    def custom_reset(*args, **kwargs):
        print("\n[动态刷新] 监听到底层环境重置信号，正在为本局生成全新泊松车流...")
        generate_route_file()
        return original_reset(*args, **kwargs)


    # 将原生 reset 替换为我们的拦截器
    env.reset = custom_reset

    # 2. SuperSuit 魔法转换
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    # 3. 拼接成向量环境
    env = ss.concat_vec_envs_v1(
        env,
        num_vec_envs=1,
        num_cpus=1,
        base_class='stable_baselines3'
    )
    # 4. 套上魔法补丁
    env = SB3CompatibilityWrapper(env)
    # 5. 包装 Monitor
    env = VecMonitor(env)
    # 6. 包装 VecNormalize (防崩溃护甲)
    env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.)

    # ==========================================
    # Asymmetric Learning Rate Setup
    # ==========================================
    base_lr = 3e-4  # Actor (Policy) Learning Rate
    vf_lr = 4.5e-4  # Critic (Value Function) Learning Rate (1.5x base)

    # We configure policy_kwargs to instruct the underlying network
    # to use different optimizer settings for the "pi" (policy) and "vf" (value) components.
    policy_kwargs = dict(
        optimizer_class=torch.optim.Adam,
        optimizer_kwargs=dict(
            weight_decay=0.0,  # standard
        ),
        # By providing a list of dicts mapped to network components,
        # SB3 constructs separate parameter groups in PyTorch's optimizer.
        # Note: Depending on your exact SB3 version, this specific dict mapping
        # for param groups via policy_kwargs might be handled slightly differently,
        # but this is the modern, cleanest way to structure it.
    )

    # Since direct injection of per-network LRs via policy_kwargs is sometimes
    # tricky depending on the SB3 version, the safest, bulletproof way to do
    # this in SB3 without writing a custom Policy subclass is to pass a
    # custom learning_rate function that returns a schedule based on a schedule variable,
    # OR if you just want static values, use the built-in support for multiple optimizers
    # if you switch algorithms (like SAC/TD3).

    # However, for PPO in standard SB3, to TRULY get different LRs, we must subclass.
    # Fortunately, there's a workaround using net_arch if we were sharing features,
    # but the cleanest way is a custom schedule function combined with policy_kwargs.

    # Let's write the custom policy class locally to ensure 100% compatibility.

    from stable_baselines3.common.policies import ActorCriticPolicy


    class AsymmetricLRPolicy(ActorCriticPolicy):
        def _build_optimizer(self) -> None:
            # We override the optimizer builder to set up param groups
            # This guarantees the Critic learns 1.5x faster.
            optimizer_kwargs = self.optimizer_kwargs.copy()
            if "lr" in optimizer_kwargs:
                del optimizer_kwargs["lr"]  # We manage it manually below

            self.optimizer = self.optimizer_class(
                [
                    {"params": self.features_extractor.parameters(), "lr": base_lr},
                    {"params": self.mlp_extractor.policy_net.parameters(), "lr": base_lr},
                    {"params": self.action_net.parameters(), "lr": base_lr},
                    # Multiply the value function learning rate
                    {"params": self.mlp_extractor.value_net.parameters(), "lr": vf_lr},
                    {"params": self.value_net.parameters(), "lr": vf_lr},
                ],
                **optimizer_kwargs,
            )


    # 7. 参数共享的PPO 模型
    model = PPO(
        AsymmetricLRPolicy,  # Use our custom policy class!
        env,
        learning_rate=base_lr,  # Passed, but overridden by our custom policy
        n_steps=2048,
        batch_size=256,
        n_epochs=10,
        clip_range=0.2,
        ent_coef=0.03,
        target_kl=0.05,
        verbose=1,
        device="cpu",
        tensorboard_log=tensorboard_log_path,
        policy_kwargs=policy_kwargs
    )

    # 第二道保险：定时自动存档 (Checkpoint)
    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path=os.path.join(run_save_dir, 'checkpoints'),
        name_prefix='rl_model'
    )

    print("环境就绪！SB3 正在接收多智能体数据流。开始训练 (非对称学习率)...")
    model.learn(total_timesteps=300000, callback=checkpoint_callback, tb_log_name=f"run_{run_idx}")

    # 8. 保存模型和归一化参数到专属文件夹
    model.save(os.path.join(run_save_dir, "ppo_marl_model"))
    env.save(os.path.join(run_save_dir, "vec_normalize_marl.pkl"))

    env.close()
    print(f"\n 多智能体训练完成！所有产出均已安全保存至专属目录:")
    print(f"模型与断点: {run_save_dir}")
    print(f"日志文件: {run_csv_dir}")