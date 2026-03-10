import os
import random
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.utils import set_random_seed
from sumo_rl import SumoEnvironment
from wrappers import ThreeJunctionCentralizedWrapper


def make_env(rank, seed=0):
    def _init():
        out_csv = f'logs/output_process_single_{rank}' # 改了下名字，区分单多智能体日志
        raw_env = SumoEnvironment(
            net_file='SUMOroutes.net.xml',
            route_file='traffic.rou.rou.xml',
            out_csv_name=out_csv,
            use_gui=False,
            num_seconds=3600,
            reward_fn='queue'   # <====== 【这里加上了 queue 奖励！】
        )
        env = ThreeJunctionCentralizedWrapper(raw_env)
        return env

    return _init


if __name__ == '__main__':

    # ====== 核心：固定全局随机种子 ======
    seed = 666
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    set_random_seed(seed)

    num_cpu = 4
    print(f"正在后台启动 {num_cpu} 个并行的 SUMO 环境...")

    # 创建多进程环境，并传入基础 seed
    env = SubprocVecEnv([make_env(i, seed=seed) for i in range(num_cpu)])
    env = VecMonitor(env)

    # ================== 核心优化 1：数据归一化 ==================
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)

    # ================== 核心优化 2：PPO 参数的“稳如老狗”配置 ==================
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-5,
        n_steps=1024,
        batch_size=256,
        n_epochs=5,
        clip_range=0.2,
        ent_coef=0.005,
        target_kl=0.05,
        verbose=1,
        device="cpu",
        seed=seed,
        tensorboard_log="./logs/ppo_single_queue/"  # 改了日志文件夹名
    )

    print(f"🔥 环境就绪！开始单智能体并行训练...")
    model.learn(total_timesteps=500000)

    # ================= 救命的修改在这里 =================
    # 保存模型 (名字改成了 single，绝对不能用 multi，否则会覆盖你的旧模型！)
    model.save("saved_models/ppo_3juc_single_queue")
    env.save("saved_models/vec_normalize_single_queue.pkl")

    env.close()
    print("🎉 训练完成，单智能体模型已保存！")