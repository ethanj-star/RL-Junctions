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

# ====== 1. 核心路径动态获取 (防迷路) ======
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CURRENT_DIR)

net_path = os.path.join(ROOT_DIR, 'SUMOroutes.net.xml')
route_path = os.path.join(ROOT_DIR, 'traffic.rou.rou.xml')


# ====== 2. 自动编号神器：寻找下一个可用的 Run 编号 ======
def get_next_run_number(base_dir, prefix="single_queue_run_"):
    """扫描目录，找到最大的 run 编号并 +1"""
    if not os.path.exists(base_dir):
        return 1
    existing_runs = []
    for folder in os.listdir(base_dir):
        if folder.startswith(prefix):
            try:
                # 提取数字部分，比如 'single_queue_run_3' 提取出 3
                num = int(folder.replace(prefix, ""))
                existing_runs.append(num)
            except ValueError:
                continue
    return max(existing_runs) + 1 if existing_runs else 1


# 稍微修改了 make_env，把动态的 csv_base_path 传进来
def make_env(rank, seed, csv_base_path):
    def _init():
        raw_env = SumoEnvironment(
            net_file=net_path,
            route_file=route_path,
            out_csv_name=f"{csv_base_path}_{rank}", # 动态挂载日志名字
            use_gui=False,
            num_seconds=3600,
            reward_fn='queue'   #queue 奖励函数
        )
        env = ThreeJunctionCentralizedWrapper(raw_env)
        return env

    return _init


if __name__ == '__main__':

    # ====== 核心：固定全局随机种子 ======
    seed = 666
    random.seed(seed)             # Python 的基础随机
    np.random.seed(seed)          # NumPy 的数学计算随机
    torch.manual_seed(seed)       # PyTorch的随机
    set_random_seed(seed)         # Stable-Baselines3 (SB3) 框架的随机

    # ================== 核心修改：动态分配本次实验的专属路径 ==================
    saved_models_base = os.path.join(ROOT_DIR, 'saved_models')
    logs_base = os.path.join(ROOT_DIR, 'logs')
    os.makedirs(saved_models_base, exist_ok=True)
    os.makedirs(logs_base, exist_ok=True)

    # 自动计算本次是第几次运行！
    run_idx = get_next_run_number(saved_models_base, "single_queue_run_")
    print(f"\n🚀 自动检测到历史记录，本次分配为: [ 第 {run_idx} 次运行 ]")

    # 创建本次运行的专属模型保存文件夹
    run_save_dir = os.path.join(saved_models_base, f'single_queue_run_{run_idx}')
    os.makedirs(run_save_dir, exist_ok=True)

    # 创建本次运行的专属 CSV 日志文件夹
    run_csv_dir = os.path.join(logs_base, f'single_queue_run_{run_idx}')
    os.makedirs(run_csv_dir, exist_ok=True)
    csv_base_path = os.path.join(run_csv_dir, 'output')

    # Tensorboard 总目录依然保持不变，方便把多条曲线画在同一张图里对比
    tensorboard_log_path = os.path.join(logs_base, 'ppo_single_tb')
    # =========================================================================

    num_cpu = 4
    print(f"正在后台启动 {num_cpu} 个并行的 SUMO 环境...")

    # 创建多进程环境，传入基础 seed 和动态生成的 csv 路径
    env = SubprocVecEnv([make_env(i, seed=seed, csv_base_path=csv_base_path) for i in range(num_cpu)])
    env = VecMonitor(env)

    # 数据归一化
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)

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
        tensorboard_log=tensorboard_log_path  # 统一定位到 TB 日志主目录
    )

    print(f"环境就绪！开始单智能体并行训练...")
    # tb_log_name 会在 Tensorboard 目录下自动新建 "run_1", "run_2" 子文件夹
    model.learn(total_timesteps=10000, tb_log_name=f"run_{run_idx}")

    # ================= 救命的修改在这里 =================
    # 文件会自动存到：saved_models/single_queue_run_X/ppo_model.zip
    model.save(os.path.join(run_save_dir, "ppo_model"))
    env.save(os.path.join(run_save_dir, "vec_normalize.pkl"))

    env.close()
    print(f"\n✅ 训练完成！所有产出均已安全保存至专属目录:")
    print(f"模型文件: {run_save_dir}")
    print(f"日志文件: {run_csv_dir}")