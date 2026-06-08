import os
import random
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecNormalize  # 【修改1】替换为防崩溃的单核 DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.utils import set_random_seed
from sumo_rl import SumoEnvironment
from wrappers import ThreeJunctionCentralizedWrapper
from typing import Callable

# 【终极修复：直接导入模块，拒绝 os.system 的静默失败】
from generate_Random_Traffic import generate_route_file


# 【修改2：参数对齐】引入与 MARL 代码完全相同的线性学习率衰减函数
def linear_schedule_with_min(initial_value: float, min_value: float) -> Callable[[float], float]:
    """
    带保底机制的线性衰减学习率生成器。
    """

    def func(progress_remaining: float) -> float:
        return min_value + progress_remaining * (initial_value - min_value)

    return func


# 路径动态获取 (Dynamic path acquisition)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))  # 获取实时路径
ROOT_DIR = os.path.dirname(CURRENT_DIR)  # 回到上一层（根目录）
net_path = os.path.join(ROOT_DIR, 'SUMOroutes.net.xml')
# 【修改3：对齐交通流】使用和 MARL 一样的随机泊松车流
route_path = os.path.join(ROOT_DIR, 'traffic.random.rou.xml')


# 自动编号器：寻找下一个可用的 Run 编号（防止新的log覆盖旧的）
def get_next_run_number(base_dir, prefix="single_queue_run_"):
    if not os.path.exists(base_dir):
        return 1
    existing_runs = []
    for folder in os.listdir(base_dir):
        if folder.startswith(prefix):
            try:
                num = int(folder.replace(prefix, ""))
                existing_runs.append(num)
            except ValueError:
                continue
    return max(existing_runs) + 1 if existing_runs else 1


# 把动态的 csv_base_path 传进来 (Pass in the dynamic csv_base_path)
def make_env(rank, seed, csv_base_path):
    def _init():
        raw_env = SumoEnvironment(
            net_file=net_path,
            route_file=route_path,
            out_csv_name=f"{csv_base_path}_{rank}",  # 动态挂载日志名字
            use_gui=False,
            num_seconds=3600,
            reward_fn='queue'  # queue 奖励函数
        )
        env = ThreeJunctionCentralizedWrapper(raw_env)

        # ==========================================
        # 🌟 核心拦截器：狸猫换太子 (Monkey Patch)
        # 挂载在底层 env 上，完美拦截 DummyVecEnv 的自动 reset 信号
        # ==========================================
        original_reset = env.reset

        def custom_reset(*args, **kwargs):
            print("\n🔄 [单智能体 Queue] 监听到底层环境重置信号，正在为本局生成全新泊松车流...")
            generate_route_file()
            return original_reset(*args, **kwargs)

        env.reset = custom_reset
        # ==========================================

        return env

    return _init


if __name__ == '__main__':
    # 固定全局随机种子 （！！！每次运行修改记录）
    seed = 8848
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    set_random_seed(seed)

    # 动态分配本次训练的路径
    saved_models_base = os.path.join(ROOT_DIR, 'saved_models')
    logs_base = os.path.join(ROOT_DIR, 'logs')
    os.makedirs(saved_models_base, exist_ok=True)
    os.makedirs(logs_base, exist_ok=True)

    # 自动计算本次是第几次运行
    run_idx = get_next_run_number(saved_models_base, "single_queue_run_")
    print(f"\n 自动检测到历史记录，本次分配为: [ 第 {run_idx} 次运行 ]")

    # 根据编号创建本次运行的专属模型保存文件夹
    run_save_dir = os.path.join(saved_models_base, f'single_queue_run_{run_idx}')
    os.makedirs(run_save_dir, exist_ok=True)

    # 根据编号创建本次运行的专属 CSV 日志文件夹
    run_csv_dir = os.path.join(logs_base, f'single_queue_run_{run_idx}')
    os.makedirs(run_csv_dir, exist_ok=True)
    csv_base_path = os.path.join(run_csv_dir, 'output')

    tensorboard_log_path = os.path.join(logs_base, 'ppo_single_tb')

    # 【核心安全修改】改为单核运行，避免多个进程同时争抢写入 traffic.random.rou.xml 导致 SUMO 崩溃
    # 同时也保证了和 MARL 对比时的硬件公平性
    num_cpu = 1
    print(f"正在后台启动 {num_cpu} 个单智能体 SUMO 环境...")

    # 【前置保障】：先生成一次兜底文件，防止首次启动找不到文件报错
    print("🚀 [系统启动] 正在生成初始的泊松随机交通流...")
    generate_route_file()

    # 使用 DummyVecEnv 替代 SubprocVecEnv 进行单核安全并行封装
    env = DummyVecEnv([make_env(i, seed=seed, csv_base_path=csv_base_path) for i in range(num_cpu)])
    env = VecMonitor(env)

    # 数据归一化 (Data normalization)
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)

    # 【修改4：超参数全量对齐】与 MARL 完全一致的黄金配置
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=linear_schedule_with_min(3e-4, 3e-5),  # 动态学习率
        n_steps=2048,  # 对齐
        batch_size=256,  # 对齐
        n_epochs=10,  # 对齐
        clip_range=0.2,
        ent_coef=0.03,  # 对齐
        target_kl=0.05,
        verbose=1,
        device="cpu",
        seed=seed,
        tensorboard_log=tensorboard_log_path
    )

    # 【修改5：激活存档点功能】
    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path=os.path.join(run_save_dir, 'checkpoints'),
        name_prefix='rl_model'
    )

    print(f"环境就绪！开始单智能体集中式训练...")
    model.learn(total_timesteps=361000, callback=checkpoint_callback, tb_log_name=f"run_{run_idx}")

    # 文件会自动存到：saved_models/single_queue_run_X/ppo_model.zip
    model.save(os.path.join(run_save_dir, "ppo_model"))
    env.save(os.path.join(run_save_dir, "vec_normalize.pkl"))

    env.close()
    print(f"\n 训练完成！所有产出均已安全保存至专属目录:")
    print(f"模型文件: {run_save_dir}")
    print(f"日志文件: {run_csv_dir}")