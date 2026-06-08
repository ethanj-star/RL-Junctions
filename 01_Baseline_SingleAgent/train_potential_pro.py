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
from typing import Callable

def linear_schedule_with_min(initial_value: float, min_value: float) -> Callable[[float], float]:
    """
    带保底机制的线性衰减学习率生成器。
    :param initial_value: 初始最大学习率 (例如 3e-4)
    :param min_value: 最低保底学习率 (例如 3e-5)
    :return: 返回一个根据剩余进度计算当前学习率的函数
    """
    def func(progress_remaining: float) -> float:
        """
        progress_remaining 的值会从 1.0 (训练开始) 线性下降到 0.0 (训练结束)
        """
        # 数学映射：当进度为 1 时，结果是 initial_value；当进度为 0 时，结果是 min_value
        return min_value + progress_remaining * (initial_value - min_value)
    return func


# 路径动态获取 (Dynamic path acquisition)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))   #获取实时路径 (Get real-time absolute path)
ROOT_DIR = os.path.dirname(CURRENT_DIR)        #回到上一层（根目录） (Go to parent directory / root dir)
# 在获取的路径上加上文件名，且不用+可以自动处理跨平台操作系统的路径斜杠问题。 (Append filenames using os.path.join to handle cross-platform slash issues automatically instead of using '+')
net_path = os.path.join(ROOT_DIR, 'SUMOroutes.net.xml')
route_path = os.path.join(ROOT_DIR, 'traffic.rou.rou.xml')


# 新增：基于潜力的奖励塑形 (Potential-Based Reward)

def pbrs_reward(traffic_signal):

    # Φ(s, a) = QLa / ∑QL

    # 1. 计算基础奖励 (Base Reward: sum up every queued cars)
    total_queue = traffic_signal.get_total_queued()
    base_reward = -total_queue

    # 2. 获取信号灯状态 get traffic light state
    current_light_state = traffic_signal.sumo.trafficlight.getRedYellowGreenState(traffic_signal.id)
    #if lane's light == green and yellow, return and sum up all waiting cars
    active_phase_queue = 0
    for i, lane in enumerate(traffic_signal.lanes):
        # 如果这个车道当前是绿灯 (G/g) 或黄灯 (Y/y)
        if current_light_state[i] in ('G', 'g', 'y', 'Y'):
            active_phase_queue += traffic_signal.sumo.lane.getLastStepHaltingNumber(lane)  #返回所有静止的车 return every paused cars

    # 计算排队占比 (加上 1e-6 防止分母为 0 报错)  calculate potential Phi(s, a)
    phi_current = active_phase_queue / (total_queue + 1e-6)

    # 3. 提取上一步的势能 Phi(s)  initialise last_potential
    if not hasattr(traffic_signal, 'last_potential'):
        traffic_signal.last_potential = 0.0

    # 4. 计算最终的塑形奖励 F
    gamma = 0.99  # 折扣因子
    #势能差计算，排队的车越多，说明本次绿灯行动越有意义，减去上一步的势能得到势能差  diff = potential - late potential
    shaping_reward = (gamma * phi_current) - traffic_signal.last_potential
    # 更新记忆，为下一步计算做准备  update last_potential memory
    traffic_signal.last_potential = phi_current

    # 5. 组合最终奖励: R' = R + (Beta * F)  calculate final reward
    beta = 100.0
    final_reward = base_reward + (beta * shaping_reward)
    return final_reward / 100.0   #加上静态缩放，弥补关闭奖励归一化以后的梯度更新锁死。/ 100 to normalize


# 自动编号器：寻找下一个可用的 Run 编号（防止新的log覆盖旧的） (Auto-numbering tool: Find the next available Run ID to prevent new logs from overwriting old ones)
def get_next_run_number(base_dir, prefix="single_queue_run_"):
# 扫描目录，找到最大的 run 编号并 +1 (Scan the directory, find the max run ID and add 1)
    if not os.path.exists(base_dir):
        return 1              #如果父目录都没有就定义成第一个 (If parent dir doesn't exist, start with 1)
    existing_runs = []
    for folder in os.listdir(base_dir):
        if folder.startswith(prefix):
            try:
                # 提取数字部分，比如 'single_queue_run_3' 提取出 3 (Extract the numeric part, e.g., get 3 from 'single_queue_run_3')
                num = int(folder.replace(prefix, ""))
                existing_runs.append(num)
            except ValueError:
                continue           #遇到名字不对的跳过 (Skip if the folder name format is incorrect)
    return max(existing_runs) + 1 if existing_runs else 1     #把最大的+1，没有就直接是1 (Return max + 1, or 1 if the list is empty)


# 把动态的 csv_base_path 传进来 (Pass in the dynamic csv_base_path)
def make_env(rank, seed, csv_base_path):
    def _init():
        raw_env = SumoEnvironment(
            net_file=net_path,
            route_file=route_path,
            out_csv_name=f"{csv_base_path}_{rank}", # 动态挂载日志名字 (Dynamically mount log filename)
            use_gui=False,
            num_seconds=3600,
            reward_fn=pbrs_reward   #queue 奖励函数 (Queue reward function)
        )
        env = ThreeJunctionCentralizedWrapper(raw_env)
        return env

    return _init


if __name__ == '__main__':

    # 固定全局随机种子 （！！！每次运行修改记录） (Fix global random seed (!!! modify record each run))
    seed = 8848
    random.seed(seed)             # Python 的基础随机 (Python's base random)
    np.random.seed(seed)          # NumPy 的数学计算随机 (NumPy's mathematical random)
    torch.manual_seed(seed)       # PyTorch的随机 (PyTorch's random)
    set_random_seed(seed)         # Stable-Baselines3 (SB3) 框架的随机 (Stable-Baselines3 framework random)

    # 动态分配本次训练的路径 (Dynamically allocate paths for this training run)
    saved_models_base = os.path.join(ROOT_DIR, 'saved_models')
    logs_base = os.path.join(ROOT_DIR, 'logs')
    # 创建目录储存模型和log (Create directories to store models and logs)
    os.makedirs(saved_models_base, exist_ok=True)
    os.makedirs(logs_base, exist_ok=True)

    # 自动计算本次是第几次运行 (Automatically calculate the current run number)
    run_idx = get_next_run_number(saved_models_base, "single_queue_run_")
    print(f"\n 自动检测到历史记录，本次分配为: [ 第 {run_idx} 次运行 ]")

    # 根据编号创建本次运行的专属模型保存文件夹 (Create an exclusive model save folder for this run based on the ID)
    run_save_dir = os.path.join(saved_models_base, f'single_queue_run_{run_idx}')
    os.makedirs(run_save_dir, exist_ok=True)

    # 根据编号创建本次运行的专属 CSV 日志文件夹 (Create an exclusive CSV log folder for this run based on the ID)
    run_csv_dir = os.path.join(logs_base, f'single_queue_run_{run_idx}')
    os.makedirs(run_csv_dir, exist_ok=True)
    csv_base_path = os.path.join(run_csv_dir, 'output')

    # Tensorboard 总目录依然保持不变，在localhost方便把多条曲线画在同一张图里对比 (Keep the main Tensorboard directory unchanged to easily plot multiple curves on the same graph on localhost)
    tensorboard_log_path = os.path.join(logs_base, 'ppo_single_tb')

    num_cpu = 4
    print(f"正在后台启动 {num_cpu} 个并行的 SUMO 环境...")

    # 创建多进程环境，传入基础 seed 和动态生成的 csv 路径 (Create multi-process environments, pass base seed and dynamically generated csv path)
    env = SubprocVecEnv([make_env(i, seed=seed, csv_base_path=csv_base_path) for i in range(num_cpu)])
    env = VecMonitor(env)

    # <==== 【修改点 2】：务必把 norm_reward 改成 False ！！！
    env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.)

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=linear_schedule_with_min(3e-4, 3e-5),
        n_steps=2048,
        batch_size=256,
        n_epochs=10,
        clip_range=0.2,
        ent_coef=0.03,
        target_kl=0.05,
        verbose=1,
        device="cpu",
        seed=seed,
        tensorboard_log=tensorboard_log_path  # 统一定位到 TB 日志主目录 (Uniformly target the TB main log directory)
    )

    print(f"环境就绪！开始单智能体并行训练...")
    # tb_log_name 会在 Tensorboard 目录下自动新建 "run_1", "run_2" 子文件夹 (tb_log_name will auto-create subfolders like "run_1", "run_2" under the Tensorboard directory)
    model.learn(total_timesteps=361000, tb_log_name=f"run_{run_idx}")

    # 文件会自动存到：saved_models/single_queue_run_X/ppo_model.zip (Files will be automatically saved to: saved_models/single_queue_run_X/ppo_model.zip)
    model.save(os.path.join(run_save_dir, "ppo_model"))
    env.save(os.path.join(run_save_dir, "vec_normalize.pkl"))

    env.close()
    print(f"\n 训练完成！所有产出均已安全保存至专属目录:")
    print(f"模型文件: {run_save_dir}")
    print(f"日志文件: {run_csv_dir}")