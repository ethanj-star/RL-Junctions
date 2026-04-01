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

# ==========================================
# 路径动态获取 (永远不用改)
# ==========================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CURRENT_DIR)
net_path = os.path.join(ROOT_DIR, 'SUMOroutes.net.xml')
route_path = os.path.join(ROOT_DIR, 'traffic.rou.rou.xml')


# ==========================================
# 奖励函数定义 (确保和上次训练时一模一样，保持记忆连贯)
# ==========================================
def pbrs_reward(traffic_signal):
    total_queue = traffic_signal.get_total_queued()
    base_reward = -total_queue

    current_light_state = traffic_signal.sumo.trafficlight.getRedYellowGreenState(traffic_signal.id)
    active_phase_queue = 0
    for i, lane in enumerate(traffic_signal.lanes):
        if current_light_state[i] in ('G', 'g', 'y', 'Y'):
            active_phase_queue += traffic_signal.sumo.lane.getLastStepHaltingNumber(lane)

    phi_current = active_phase_queue / (total_queue + 1e-6)

    if not hasattr(traffic_signal, 'last_potential'):
        traffic_signal.last_potential = 0.0

    gamma = 0.99
    shaping_reward = (gamma * phi_current) - traffic_signal.last_potential
    traffic_signal.last_potential = phi_current

    beta = 300.0
    final_reward = base_reward + (beta * shaping_reward)
    return final_reward / 100.0


# 自动编号器 (寻找最新的文件夹)
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


# 环境构建器
def make_env(rank, seed, csv_base_path):
    def _init():
        raw_env = SumoEnvironment(
            net_file=net_path,
            route_file=route_path,
            out_csv_name=f"{csv_base_path}_{rank}",
            use_gui=False,
            num_seconds=3600,
            reward_fn=pbrs_reward
        )
        env = ThreeJunctionCentralizedWrapper(raw_env)
        return env

    return _init


if __name__ == '__main__':

    # ========================================================================= #
    # ===================== 🛑 每次运行前，只需修改以下 3 个参数 🛑 ==================== #
    # ========================================================================= #

    # 1. 你要继承的那个旧模型的文件夹名字（比如上次跑到了第 5 次，就写 'single_queue_run_5'）
    OLD_RUN_FOLDER_NAME = 'single_queue_run_6'

    # 2. 你这次打算【额外】再训练多少步？(比如再训 150 万步)
    ADDITIONAL_TIMESTEPS = 1500000

    # 3. 开启多少个进程跑？(根据你电脑 CPU 性能设定，8 核就写 8)
    NUM_CPU = 8

    # ========================================================================= #
    # ===================== 🛑 修改区结束，下面的代码永远不用动 🛑 ==================== #
    # ========================================================================= #

    print(f"\n🚀 初始化断点续训脚本...")
    print(f"-> 目标继承文件夹: {OLD_RUN_FOLDER_NAME}")
    print(f"-> 计划追加步数: {ADDITIONAL_TIMESTEPS}")

    seed = 8848
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    set_random_seed(seed)

    saved_models_base = os.path.join(ROOT_DIR, 'saved_models')
    logs_base = os.path.join(ROOT_DIR, 'logs')
    os.makedirs(saved_models_base, exist_ok=True)
    os.makedirs(logs_base, exist_ok=True)

    # 1. 自动计算本次续训要存入的“新”文件夹编号 (防止覆盖老模型)
    run_idx = get_next_run_number(saved_models_base, "single_queue_run_")
    print(f"-> 本次续训产出将被安全保存在: [ 第 {run_idx} 次运行 ]")

    run_save_dir = os.path.join(saved_models_base, f'single_queue_run_{run_idx}')
    os.makedirs(run_save_dir, exist_ok=True)
    run_csv_dir = os.path.join(logs_base, f'single_queue_run_{run_idx}')
    os.makedirs(run_csv_dir, exist_ok=True)
    csv_base_path = os.path.join(run_csv_dir, 'output')
    tensorboard_log_path = os.path.join(logs_base, 'ppo_single_tb')

    print(f"正在后台启动 {NUM_CPU} 个并行的 SUMO 环境...")
    env = SubprocVecEnv([make_env(i, seed=seed, csv_base_path=csv_base_path) for i in range(NUM_CPU)])
    env = VecMonitor(env)

    # 2. 定位旧模型所在的完整路径
    old_run_dir_path = os.path.join(saved_models_base, OLD_RUN_FOLDER_NAME)
    stats_path = os.path.join(old_run_dir_path, "vec_normalize.pkl")
    model_path = os.path.join(old_run_dir_path, "ppo_model.zip")

    if not os.path.exists(stats_path) or not os.path.exists(model_path):
        raise FileNotFoundError(
            f"\n❌ 灾难性错误：在 {old_run_dir_path} 里找不到旧的 ppo_model.zip 或 vec_normalize.pkl，请检查文件夹名字是否写对！")

    # 3. 加载 AI 的旧“眼镜” (状态归一化参数)
    print(f"正在加载历史状态参数...")
    env = VecNormalize.load(stats_path, env)

    # 强制将眼镜切回训练模式，且坚决保持奖励归一化关闭！
    env.training = True
    env.norm_reward = False

    # 4. 加载 AI 的旧“大脑” (PPO 模型)
    print(f"正在加载历史模型权重...")
    model = PPO.load(
        model_path,
        env=env,
        tensorboard_log=tensorboard_log_path,
        device="cpu"
    )

    print(f"✅ 环境与模型就绪！开始追加训练...")

    # 5. 开启无缝续训
    # reset_num_timesteps=False 保证 TensorBoard 曲线平滑连接！
    model.learn(
        total_timesteps=ADDITIONAL_TIMESTEPS,
        tb_log_name=f"run_{run_idx}_resumed",
        reset_num_timesteps=False
    )

    # 6. 保存新的大成模型
    model.save(os.path.join(run_save_dir, "ppo_model"))
    env.save(os.path.join(run_save_dir, "vec_normalize.pkl"))

    env.close()
    print(f"\n🎉 续训大功告成！所有产出均已安全保存至:")
    print(f"新模型文件夹: {run_save_dir}")
    print(f"去 TensorBoard 看看连贯的曲线吧！")