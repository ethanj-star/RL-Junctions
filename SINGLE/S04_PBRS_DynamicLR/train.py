from pathlib import Path
from typing import Callable, Optional
import os
import random
import sys
import time

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, VecNormalize
from sumo_rl import SumoEnvironment

# 路径统一指向正式代码根目录下的共享 envs，模型和日志保存在本模块内部。
# Paths use the shared envs folder under the formal code root, while models and logs stay local to this module.
BASE_DIR = Path(__file__).resolve().parent
FORMAL_ROOT = BASE_DIR.parent.parent
ENV_DIR = FORMAL_ROOT / "envs"
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"
NET_FILE = ENV_DIR / "SUMOroutes.net.xml"
ROUTE_FILE = ENV_DIR / "traffic.random.rou.xml"
RUN_PREFIX = "single_pbrs_dynamic_lr_run_"

# 随机车流生成器只从共享 envs 引用，不在模块内复制环境实现。
# The random traffic generator is imported from shared envs without duplicating environment code.
sys.path.insert(0, str(ENV_DIR))
from generate_Random_Traffic import generate_route_file

# 训练参数可由顺序控制器传入；未传入环境变量时保留手动运行默认值。
# The sequential controller can override these settings; manual defaults remain available.
RUN_IDX = int(os.environ["JUC_RUN_IDX"]) if os.environ.get("JUC_RUN_IDX") else None
SEED = int(os.environ.get("JUC_SEED", "8848"))
TOTAL_TIMESTEPS = int(os.environ.get("JUC_TOTAL_TIMESTEPS", "1000000"))
NUM_SECONDS = 3600
NUM_CPU = 4
WORKER_START_DELAY = 0.5

# 动态学习率从 3e-4 线性下降到 3e-5，用于和固定学习率版本做对比。
# The dynamic learning rate decays linearly from 3e-4 to 3e-5 for comparison with the fixed-LR version.
def linear_schedule_with_min(initial_value: float, min_value: float) -> Callable[[float], float]:
    def schedule(progress_remaining: float) -> float:
        return min_value + progress_remaining * (initial_value - min_value)

    return schedule
# 该 wrapper 将三个路口合并成一个集中式 single-agent 控制问题。
# This wrapper converts the three intersections into one centralized single-agent control problem.
class ThreeJunctionCentralizedWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.agents = ["A0", "B0", "C0"]
        self.action_space = spaces.MultiDiscrete([2, 2, 2])
        single_obs_space = env.observation_space
        self.observation_space = spaces.Box(
            low=np.tile(single_obs_space.low, 3),
            high=np.tile(single_obs_space.high, 3),
            shape=(33,),
            dtype=np.float32,
        )

    def reset(self, seed=None, options=None):
        obs_dict = self.env.reset()
        return np.concatenate([obs_dict[agent] for agent in self.agents]), {}

    def step(self, action_array):
        action_dict = dict(zip(self.agents, action_array))
        next_obs_dict, reward_dict, done_dict, info_dict = self.env.step(action_dict)
        next_obs = np.concatenate([next_obs_dict[agent] for agent in self.agents])
        reward = sum(reward_dict.values()) / len(self.agents)
        terminated = done_dict["__all__"] if isinstance(done_dict, dict) else done_dict
        return next_obs, reward, terminated, False, info_dict


# PBRS reward 在基础排队惩罚上加入势能塑形项，用于鼓励当前绿灯服务排队车辆。
# The PBRS reward adds a potential-based shaping term to the queue penalty to encourage green phases that serve queued vehicles.
def pbrs_reward(traffic_signal) -> float:
    total_queue = traffic_signal.get_total_queued()
    base_reward = -total_queue
    light_state = traffic_signal.sumo.trafficlight.getRedYellowGreenState(traffic_signal.id)
    active_phase_queue = 0
    for index, lane in enumerate(traffic_signal.lanes):
        if light_state[index] in ("G", "g", "Y", "y"):
            active_phase_queue += traffic_signal.sumo.lane.getLastStepHaltingNumber(lane)

    phi_current = active_phase_queue / (total_queue + 1e-6)
    current_step = getattr(traffic_signal.env, "sim_step", 0)
    delta_time = getattr(traffic_signal.env, "delta_time", 5)
    is_new_episode = current_step <= delta_time

    # 每个回合首步只初始化势能，不产生人为 shaping bonus。
    # The first step only initializes the potential and produces no artificial shaping bonus.
    if not hasattr(traffic_signal, "last_potential") or is_new_episode:
        traffic_signal.last_potential = phi_current
        shaping_reward = 0.0
    else:
        gamma = 0.99
        shaping_reward = gamma * phi_current - traffic_signal.last_potential
        traffic_signal.last_potential = phi_current

    beta = 100.0
    return (base_reward + beta * shaping_reward) / 100.0


# 自动查找下一个 run 编号，避免新模型和日志覆盖已有实验。
# This finds the next run number automatically so new models and logs do not overwrite existing experiments.
def next_run_number() -> int:
    run_ids = []
    for path in MODELS_DIR.glob(f"{RUN_PREFIX}*"):
        suffix = path.name.removeprefix(RUN_PREFIX)
        if suffix.isdigit():
            run_ids.append(int(suffix))
    return max(run_ids, default=0) + 1


# 每个 worker 使用独立 route 文件，并按训练 seed、worker 和 episode 生成可复现车流。
# Each worker uses its own route file and derives reproducible traffic from the training seed, worker, and episode.
def make_env(rank: int, csv_base_path: Path):
    route_file = csv_base_path.parent / f"traffic_worker_{rank}.rou.xml"

    def init_env():
        # 错开 SUMO 子进程启动，避免并行 worker 同时申请同一个 TraCI 端口。
        # Stagger SUMO startup so parallel workers do not claim the same TraCI port.
        time.sleep(rank * WORKER_START_DELAY)
        episode_index = 0

        def generate_episode_traffic():
            traffic_seed = SEED + rank * 1_000_000 + episode_index
            generate_route_file(seed=traffic_seed, output_file=route_file)

        generate_episode_traffic()
        raw_env = SumoEnvironment(
            net_file=str(NET_FILE),
            route_file=str(route_file),
            out_csv_name=f"{csv_base_path}_{rank}",
            use_gui=False,
            num_seconds=NUM_SECONDS,
            reward_fn=pbrs_reward,
            sumo_seed=SEED + rank,
        )
        env = ThreeJunctionCentralizedWrapper(raw_env)
        original_reset = env.reset

        # 首次 reset 使用已生成文件，后续 reset 再推进 episode seed。
        # The first reset uses the prepared file; later resets advance the episode seed.
        def reset_with_new_traffic(*args, **kwargs):
            nonlocal episode_index
            if episode_index:
                generate_episode_traffic()
            time.sleep(rank * WORKER_START_DELAY)
            result = original_reset(*args, **kwargs)
            episode_index += 1
            return result

        env.reset = reset_with_new_traffic
        return env

    return init_env


# 主训练流程：设置随机种子、创建本地输出目录、训练 PPO 并保存模型和归一化状态。
# Main training flow: set random seeds, create local output folders, train PPO, and save the model plus normalization state.
def train(run_idx: Optional[int] = RUN_IDX) -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    set_random_seed(SEED)

    MODELS_DIR.mkdir(exist_ok=True)
    LOGS_DIR.mkdir(exist_ok=True)
    run_idx = next_run_number() if run_idx is None else run_idx
    model_dir = MODELS_DIR / f"{RUN_PREFIX}{run_idx}"
    log_dir = LOGS_DIR / f"{RUN_PREFIX}{run_idx}"
    model_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    env = SubprocVecEnv([make_env(i, log_dir / "output") for i in range(NUM_CPU)])
    env = VecMonitor(env)
    env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.0)

    # 当前版本使用动态学习率，用于比较学习率策略对训练表现的影响。
    # This version uses a dynamic learning rate to compare how the learning-rate strategy affects training performance.
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
        seed=SEED,
        tensorboard_log=str(LOGS_DIR / "ppo_single_tb"),
    )
    checkpoint_callback = CheckpointCallback(
        save_freq=max(50_000 // NUM_CPU, 1),
        save_path=str(model_dir / "checkpoints"),
        name_prefix="rl_model",
        save_vecnormalize=True,
    )
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=checkpoint_callback, tb_log_name=f"run_{run_idx}")
    model.save(str(model_dir / "ppo_model"))
    env.save(str(model_dir / "vec_normalize.pkl"))
    env.close()
    print(f"Saved model to {model_dir}")
    print(f"Saved logs to {log_dir}")


if __name__ == "__main__":
    train()
