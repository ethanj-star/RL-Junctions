from pathlib import Path
from typing import Optional
import random
import importlib.util
import os

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import VecMonitor, VecNormalize, VecEnvWrapper

# 路径统一指向正式代码根目录下的共享 envs，模型和日志保存在当前 PBRS 文件夹内部。
# Paths use the shared envs folder under the formal code root, while models and logs stay local to this PBRS folder.
BASE_DIR = Path(__file__).resolve().parent
FORMAL_ROOT = BASE_DIR.parent.parent
ENV_DIR = FORMAL_ROOT / "envs"
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"
NET_FILE = ENV_DIR / "SUMOroutes.net.xml"
ROUTE_FILE = ENV_DIR / "traffic.random.rou.xml"
RUN_PREFIX = "fixlr_m03t_total_queue_run_"

# 随机车流生成脚本按文件路径加载，避免编辑器无法解析动态 sys.path 导入。
# The random traffic generator is loaded by file path so editors do not complain about dynamic sys.path imports.
def load_generate_route_file():
    generator_path = ENV_DIR / "generate_Random_Traffic.py"
    spec = importlib.util.spec_from_file_location("formal_generate_random_traffic", generator_path)
    if spec is None or spec.loader is None:
        raise FileNotFoundError(f"Cannot load traffic generator: {generator_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, "generate_route_file")


generate_route_file = load_generate_route_file()

# 训练参数可由顺序控制器传入；未传入环境变量时保留手动运行默认值。
# The sequential controller can override these settings; manual defaults remain available.
RUN_IDX = int(os.environ["JUC_RUN_IDX"]) if os.environ.get("JUC_RUN_IDX") else None
SEED = int(os.environ.get("JUC_SEED", "8848"))
TOTAL_TIMESTEPS = int(os.environ.get("JUC_TOTAL_TIMESTEPS", "300000"))
NUM_SECONDS = 3600
GAMMA = 0.99
BETA = 1.0
REWARD_NORMALIZER = 100.0


# 固定学习率统一为 3e-5，保证与固定学习率基线公平比较。
# The fixed learning rate is 3e-5 for fair comparison with the fixed-LR baseline.
# 这个 wrapper 兼容 Gymnasium 新版 5 返回值和 Stable-Baselines3 旧版 4 返回值接口。
# This wrapper adapts Gymnasium's newer five-return API to the four-return API expected by Stable-Baselines3.
class SB3CompatibilityWrapper(VecEnvWrapper):
    def __init__(self, venv, reset_callback=None):
        super().__init__(venv)
        self.reset_callback = reset_callback

    def reset(self):
        if self.reset_callback is not None:
            self.reset_callback()
        obs = self.venv.reset()
        if isinstance(obs, tuple) and len(obs) == 2:
            return obs[0]
        return obs

    # SUMO 与车流已显式 seed；此接口仅补齐 SuperSuit 3.7.1 缺失的 SB3 协议。
    # SUMO and traffic are seeded explicitly; this completes the SB3 protocol missing in SuperSuit 3.7.1.
    def seed(self, seed=None):
        return [None if seed is None else seed + index for index in range(self.num_envs)]

    def step_async(self, actions):
        self.venv.step_async(actions)

    def step_wait(self):
        results = self.venv.step_wait()
        if len(results) == 5:
            obs, rewards, terminated, truncated, infos = results
            dones = np.logical_or(terminated, truncated)
            return obs, rewards, dones, infos
        return results


# 自动查找下一个 run 编号，避免 total-queue PBRS 的模型和日志覆盖已有实验。
# This finds the next run number so total-queue PBRS models and logs do not overwrite existing experiments.
def next_run_number() -> int:
    run_ids = []
    for path in MODELS_DIR.glob(f"{RUN_PREFIX}*"):
        suffix = path.name.removeprefix(RUN_PREFIX)
        if suffix.isdigit():
            run_ids.append(int(suffix))
    return max(run_ids, default=0) + 1


# total-queue PBRS 直接用整个路口总排队数定义势能，排队越少势能越高。
# Total-queue PBRS defines potential from total intersection queue, so lower queue means higher potential.
def pbrs_reward(traffic_signal):
    total_queue = traffic_signal.get_total_queued()
    base_reward = -total_queue
    phi_current = -float(total_queue)

    current_step = getattr(traffic_signal.env, "sim_step", 0)
    delta_time = getattr(traffic_signal.env, "delta_time", 5)
    is_new_episode = current_step <= delta_time

    if not hasattr(traffic_signal, "last_potential") or is_new_episode:
        traffic_signal.last_potential = phi_current
        shaping_reward = 0.0
    else:
        shaping_reward = GAMMA * phi_current - traffic_signal.last_potential
        traffic_signal.last_potential = phi_current

    return (base_reward + BETA * shaping_reward) / REWARD_NORMALIZER


# 创建 PettingZoo MARL 环境，三个路口分别作为 agent，reward 使用 total-queue PBRS。
# This creates the PettingZoo MARL environment where each intersection is an agent using total-queue PBRS.
def make_pettingzoo_env(csv_base_path: Path, use_gui: bool):
    from sumo_rl import parallel_env

    route_file = csv_base_path.parent / "traffic_train.rou.xml"
    episode_index = 1
    first_reset = True
    generate_route_file(seed=SEED, output_file=route_file)
    env = parallel_env(
        net_file=str(NET_FILE),
        route_file=str(route_file),
        out_csv_name=str(csv_base_path),
        use_gui=use_gui,
        num_seconds=NUM_SECONDS,
        reward_fn=pbrs_reward,
        sumo_seed=SEED,
    )
    # 首次 reset 使用预生成车流；之后按递增 seed 为每个 episode 生成新车流。
    # The first reset uses the prepared traffic; later resets regenerate traffic with increasing seeds.
    def prepare_episode_traffic():
        nonlocal episode_index, first_reset
        if first_reset:
            first_reset = False
            return
        generate_route_file(seed=SEED + episode_index, output_file=route_file)
        episode_index += 1

    return env, prepare_episode_traffic


# SuperSuit 将 PettingZoo 多智能体环境转换为 SB3 可训练的向量环境。
# SuperSuit converts the PettingZoo multi-agent environment into an SB3-compatible vector environment.
def make_vec_env(csv_base_path: Path, use_gui: bool = False):
    import supersuit as ss

    env, reset_callback = make_pettingzoo_env(csv_base_path, use_gui)
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, num_vec_envs=1, num_cpus=1, base_class="stable_baselines3")
    env = SB3CompatibilityWrapper(env, reset_callback)
    return VecMonitor(env)


# 主训练流程：设置随机种子、创建本地输出目录、训练 PPO 并保存模型和归一化状态。
# Main training flow: set random seeds, create local output folders, train PPO, and save the model plus normalization state.
def train(run_idx: Optional[int] = RUN_IDX) -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    MODELS_DIR.mkdir(exist_ok=True)
    LOGS_DIR.mkdir(exist_ok=True)
    run_idx = next_run_number() if run_idx is None else run_idx
    model_dir = MODELS_DIR / f"{RUN_PREFIX}{run_idx}"
    log_dir = LOGS_DIR / f"{RUN_PREFIX}{run_idx}"
    model_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    env = make_vec_env(log_dir / "marl_output", use_gui=False)
    env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.0)

    # 当前版本使用 total-queue PBRS 和固定学习率 3e-5。
    # This version uses total-queue PBRS with a fixed learning rate of 3e-5.
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-5,
        n_steps=2048,
        batch_size=256,
        n_epochs=10,
        clip_range=0.2,
        ent_coef=0.03,
        target_kl=0.05,
        verbose=1,
        device="cpu",
        seed=SEED,
        tensorboard_log=str(LOGS_DIR / "ppo_marl_total_queue_tb"),
    )
    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path=str(model_dir / "checkpoints"),
        name_prefix="rl_model",
        save_vecnormalize=True,
    )
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=checkpoint_callback,
        tb_log_name=f"total_queue_run_{run_idx}",
    )
    model.save(str(model_dir / "ppo_marl_model"))
    env.save(str(model_dir / "vec_normalize_marl.pkl"))
    env.close()
    print(f"Saved model to {model_dir}")
    print(f"Saved logs to {log_dir}")


if __name__ == "__main__":
    train()
