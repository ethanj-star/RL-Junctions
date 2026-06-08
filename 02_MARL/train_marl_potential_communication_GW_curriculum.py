"""
课程式绿波训练脚本。
Curriculum green-wave training script.

本文件把训练拆成两个阶段：Stage 1 先用更强的主路绿波奖励制造可见的时空图直线，
Stage 2 再提高 PBRS 和支路保护权重，尝试恢复平均等待时间和排队长度。
This script first encourages visible arterial progression, then rebalances delay and side-street service.
"""

import os
import random
import sys
from contextlib import contextmanager
from typing import Callable

import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import VecEnvWrapper, VecMonitor, VecNormalize
from sumo_rl import parallel_env
from sumo_rl.environment.observations import DefaultObservationFunction
import supersuit as ss

"""
课程式绿波训练脚本：先激进塑造主路绿波，再尝试回到平衡控制。
(Curriculum green-wave training: first shape progression, then rebalance.)

该脚本用于验证“先让绿波出现，再恢复等待时间”的实验思路。
(It tests the idea of making progression visible before restoring efficiency.)
"""

# 路径配置：自动定位项目根目录和 SUMO 输入文件。
# (Path setup: locate project root and SUMO input files.)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CURRENT_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from generate_Random_Traffic import generate_route_file


net_path = os.path.join(ROOT_DIR, "SUMOroutes.net.xml")
route_path = os.path.join(ROOT_DIR, "traffic.random.rou.xml")

# 训练随机种子：固定随机性，方便不同 Run 之间对比。
# (Random seed: keep experiments comparable.)
SEED = 8848

# 可选热启动：设置已有 Run 编号后，可从之前模型继续训练。
# (Optional warm start: continue from a previous run if set.)
_load_run = os.environ.get("JUC_LOAD_MODEL_RUN_IDX", "").strip()
LOAD_MODEL_RUN_IDX = int(_load_run) if _load_run else None

# 两阶段训练：Stage 1 偏向主路绿波，Stage 2 恢复 PBRS 和全局效率。
# (Two-stage training: Stage 1 favors progression; Stage 2 restores efficiency.)
STAGE1_TIMESTEPS = int(os.environ.get("JUC_STAGE1_TIMESTEPS", "250000"))
STAGE2_TIMESTEPS = int(os.environ.get("JUC_STAGE2_TIMESTEPS", "350000"))

# 信号灯状态：基于完整灯色字符串判断主路/支路绿灯。
# (Signal states: identify main/side green by full state strings.)
MAIN_GREEN_STATE = os.environ.get("JUC_MAIN_GREEN_STATE", "rrrrGGggrrrrGGgg")
SIDE_GREEN_STATE = os.environ.get("JUC_SIDE_GREEN_STATE", "GGggrrrrGGggrrrr")
SIGNALS = ("A0", "B0", "C0")

# ETA 与自由流参数：描述主路车辆接近停止线和连续通行的程度。
# (ETA/free-flow settings: describe approaching and free-flow arterial vehicles.)
ETA_HORIZON = 22.0
ETA_DECAY = 8.0
NEAR_ETA = 7.0
MIN_ETA_SPEED = 2.0
STOPLINE_DISTANCE = 30.0
FREE_FLOW_SPEED = 5.0
MAX_PRESSURE = 5.0

# 主路进口和相邻关系：用于双向主路压力与路口通信。
# (Main approaches and neighbors: used for bidirectional pressure and communication.)
MAIN_APPROACH_EDGES = {
    "A0": ["left0A0", "B0A0"],
    "B0": ["A0B0", "C0B0"],
    "C0": ["B0C0", "right0C0"],
}

NEIGHBOR_MAP = {
    "A0": [None, "B0"],
    "B0": ["A0", "C0"],
    "C0": ["B0", None],
}

REWARD_STAGE = "aggressive"

# 奖励配置：aggressive 强调绿波，balanced 强调效率和支路保护。
# (Reward profiles: aggressive favors progression; balanced favors efficiency.)
REWARD_PROFILES = {
    "aggressive": {
        "pbrs": 0.25,
        "green_pressure": 0.28,
        "red_pressure": 0.35,
        "all_green": 0.80,
        "free_flow": 0.18,
        "side_queue": 0.005,
    },
    "balanced": {
        "pbrs": 1.00,
        "green_pressure": 0.05,
        "red_pressure": 0.06,
        "all_green": 0.08,
        "free_flow": 0.04,
        "side_queue": 0.025,
    },
}


# 文件目录切换工具：随机交通生成函数需要在项目根目录下运行。
# (Directory helper: route generation expects the project root as cwd.)
@contextmanager
def pushd(path: str):
    old_cwd = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old_cwd)


# 随机交通生成：每个 episode 重新生成交通流，避免只适应单一流量。
# (Traffic generation: regenerate flows for every episode.)
def generate_route_file_in_root():
    with pushd(ROOT_DIR):
        generate_route_file()


# 学习率调度：训练后期保留较小学习率，便于微调。
# (Learning-rate schedule: keep a small floor for fine-tuning.)
def linear_schedule_with_min(initial_value: float, min_value: float) -> Callable[[float], float]:
    def func(progress_remaining: float) -> float:
        return min_value + progress_remaining * (initial_value - min_value)

    return func


# Run 编号工具：自动找到下一个可用输出目录。
# (Run numbering: find the next available output directory.)
def get_next_run_number(base_dir, prefix="marl_run_"):
    if not os.path.exists(base_dir):
        return 1

    existing_runs = []
    for folder in os.listdir(base_dir):
        if folder.startswith(prefix):
            try:
                existing_runs.append(int(folder.replace(prefix, "")))
            except ValueError:
                continue

    return max(existing_runs) + 1 if existing_runs else 1


# 兼容包装器：处理 PettingZoo/Gymnasium/SB3 返回值差异。
# (Compatibility wrapper: handle API differences between libraries.)
class SB3CompatibilityWrapper(VecEnvWrapper):
    def __init__(self, venv):
        super().__init__(venv)

    def reset(self):
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


# 安全调用工具：TraCI 查询失败时使用默认值，防止训练中断。
# (Safe TraCI call: use defaults when TraCI queries fail.)
def safe_call(default, func, *args):
    try:
        return func(*args)
    except Exception:
        return default


# 信号状态查询：读取真实红黄绿灯色字符串。
# (Signal query: read actual red/yellow/green state strings.)
def signal_state(sumo, signal_id: str) -> str:
    return sumo.trafficlight.getRedYellowGreenState(signal_id)


# 主路绿灯判断：统一基于 state 字符串，不依赖 phase 编号。
# (Main green check: use state strings, not phase IDs.)
def is_main_green_state(state: str) -> bool:
    return state == MAIN_GREEN_STATE


# 黄灯判断：黄灯是过渡状态，不直接作为主路通行奖励依据。
# (Yellow check: treat yellow as a transition state.)
def is_yellow_state(state: str) -> bool:
    return "y" in state.lower()


# 基础 PBRS 奖励：提供排队和势能塑形的稳定优化方向。
# (Base PBRS reward: stable queue and potential shaping objective.)
def pbrs_reward(traffic_signal):
    total_queue = traffic_signal.get_total_queued()
    base_reward = -total_queue

    current_light_state = traffic_signal.sumo.trafficlight.getRedYellowGreenState(traffic_signal.id)
    active_phase_queue = 0
    for i, lane in enumerate(traffic_signal.lanes):
        if i < len(current_light_state) and current_light_state[i] in ("G", "g", "y", "Y"):
            active_phase_queue += traffic_signal.sumo.lane.getLastStepHaltingNumber(lane)

    phi_current = active_phase_queue / (total_queue + 1e-6)

    current_step = getattr(traffic_signal.env, "sim_step", 0)
    delta_time = getattr(traffic_signal.env, "delta_time", 5)
    is_new_episode = current_step <= delta_time

    if not hasattr(traffic_signal, "last_potential") or is_new_episode:
        traffic_signal.last_potential = phi_current
        shaping_reward = 0.0
    else:
        gamma = 0.99
        shaping_reward = gamma * phi_current - traffic_signal.last_potential
        traffic_signal.last_potential = phi_current

    return (base_reward + 100.0 * shaping_reward) / 100.0


# 主路进口指标：统计 ETA 压力、近端车辆和自由流车辆。
# (Approach metrics: compute ETA pressure, near vehicles, and free-flow count.)
def edge_progression_metrics(sumo, edge_id: str) -> dict:
    lane_id = f"{edge_id}_0"
    lane_length = float(safe_call(0.0, sumo.lane.getLength, lane_id))
    vehicle_ids = list(safe_call([], sumo.lane.getLastStepVehicleIDs, lane_id))

    pressure = 0.0
    near_count = 0.0
    free_flow_count = 0.0

    for veh_id in vehicle_ids:
        lane_pos = float(safe_call(0.0, sumo.vehicle.getLanePosition, veh_id))
        speed = float(safe_call(0.0, sumo.vehicle.getSpeed, veh_id))
        distance_to_stopline = max(lane_length - lane_pos, 0.0)
        eta = distance_to_stopline / max(speed, MIN_ETA_SPEED)

        if eta <= ETA_HORIZON:
            pressure += np.exp(-eta / ETA_DECAY)
            if eta <= NEAR_ETA:
                near_count += 1.0

        if distance_to_stopline <= STOPLINE_DISTANCE and speed >= FREE_FLOW_SPEED:
            free_flow_count += 1.0

    return {
        "pressure": float(min(pressure, MAX_PRESSURE)),
        "near_count": float(near_count),
        "free_flow_count": float(free_flow_count),
    }


# 路口主路压力：聚合当前信号灯两侧主路进口的压力。
# (Main-road pressure: aggregate both arterial approaches of the signal.)
def main_arrival_pressure(traffic_signal) -> dict:
    pressure = 0.0
    near_count = 0.0
    free_flow_count = 0.0

    for edge_id in MAIN_APPROACH_EDGES.get(traffic_signal.id, []):
        metrics = edge_progression_metrics(traffic_signal.sumo, edge_id)
        pressure += metrics["pressure"]
        near_count += metrics["near_count"]
        free_flow_count += metrics["free_flow_count"]

    return {
        "pressure": float(min(pressure, MAX_PRESSURE)),
        "near_count": float(near_count),
        "free_flow_count": float(min(free_flow_count, 6.0)),
    }


# 全走廊主路绿灯判断：用于激进阶段奖励三个路口同步主路绿。
# (Corridor green check: reward synchronized main green in aggressive stage.)
def all_signals_main_green(traffic_signal) -> bool:
    for ts_id in SIGNALS:
        state = signal_state(traffic_signal.sumo, ts_id)
        if not is_main_green_state(state):
            return False
    return True


# 支路排队统计：用于防止支路长期被主路绿波压制。
# (Side queue: prevent side streets from being starved.)
def side_queue(traffic_signal) -> float:
    queue = 0.0
    for lane in traffic_signal.lanes:
        if "top" in lane or "bottom" in lane:
            queue += safe_call(0, traffic_signal.sumo.lane.getLastStepHaltingNumber, lane)
    return float(queue)


# 课程式奖励：根据当前阶段切换绿波塑形和效率恢复的权重。
# (Curriculum reward: switch weights between progression and balanced stages.)
def green_wave_curriculum_reward(traffic_signal):
    # 1. 根据当前训练阶段选择奖励权重 (Select reward profile by stage)
    # aggressive 阶段偏向主路绿波；balanced 阶段偏向 PBRS、等待时间和支路公平性。
    profile = REWARD_PROFILES[REWARD_STAGE]

    # 2. PBRS 是基础效率目标 (Base efficiency objective)
    # profile["pbrs"] 控制排队优化在总奖励中的比例，防止模型只追求时空图直线。
    base = profile["pbrs"] * pbrs_reward(traffic_signal)

    # 3. 使用真实灯色 state 判断主路绿灯 (Use signal state string)
    # 不依赖 phase 编号，因为 phase=0 不一定在所有脚本里都能安全代表主路绿灯。
    state = signal_state(traffic_signal.sumo, traffic_signal.id)
    is_main_green = is_main_green_state(state)
    is_yellow = is_yellow_state(state)

    # 4. 汇总当前路口两个主路进口的 ETA 压力。
    # pressure 越大，说明越多主路车即将到达停止线；free_flow_count 表示较顺畅通过的车辆数。
    metrics = main_arrival_pressure(traffic_signal)
    pressure = metrics["pressure"]

    progression = 0.0
    if pressure > 0:
        if is_main_green:
            # 主路有来车且当前为主路绿灯：奖励绿波对齐。
            progression += profile["green_pressure"] * pressure
            # 自由流车辆越多，时空图越可能出现不停顿的斜直线。
            progression += profile["free_flow"] * metrics["free_flow_count"]
        elif not is_yellow:
            # 主路有来车但当前非主路绿灯：惩罚可能造成停车的相位。
            progression -= profile["red_pressure"] * pressure

    if all_signals_main_green(traffic_signal):
        # 5. 走廊同步主路绿灯奖励。
        # 该项会推动 A0/B0/C0 同时保持主路绿灯，能增强绿波可视化效果，
        # 但权重过大时也容易压制支路，因此后续阶段需要 balanced 恢复。
        corridor_pressure = 0.0
        for ts in traffic_signal.env.traffic_signals.values():
            corridor_pressure += main_arrival_pressure(ts)["pressure"]
        corridor_pressure = min(corridor_pressure / 6.0, 1.0)
        progression += profile["all_green"] * max(corridor_pressure, 0.25)

    # 6. 支路公平性惩罚。
    # 支路排队超过 10 辆后开始扣分，用来限制主路长期绿灯造成的支路饿死。
    fairness = -profile["side_queue"] * max(0.0, side_queue(traffic_signal) - 10.0)

    return base + progression + fairness


# 通信观测：基础观测 + 相邻路口主路排队。
# (Communication observation: base observation plus neighboring arterial queues.)
class CommObservationFunction(DefaultObservationFunction):
    """
    Same compact observation as train_marl_potential_communication.py:
    base observation + two neighbor main-road queue values.
    """

    def __init__(self, ts):
        super().__init__(ts)

    def observation_space(self):
        base_space = super().observation_space()
        new_dim = base_space.shape[0] + 2
        return spaces.Box(
            low=np.zeros(new_dim, dtype=np.float32),
            high=np.ones(new_dim, dtype=np.float32) * np.inf,
        )

    def __call__(self):
        base_obs = super().__call__()
        neighbors = NEIGHBOR_MAP.get(self.ts.id, [None, None])

        extra_obs = []
        for neighbor_id in neighbors:
            if neighbor_id is None:
                extra_obs.append(0.0)
                continue

            neighbor_ts = self.ts.env.traffic_signals.get(neighbor_id)
            if neighbor_ts is None:
                extra_obs.append(0.0)
                continue

            main_arterial_queue = 0
            for lane in neighbor_ts.lanes:
                if "top" not in lane and "bottom" not in lane:
                    main_arterial_queue += neighbor_ts.sumo.lane.getLastStepHaltingNumber(lane)

            extra_obs.append(float(main_arterial_queue))

        return np.array(np.concatenate([base_obs, extra_obs]), dtype=np.float32)


# 热启动路径查找：从指定 Run 中读取模型和 VecNormalize。
# (Warm-start lookup: load model and VecNormalize from a selected run.)
def find_warm_start_paths():
    if LOAD_MODEL_RUN_IDX is None:
        return None, None

    source_dir = os.path.join(ROOT_DIR, "saved_models", f"marl_run_{LOAD_MODEL_RUN_IDX}")

    model_path = os.path.join(source_dir, "ppo_marl_model.zip")
    if not os.path.exists(model_path):
        model_path = os.path.join(source_dir, "ppo_marl_model_gw_lite.zip")
    if not os.path.exists(model_path):
        model_path = None

    vec_norm_path = os.path.join(source_dir, "vec_normalize_marl.pkl")
    if not os.path.exists(vec_norm_path):
        vec_norm_path = os.path.join(source_dir, "vec_normalize_marl_gw_lite.pkl")
    if not os.path.exists(vec_norm_path):
        vec_norm_path = None

    if model_path is None:
        raise FileNotFoundError(f"Cannot find warm-start model in: {source_dir}")

    return model_path, vec_norm_path


# 环境构建：创建 SUMO-RL 环境，并在每回合重生成随机交通。
# (Environment builder: create SUMO-RL env and regenerate traffic per episode.)
def build_env(csv_base_path, vec_norm_path=None):
    # 1. 创建 SUMO-RL 多智能体环境。
    # reward_fn 使用课程式奖励函数，observation_class 使用紧凑通信观测。
    env = parallel_env(
        net_file=net_path,
        route_file=route_path,
        out_csv_name=csv_base_path,
        use_gui=False,
        num_seconds=3600,
        reward_fn=green_wave_curriculum_reward,
        observation_class=CommObservationFunction,
        min_green=15,
        max_green=70,
    )

    original_reset = env.reset

    def custom_reset(*args, **kwargs):
        # 每个 episode 开始前重生成随机交通流，避免模型只适配固定 route 文件。
        print("Regenerating random bidirectional traffic for this episode...")
        generate_route_file_in_root()
        return original_reset(*args, **kwargs)

    env.reset = custom_reset
    # 2. PettingZoo parallel_env 转换为 SB3 PPO 可使用的向量环境。
    # concat_vec_envs_v1 即使只使用 1 个环境，也能让接口符合 Stable-Baselines3。
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, num_vec_envs=1, num_cpus=1, base_class="stable_baselines3")
    env = SB3CompatibilityWrapper(env)
    env = VecMonitor(env)

    if vec_norm_path is not None:
        # 3A. 继续训练时加载旧 VecNormalize 统计量。
        # 这样观测均值/方差与源模型保持一致，避免热启动后模型突然“看不懂”环境。
        print(f"Loading VecNormalize statistics from: {vec_norm_path}")
        env = VecNormalize.load(vec_norm_path, env)
        env.training = True
        env.norm_reward = False
        return env

    # 3B. 从零训练时新建 VecNormalize。
    # norm_obs=True 归一化观测；norm_reward=False 保留真实奖励尺度用于对比。
    return VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.0)


# 模型加载：如果提供热启动则继续训练，否则新建 PPO。
# (Model loader: warm-start PPO if available, otherwise create a new model.)
def maybe_load_model(env, model_path=None):
    if model_path is None:
        # 没有可用源模型时创建新 PPO。
        # ent_coef 用于鼓励探索，target_kl 用于限制每轮策略更新幅度。
        return PPO(
            "MlpPolicy",
            env,
            learning_rate=linear_schedule_with_min(3e-4, 3e-5),
            n_steps=2048,
            batch_size=256,
            n_epochs=10,
            clip_range=0.2,
            ent_coef=0.02,
            target_kl=0.05,
            verbose=1,
            device="cpu",
            tensorboard_log=os.path.join(ROOT_DIR, "logs", "ppo_marl_tb"),
        )

    print(f"Warm-starting from: {model_path}")
    # 从已有模型继续训练：PPO.load 会恢复策略网络和值函数网络参数。
    return PPO.load(model_path, env=env, tensorboard_log=os.path.join(ROOT_DIR, "logs", "ppo_marl_tb"))


# 主程序：依次执行激进绿波阶段和平衡恢复阶段，并保存阶段模型。
# (Main entry: run aggressive and balanced stages, then save models.)
if __name__ == "__main__":
    print("Initializing MARL green-wave curriculum training...")

    # 1. 固定随机种子，保证同一脚本多次运行时尽量可复现。
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    saved_models_base = os.path.join(ROOT_DIR, "saved_models")
    logs_base = os.path.join(ROOT_DIR, "logs")
    os.makedirs(saved_models_base, exist_ok=True)
    os.makedirs(logs_base, exist_ok=True)

    run_idx = get_next_run_number(saved_models_base, "marl_run_")
    print(f"Detected run index: {run_idx}")

    run_save_dir = os.path.join(saved_models_base, f"marl_run_{run_idx}")
    os.makedirs(run_save_dir, exist_ok=True)
    run_csv_dir = os.path.join(logs_base, f"marl_run_{run_idx}")
    os.makedirs(run_csv_dir, exist_ok=True)

    csv_base_path = os.path.join(run_csv_dir, "marl_output")

    print("Generating initial random bidirectional traffic...")
    generate_route_file_in_root()

    # 2. 查找热启动模型和 VecNormalize。
    # 如果 LOAD_MODEL_RUN_IDX 为 None，则从零训练；否则从指定 Run 接着训练。
    warm_model_path, warm_vec_norm_path = find_warm_start_paths()
    env = build_env(csv_base_path, warm_vec_norm_path)
    model = maybe_load_model(env, warm_model_path)

    print("=" * 70)
    print("Observation space:", model.policy.observation_space)
    print("Action space:", model.policy.action_space)
    print(f"Main-road green state: {MAIN_GREEN_STATE}")
    print(f"Stage 1 timesteps: {STAGE1_TIMESTEPS}, profile: aggressive")
    print(f"Stage 2 timesteps: {STAGE2_TIMESTEPS}, profile: balanced")
    print("=" * 70)

    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path=os.path.join(run_save_dir, "checkpoints"),
        name_prefix="rl_model",
    )

    # 3. Stage 1：强绿波塑形。
    # 该阶段优先鼓励主路同步绿灯和自由流通过，用来观察时空图能否先出现直线。
    REWARD_STAGE = "aggressive"
    print("Stage 1: aggressive green-wave shaping...")
    model.learn(
        total_timesteps=STAGE1_TIMESTEPS,
        callback=checkpoint_callback,
        tb_log_name=f"run_{run_idx}_gw_curriculum_stage1",
        reset_num_timesteps=True,
    )
    # 保存 Stage 1 中间模型，便于单独测试“激进绿波”对时空图的影响。
    model.save(os.path.join(run_save_dir, "ppo_marl_model_gw_stage1"))
    env.save(os.path.join(run_save_dir, "vec_normalize_marl_gw_stage1.pkl"))

    # 4. Stage 2：平衡恢复。
    # 降低绿波项，恢复 PBRS 和支路保护，尝试降低平均等待时间。
    REWARD_STAGE = "balanced"
    print("Stage 2: balanced fine-tuning back to PBRS...")
    model.learn(
        total_timesteps=STAGE2_TIMESTEPS,
        callback=checkpoint_callback,
        tb_log_name=f"run_{run_idx}_gw_curriculum_stage2",
        reset_num_timesteps=False,
    )

    # 5. 保存最终模型。
    # 通用文件名兼容旧测试脚本；gw_curriculum 文件名用于区分该实验阶段。
    model.save(os.path.join(run_save_dir, "ppo_marl_model"))
    env.save(os.path.join(run_save_dir, "vec_normalize_marl.pkl"))
    model.save(os.path.join(run_save_dir, "ppo_marl_model_gw_curriculum"))
    env.save(os.path.join(run_save_dir, "vec_normalize_marl_gw_curriculum.pkl"))

    env.close()
    print(f"Curriculum training finished. Outputs saved to: {run_save_dir}")
