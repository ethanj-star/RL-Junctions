"""
绿波连续通行微调脚本。
Green-wave progression fine-tuning script.

本文件用于在 recovery 模型基础上继续优化主路连续通行。
It fine-tunes a recovery model to improve arterial progression.

核心思想：加入近端车辆、车队成组、自由流、停车惩罚等指标，同时保留支路等待时间硬保护，
避免再次出现“主路直线很好但支路等待爆炸”的问题。
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
连续通行绿波训练脚本：在 Run 34 recovery 模型基础上继续优化主路连续通过。
(English: Progression fine-tuning from the Run 34 recovery model.)

本脚本加入主路 ETA、车队、连续通过代理指标和支路等待硬保护。
(English: It adds ETA, platoon, progression proxies, and side-street wait protection.)
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

# 训练随机种子：固定随机性，方便与 Run 34 对比。
# (Random seed: keep results comparable with Run 34.)
SEED = 8848

# 热启动配置：Run 35 默认从 Run 34 的 recovery 模型继续训练。
# (Warm-start settings: Run 35 starts from the Run 34 recovery model.)
LOAD_MODEL_RUN_IDX = int(os.environ.get("JUC_LOAD_MODEL_RUN_IDX", "34"))
LOAD_MODEL_BASENAME = os.environ.get("JUC_LOAD_MODEL_BASENAME", "ppo_marl_model_gw_recovery")
LOAD_VECNORM_BASENAME = os.environ.get("JUC_LOAD_VECNORM_BASENAME", "vec_normalize_marl_gw_recovery")

TOTAL_TIMESTEPS = int(os.environ.get("JUC_TOTAL_TIMESTEPS", "400000"))

# 信号灯状态：用完整灯色字符串判断主路/支路绿灯。
# (Signal states: identify main/side green by full state strings.)
MAIN_GREEN_STATE = os.environ.get("JUC_MAIN_GREEN_STATE", "rrrrGGggrrrrGGgg")
SIDE_GREEN_STATE = os.environ.get("JUC_SIDE_GREEN_STATE", "GGggrrrrGGggrrrr")

# ETA 与自由流参数：描述主路车辆接近停止线和通过状态。
# (ETA/free-flow settings: describe approaching and passing arterial vehicles.)
ETA_HORIZON = 22.0
ETA_DECAY = 8.0
NEAR_ETA = 7.0
MIN_ETA_SPEED = 2.0
STOPLINE_DISTANCE = 30.0
FREE_FLOW_SPEED = 5.0
MAX_PRESSURE = 5.0

# 奖励权重：保留 Run 34 的支路保护，并谨慎增强主路连续通行。
# (Reward weights: keep side protection and carefully strengthen progression.)
PBRS_WEIGHT = 1.15
PRESSURE_REWARD_WEIGHT = 0.035
GREEN_PRESSURE_WEIGHT = 0.090
RED_PRESSURE_WEIGHT = 0.070
FREE_FLOW_WEIGHT = 0.045

PROGRESSION_REWARD_WEIGHT = 0.080
PROGRESSION_STOP_PENALTY = 0.060
PLATOON_REWARD_WEIGHT = 0.050
PLATOON_WINDOW = 10.0
PLATOON_MIN_COUNT = 3

SIDE_QUEUE_THRESHOLD = 6.0
SIDE_QUEUE_WEIGHT = 0.065
SIDE_RESCUE_REWARD = 0.040
SIDE_MAX_WAIT_SOFT = 45.0
SIDE_MAX_WAIT_HARD = 60.0
SIDE_EMERGENCY_WAIT = 75.0
SIDE_WAIT_PENALTY = 0.040
SIDE_EMERGENCY_PENALTY = 0.120

MAX_MAIN_GREEN_SECONDS = 45.0
MAIN_OVERTIME_WEIGHT = 0.060
IDLE_MAIN_GREEN_WEIGHT = 0.080
MIN_MAIN_PRESSURE_FOR_HOLD = 0.35

# 主路进口和邻居关系：用于 ETA 压力、车队指标和通信观测。
# (Approaches and neighbors: used for ETA pressure, platoons, and communication.)
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


# 随机交通生成：每个 episode 重新生成双向随机交通。
# (Traffic generation: regenerate bidirectional random traffic per episode.)
def generate_route_file_in_root():
    with pushd(ROOT_DIR):
        generate_route_file()


# 学习率调度：Run 35 继续训练时使用较小学习率逐步微调。
# (Learning-rate schedule: small decaying learning rate for fine-tuning.)
def linear_schedule_with_min(initial_value: float, min_value: float) -> Callable[[float], float]:
    def func(progress_remaining: float) -> float:
        return min_value + progress_remaining * (initial_value - min_value)

    return func


# Run 编号工具：自动创建新的训练输出目录。
# (Run numbering: create the next training output folder.)
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


# 兼容包装器：统一 PettingZoo/Gymnasium/SB3 的返回格式。
# (Compatibility wrapper: normalize API return formats.)
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


# 安全调用工具：TraCI 查询异常时返回默认值。
# (Safe TraCI call: return defaults on query failure.)
def safe_call(default, func, *args):
    try:
        return func(*args)
    except Exception:
        return default


# 信号状态查询：读取真实红黄绿状态字符串。
# (Signal query: read actual red/yellow/green state strings.)
def signal_state(sumo, signal_id: str) -> str:
    return sumo.trafficlight.getRedYellowGreenState(signal_id)


# 主路绿灯判断：所有绿波逻辑统一使用 state 字符串。
# (Main green check: use state strings for all progression logic.)
def is_main_green_state(state: str) -> bool:
    return state == MAIN_GREEN_STATE


# 支路绿灯判断：用于判断当前是否正在服务支路。
# (Side green check: determine whether side streets are being served.)
def is_side_green_state(state: str) -> bool:
    return state == SIDE_GREEN_STATE


# 黄灯判断：黄灯为过渡状态，不作为稳定主路绿灯奖励。
# (Yellow check: treat yellow as a transition state.)
def is_yellow_state(state: str) -> bool:
    return "y" in state.lower()


# 仿真时间工具：用于绿灯持续时间和回合重置判断。
# (Simulation-time helper: track green duration and episode reset.)
def current_step(traffic_signal) -> float:
    return float(getattr(traffic_signal.env, "sim_step", 0.0))


# 回合起点判断：新 episode 时清理缓存的势能和计时信息。
# (New-episode check: clear cached shaping/timing state.)
def is_new_episode(traffic_signal) -> bool:
    delta_time = float(getattr(traffic_signal.env, "delta_time", 5.0))
    return current_step(traffic_signal) <= delta_time


# 主路绿灯计时：记录主路绿灯已经连续保持多久。
# (Main-green timer: track continuous main-green duration.)
def update_main_green_timer(traffic_signal, state: str):
    if is_new_episode(traffic_signal):
        for attr in ("main_green_start", "last_potential"):
            if hasattr(traffic_signal, attr):
                delattr(traffic_signal, attr)

    if is_main_green_state(state):
        if not hasattr(traffic_signal, "main_green_start"):
            traffic_signal.main_green_start = current_step(traffic_signal)
    elif hasattr(traffic_signal, "main_green_start"):
        delattr(traffic_signal, "main_green_start")


# 主路绿灯持续时间：为主路超时惩罚提供输入。
# (Main-green elapsed time: input for overtime penalty.)
def main_green_elapsed(traffic_signal) -> float:
    if not hasattr(traffic_signal, "main_green_start"):
        return 0.0
    return max(0.0, current_step(traffic_signal) - traffic_signal.main_green_start)


# 基础 PBRS 奖励：保留原模型的排队和势能塑形能力。
# (Base PBRS reward: retain queue and potential shaping.)
def pbrs_reward(traffic_signal):
    total_queue = traffic_signal.get_total_queued()
    base_reward = -total_queue

    current_light_state = traffic_signal.sumo.trafficlight.getRedYellowGreenState(traffic_signal.id)
    active_phase_queue = 0
    for i, lane in enumerate(traffic_signal.lanes):
        if i < len(current_light_state) and current_light_state[i] in ("G", "g", "y", "Y"):
            active_phase_queue += traffic_signal.sumo.lane.getLastStepHaltingNumber(lane)

    phi_current = active_phase_queue / (total_queue + 1e-6)

    if not hasattr(traffic_signal, "last_potential") or is_new_episode(traffic_signal):
        traffic_signal.last_potential = phi_current
        shaping_reward = 0.0
    else:
        gamma = 0.99
        shaping_reward = gamma * phi_current - traffic_signal.last_potential
        traffic_signal.last_potential = phi_current

    return (base_reward + 100.0 * shaping_reward) / 100.0


# 主路进口指标：统计 ETA 压力、近端车辆、自由流车辆、停车车辆和车队代理指标。
# (Approach metrics: ETA pressure, near/free-flow/stopped vehicles, and platoon proxy.)
def edge_progression_metrics(sumo, edge_id: str) -> dict:
    lane_id = f"{edge_id}_0"
    lane_length = float(safe_call(0.0, sumo.lane.getLength, lane_id))
    vehicle_ids = list(safe_call([], sumo.lane.getLastStepVehicleIDs, lane_id))

    pressure = 0.0
    near_count = 0.0
    free_flow_count = 0.0
    stopped_count = 0.0
    moving_etas = []

    for veh_id in vehicle_ids:
        lane_pos = float(safe_call(0.0, sumo.vehicle.getLanePosition, veh_id))
        speed = float(safe_call(0.0, sumo.vehicle.getSpeed, veh_id))
        distance_to_stopline = max(lane_length - lane_pos, 0.0)
        eta = distance_to_stopline / max(speed, MIN_ETA_SPEED)

        if speed < 0.1:
            stopped_count += 1.0

        if eta <= ETA_HORIZON:
            pressure += np.exp(-eta / ETA_DECAY)
            if eta <= NEAR_ETA:
                near_count += 1.0
            if speed >= FREE_FLOW_SPEED:
                moving_etas.append(eta)

        if distance_to_stopline <= STOPLINE_DISTANCE and speed >= FREE_FLOW_SPEED:
            free_flow_count += 1.0

    moving_etas.sort()
    best_group = 0
    left = 0
    for right, eta in enumerate(moving_etas):
        while eta - moving_etas[left] > PLATOON_WINDOW:
            left += 1
        best_group = max(best_group, right - left + 1)

    return {
        "pressure": float(min(pressure, MAX_PRESSURE)),
        "near_count": float(near_count),
        "free_flow_count": float(min(free_flow_count, 6.0)),
        "stopped_count": float(stopped_count),
        "platoon_score": float(max(0, best_group - PLATOON_MIN_COUNT + 1)),
    }


# 主路到达指标：聚合当前路口双向主路进口的连续通行信息。
# (Main arrival metrics: aggregate bidirectional arterial progression data.)
def main_arrival_metrics(traffic_signal) -> dict:
    pressure = 0.0
    near_count = 0.0
    free_flow_count = 0.0
    stopped_count = 0.0
    platoon_score = 0.0

    for edge_id in MAIN_APPROACH_EDGES.get(traffic_signal.id, []):
        metrics = edge_progression_metrics(traffic_signal.sumo, edge_id)
        pressure += metrics["pressure"]
        near_count += metrics["near_count"]
        free_flow_count += metrics["free_flow_count"]
        stopped_count += metrics["stopped_count"]
        platoon_score += metrics["platoon_score"]

    return {
        "pressure": float(min(pressure, MAX_PRESSURE)),
        "near_count": float(near_count),
        "free_flow_count": float(min(free_flow_count, 6.0)),
        "stopped_count": float(stopped_count),
        "platoon_score": float(min(platoon_score, 6.0)),
    }


# 支路队列和等待：同时统计支路排队数量和最大等待时间。
# (Side queue and wait: count side queues and maximum waiting time.)
def side_queue_and_wait(traffic_signal) -> tuple[float, float]:
    queue = 0.0
    max_wait = 0.0
    for lane in traffic_signal.lanes:
        if "top" in lane or "bottom" in lane:
            queue += safe_call(0, traffic_signal.sumo.lane.getLastStepHaltingNumber, lane)
            for veh_id in safe_call([], traffic_signal.sumo.lane.getLastStepVehicleIDs, lane):
                max_wait = max(max_wait, safe_call(0.0, traffic_signal.sumo.vehicle.getWaitingTime, veh_id))
    return float(queue), float(max_wait)


# 绿波缩放因子：支路等待过长时削弱或关闭绿波奖励。
# (Green-wave scale: reduce/disable progression reward under side-street pressure.)
def green_wave_scale(side_max_wait: float, side_q: float) -> float:
    if side_max_wait >= SIDE_EMERGENCY_WAIT:
        return 0.0
    if side_max_wait >= SIDE_MAX_WAIT_HARD:
        return 0.25
    if side_q > SIDE_QUEUE_THRESHOLD + 4.0:
        return 0.5
    return 1.0


# 连续通行奖励：结合 PBRS、pressure、车队奖励和支路硬保护。
# (Progression reward: combine PBRS, pressure, platoon rewards, and side protection.)
def progression_reward(traffic_signal):
    # 1. 读取真实灯色 state，并更新主路连续绿灯计时器。
    # 该计时器用于限制主路绿灯过长，防止重新造成支路饥饿。
    state = signal_state(traffic_signal.sumo, traffic_signal.id)
    update_main_green_timer(traffic_signal, state)

    # 2. 判断主路绿、支路绿、黄灯。
    # 黄灯不作为稳定绿波奖励依据，避免把换相过程误判为有效放行。
    is_main_green = is_main_green_state(state)
    is_side_green = is_side_green_state(state)
    is_yellow = is_yellow_state(state)

    # 3. PBRS 仍然作为基础效率目标，防止绿波奖励压倒平均等待时间。
    reward = PBRS_WEIGHT * pbrs_reward(traffic_signal)

    # 4. 计算主路连续通行指标和支路风险。
    # metrics 包含 pressure、near_count、free_flow_count、platoon_score、stopped_count 等。
    # side_max_wait 用于动态压低绿波奖励，避免支路等待时间继续扩大。
    metrics = main_arrival_metrics(traffic_signal)
    pressure = metrics["pressure"]
    sq, side_max_wait = side_queue_and_wait(traffic_signal)
    wave_scale = green_wave_scale(side_max_wait, sq)

    # 5. 加入 SUMO-RL 内置 pressure 指标。
    # get_pressure() 反映进出方向压力差，clip 到 [-1,1] 后作为小权重辅助项。
    pressure_reward = np.clip(safe_call(0.0, traffic_signal.get_pressure), -10.0, 10.0) / 10.0
    reward += PRESSURE_REWARD_WEIGHT * pressure_reward

    if pressure > 0:
        if is_main_green:
            # 6. 主路绿灯且主路有车：奖励绿波对齐、自由流、近端通过和车队成组。
            # wave_scale 会在支路等待过高时自动变小，使绿波奖励让位于支路保护。
            reward += wave_scale * GREEN_PRESSURE_WEIGHT * pressure
            reward += wave_scale * FREE_FLOW_WEIGHT * metrics["free_flow_count"]
            reward += wave_scale * PROGRESSION_REWARD_WEIGHT * metrics["near_count"]
            reward += wave_scale * PLATOON_REWARD_WEIGHT * metrics["platoon_score"]
            # 主路车已经进入近端但停车，说明绿波质量不好，因此扣分。
            reward -= PROGRESSION_STOP_PENALTY * metrics["stopped_count"]
        elif not is_yellow:
            # 主路有车但红灯，可能破坏连续通行。
            reward -= RED_PRESSURE_WEIGHT * pressure

    if is_main_green:
        # 7. 主路超长绿灯惩罚：防止模型为了绿波长期不切给支路。
        elapsed = main_green_elapsed(traffic_signal)
        if elapsed > MAX_MAIN_GREEN_SECONDS:
            reward -= MAIN_OVERTIME_WEIGHT * (elapsed - MAX_MAIN_GREEN_SECONDS)

        # 8. 空放主路惩罚：主路压力不足但支路有排队时，应尽快释放支路。
        if pressure < MIN_MAIN_PRESSURE_FOR_HOLD and sq > 0:
            reward -= IDLE_MAIN_GREEN_WEIGHT * min(sq, 10.0)

        # 9. 支路队列和等待时间保护。
        # soft/hard 两级阈值让模型先温和修正，严重时强制更大惩罚。
        if sq > SIDE_QUEUE_THRESHOLD:
            reward -= SIDE_QUEUE_WEIGHT * (sq - SIDE_QUEUE_THRESHOLD)

        if side_max_wait > SIDE_MAX_WAIT_SOFT:
            reward -= SIDE_WAIT_PENALTY * ((side_max_wait - SIDE_MAX_WAIT_SOFT) / 5.0)

        if side_max_wait > SIDE_MAX_WAIT_HARD:
            reward -= SIDE_EMERGENCY_PENALTY * ((side_max_wait - SIDE_MAX_WAIT_HARD) / 5.0)

    if is_side_green and sq > SIDE_QUEUE_THRESHOLD:
        # 10. 支路救援奖励：支路排队时给支路绿灯，说明策略在恢复公平性。
        reward += SIDE_RESCUE_REWARD * min(sq - SIDE_QUEUE_THRESHOLD, 10.0)

    if is_side_green and side_max_wait > SIDE_MAX_WAIT_SOFT:
        # 11. 等待时间救援奖励：支路车辆已经等太久时，支路绿灯得到额外正反馈。
        reward += SIDE_RESCUE_REWARD * min((side_max_wait - SIDE_MAX_WAIT_SOFT) / 5.0, 10.0)

    return reward


# 通信观测：基础观测 + 相邻路口主路排队信息。
# (Communication observation: base observation plus neighboring arterial queues.)
class CommObservationFunction(DefaultObservationFunction):
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


# 环境构建：加载 Run 34 标准化统计，并在每回合重生成交通。
# (Environment builder: load Run 34 VecNormalize and regenerate traffic per episode.)
def build_env(csv_base_path, vec_norm_path):
    # 1. 构建 progression 微调环境。
    # reward_fn 使用 progression_reward，max_green 提供主路连续绿灯上限的环境约束。
    env = parallel_env(
        net_file=net_path,
        route_file=route_path,
        out_csv_name=csv_base_path,
        use_gui=False,
        num_seconds=3600,
        reward_fn=progression_reward,
        observation_class=CommObservationFunction,
        min_green=10,
        max_green=60,
    )

    original_reset = env.reset

    def custom_reset(*args, **kwargs):
        # 每个 episode 更新 route 文件，避免模型只适配单一交通流样本。
        print("Regenerating random bidirectional traffic for this episode...")
        generate_route_file_in_root()
        return original_reset(*args, **kwargs)

    env.reset = custom_reset
    # 2. 转换为 SB3 兼容的向量环境，并记录 episode 统计。
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, num_vec_envs=1, num_cpus=1, base_class="stable_baselines3")
    env = SB3CompatibilityWrapper(env)
    env = VecMonitor(env)

    if not os.path.exists(vec_norm_path):
        raise FileNotFoundError(f"VecNormalize file not found: {vec_norm_path}")

    # 3. 加载 Run 34/recovery 阶段的 VecNormalize。
    # 继续训练必须沿用源模型的观测归一化，否则同一个物理状态会变成不同尺度的神经网络输入。
    env = VecNormalize.load(vec_norm_path, env)
    env.training = True
    env.norm_reward = False
    return env


# 源模型路径：定位要继续训练的 Run 34 模型与标准化文件。
# (Source paths: locate the model and normalization file for warm start.)
def source_paths():
    source_dir = os.path.join(ROOT_DIR, "saved_models", f"marl_run_{LOAD_MODEL_RUN_IDX}")
    model_path = os.path.join(source_dir, f"{LOAD_MODEL_BASENAME}.zip")
    vec_norm_path = os.path.join(source_dir, f"{LOAD_VECNORM_BASENAME}.pkl")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not os.path.exists(vec_norm_path):
        raise FileNotFoundError(f"VecNormalize file not found: {vec_norm_path}")

    return model_path, vec_norm_path


# 主程序：从 recovery 模型继续训练，并保存 progression 模型。
# (Main entry: fine-tune from recovery and save progression outputs.)
if __name__ == "__main__":
    print("Initializing MARL green-wave progression fine-tuning...")

    # 1. 固定随机种子，便于复现实验。
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    saved_models_base = os.path.join(ROOT_DIR, "saved_models")
    logs_base = os.path.join(ROOT_DIR, "logs")
    os.makedirs(saved_models_base, exist_ok=True)
    os.makedirs(logs_base, exist_ok=True)

    run_idx = get_next_run_number(saved_models_base, "marl_run_")
    print(f"Detected new run index: {run_idx}")

    run_save_dir = os.path.join(saved_models_base, f"marl_run_{run_idx}")
    os.makedirs(run_save_dir, exist_ok=True)
    run_csv_dir = os.path.join(logs_base, f"marl_run_{run_idx}")
    os.makedirs(run_csv_dir, exist_ok=True)
    csv_base_path = os.path.join(run_csv_dir, "marl_output")

    model_path, vec_norm_path = source_paths()
    print(f"Fine-tuning from model: {model_path}")
    print(f"Using VecNormalize stats: {vec_norm_path}")

    # 2. 从 recovery 模型和归一化统计继续训练，而不是从零训练。
    generate_route_file_in_root()
    env = build_env(csv_base_path, vec_norm_path)
    model = PPO.load(
        model_path,
        env=env,
        tensorboard_log=os.path.join(ROOT_DIR, "logs", "ppo_marl_tb"),
    )
    model.learning_rate = linear_schedule_with_min(5e-5, 1e-5)
    model.ent_coef = 0.01

    print("=" * 70)
    print("Observation space:", model.policy.observation_space)
    print("Action space:", model.policy.action_space)
    print(f"Source run: marl_run_{LOAD_MODEL_RUN_IDX}")
    print(f"Source model: {LOAD_MODEL_BASENAME}")
    print(f"Main overtime cap: {MAX_MAIN_GREEN_SECONDS}s")
    print(f"Side queue threshold: {SIDE_QUEUE_THRESHOLD}")
    print(f"Side hard wait threshold: {SIDE_MAX_WAIT_HARD}s")
    print("=" * 70)

    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path=os.path.join(run_save_dir, "checkpoints"),
        name_prefix="rl_model",
    )

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=checkpoint_callback,
        tb_log_name=f"run_{run_idx}_gw_progression",
        reset_num_timesteps=True,
    )

    # 3. 保存通用名称和 progression 专用名称。
    # 通用名称方便旧脚本自动查找；专用名称方便报告中区分实验阶段。
    model.save(os.path.join(run_save_dir, "ppo_marl_model"))
    env.save(os.path.join(run_save_dir, "vec_normalize_marl.pkl"))
    model.save(os.path.join(run_save_dir, "ppo_marl_model_gw_progression"))
    env.save(os.path.join(run_save_dir, "vec_normalize_marl_gw_progression.pkl"))

    env.close()
    print(f"Progression fine-tuning finished. Outputs saved to: {run_save_dir}")
