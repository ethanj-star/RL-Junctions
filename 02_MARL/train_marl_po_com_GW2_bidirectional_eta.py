"""
双向 ETA 绿波 MARL 训练脚本。
Bidirectional ETA green-wave MARL training script.

本文件尝试直接把 A-B-C 与 C-B-A 两个方向的主路到达时间 ETA 纳入观测和奖励。
It uses ETA features from both arterial directions in both observation and reward design.

核心设计：
1. 使用真实红绿灯 state 字符串判断主路绿灯，而不是依赖 phase 编号；
2. 观测中加入邻居路口、主路 ETA、支路排队和下游阻塞信息；
3. 奖励函数同时考虑主路 ETA 对齐、自由流通过、支路公平性和下游 spillback。
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
双向 ETA 绿波训练脚本：同时考虑 A-B-C 与 C-B-A 两个主路方向。
( Bidirectional ETA green-wave training for both arterial directions.)

本脚本用于修正早期只优化单向主路车流的问题，并加入下游阻塞保护。
( It fixes one-way-only progression and adds downstream spillback protection.)
"""

# 学习率调度：训练后期保留最低学习率，便于稳定微调。
# (Learning-rate schedule: decay with a non-zero floor.)
def linear_schedule_with_min(initial_value: float, min_value: float) -> Callable[[float], float]:
    def func(progress_remaining: float) -> float:
        return min_value + progress_remaining * (initial_value - min_value)

    return func


# 路径配置：自动定位项目根目录、路网文件和随机交通流文件。
# (Path setup: locate project root, SUMO net file, and route file.)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CURRENT_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from generate_Random_Traffic import generate_route_file

net_path = os.path.join(ROOT_DIR, "SUMOroutes.net.xml")
route_path = os.path.join(ROOT_DIR, "traffic.random.rou.xml")

# 信号灯状态：用完整灯色字符串判断主路/支路绿灯。
# (Signal states: identify main/side green by full state strings.)
MAIN_GREEN_STATE = os.environ.get("JUC_MAIN_GREEN_STATE", "rrrrGGggrrrrGGgg")
SIDE_GREEN_STATE = os.environ.get("JUC_SIDE_GREEN_STATE", "GGggrrrrGGggrrrr")

# ETA 与奖励参数：描述车辆接近停止线、自由流通过和支路保护。
# (ETA/reward settings: approaching vehicles, free-flow passage, and side protection.)
ETA_HORIZON = 25.0                             #estimated time of arrival车辆预计 25 秒内到达停止线
ETA_NEAR_WINDOW = 8.0                          #车辆预计 8 秒内到达停止线，算作“近端来车”。这类车更需要当前或即将获得绿灯，否则容易停车。
ETA_DECAY = 8.0                                #ETA 压力衰减系数。车辆越接近停止线，奖励压力越大；越远，影响越小。8.0 控制这个衰减速度。
MIN_ETA_SPEED = 2.0                            #计算 ETA 时使用的最低速度
FREE_FLOW_SPEED = 5.0                          #车辆速度大于等于 5 m/s 时，认为它处于较顺畅通行状态。用于判断主路车辆是否自由流通过。
STOPLINE_DISTANCE = 25.0                       #距离停止线 25 m以内且速度较快的车辆，会被认为是接近停止线并顺畅通过的车辆。
#基础排队与势能参数
QUEUE_WEIGHT = 0.04                            #排队惩罚权重。排队车辆越多，奖励越低，用于防止模型只顾绿波、不管排队。
POTENTIAL_QUEUE_NORM = 50.0                    #势能奖励归一化参数。用于把排队数量缩放到比较稳定的范围，避免奖励数值过大。
#主路绿波奖励参数
ETA_GREEN_WEIGHT = 0.18                        #如果主路有 ETA 压力，主路车辆即将到达，且当前是主路绿灯就奖励。这个值越大模型越保持主路绿灯。
ETA_RED_WEIGHT = 0.24                          #如果主路车辆即将到达，但当前不是主路绿灯，就给惩罚。
FREE_FLOW_WEIGHT = 0.08                        #如果主路车辆在停止线附近以较高速度通过，就给奖励。它鼓励时空图里出现不停顿的斜直线。
#支路保护参数
SIDE_QUEUE_WEIGHT = 0.03                       #支路排队惩罚权重。支路排队越多，继续偏向主路的代价越大。
SIDE_QUEUE_THRESHOLD = 6.0                     #支路排队阈值。支路排队超过 6 辆后，开始明显惩罚主路长期绿灯，避免支路被饿死。
#主路长绿惩罚参数
LONG_MAIN_GREEN_START = 35.0                   #主路连续绿灯超过 35 秒后，开始认为主路绿灯偏长，需要检查是否空放或压制支路。
IDLE_MAIN_GREEN_PENALTY = 0.04                 #空放主路绿灯惩罚。如果主路没有明显来车压力，但仍然保持主路绿灯，就扣分。
#下游溢出保护参数
SPILLBACK_WEIGHT = 0.18                        #下游阻塞惩罚权重。如果下游主路已经拥堵，还继续放车进入，就会加重惩罚。
SPILLBACK_OCC_THRESHOLD = 0.45                 #下游占有率阈值。下游道路占有率超过 45%，认为可能出现车辆排队回溢，需要减少继续放行主路的倾向。

# 双向主路进口边：每个路口同时考虑 A-B-C 与 C-B-A 两个方向。
# (Bidirectional main approaches: include both corridor directions.)
MAIN_APPROACH_EDGES = {
    "A0": ["left0A0", "B0A0"],
    "B0": ["A0B0", "C0B0"],
    "C0": ["B0C0", "right0C0"],
}

# 主路下游边：用于识别下游阻塞，避免把车放入堵塞路段。
# (Downstream links: avoid releasing vehicles into blocked downstream links.)
MAIN_DOWNSTREAM_EDGES = {
    "A0": ["A0B0", "A0left0"],
    "B0": ["B0C0", "B0A0"],
    "C0": ["C0right0", "C0B0"],
}

# 邻居关系：用于将相邻路口的主路排队加入观测。
# (Neighbor map: add neighboring arterial queues to observation.)
NEIGHBOR_MAP = {
    "A0": [None, "B0"],
    "B0": ["A0", "C0"],
    "C0": ["B0", None],
}


# 文件目录切换工具：随机交通生成函数需要在项目根目录运行。
# (Directory helper: route generation expects the project root as cwd.)
@contextmanager
def pushd(path: str):
    old_cwd = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old_cwd)


# 随机交通生成：每个 episode 前重新生成双向随机流。
# (Traffic generation: regenerate bidirectional random traffic per episode.)
def generate_route_file_in_root():
    with pushd(ROOT_DIR):
        generate_route_file()


# Run 编号工具：自动创建新的 marl_run_xx 输出目录。
# (Run numbering: create the next marl_run_xx output folder.)
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


# 兼容包装器：统一 PettingZoo/Gymnasium/SB3 的 reset 和 step 返回格式。
# (Compatibility wrapper: normalize reset/step API formats.)
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


# 安全调用工具：TraCI 查询失败时返回默认值，避免训练中断。
# (Safe TraCI call: return defaults on query failure.)
def _safe(default, func, *args):
    try:
        return func(*args)
    except Exception:
        return default


# 车道名工具：由 edge ID 拼接 SUMO 单车道 lane ID。
# (Lane helper: convert edge ID to single-lane SUMO lane ID.)
def _lane_id(edge_id: str) -> str:
    return f"{edge_id}_0"


# 仿真时间工具：用于回合重置和主路绿灯持续时间计算。
# (Simulation-time helper: track resets and main-green duration.)
def _current_step(traffic_signal) -> float:
    return float(getattr(traffic_signal.env, "sim_step", 0.0))


# 回合起点判断：新 episode 时清理缓存状态。
# (New-episode check: clear cached state at episode start.)
def _is_new_episode(traffic_signal) -> bool:
    current_step = _current_step(traffic_signal)
    delta_time = float(getattr(traffic_signal.env, "delta_time", 5.0))
    return current_step <= delta_time


# 信号状态查询：读取真实灯色字符串。
# (Signal query: read actual red/yellow/green state strings.)
def _signal_state(sumo, signal_id: str) -> str:
    return sumo.trafficlight.getRedYellowGreenState(signal_id)


# 主路绿灯判断：用 state 字符串识别主路是否放行。
# (Main green check: identify main-road service by state string.)
def _is_main_green_state(state: str) -> bool:
    return state == MAIN_GREEN_STATE


# 黄灯判断：过渡状态不作为稳定绿波服务。
# (Yellow check: exclude transition states from progression service.)
def _is_yellow_state(state: str) -> bool:
    return "y" in state.lower()


# 主路绿灯跟踪：记录连续主路绿灯持续时间和奖励势能。
# (Green tracking: track continuous main green and cached reward potential.)
def _update_green_tracking(traffic_signal):
    if _is_new_episode(traffic_signal):
        for attr in ("my_green_start", "last_signal_state", "last_potential"):
            if hasattr(traffic_signal, attr):
                delattr(traffic_signal, attr)

    current_state = _signal_state(traffic_signal.sumo, traffic_signal.id)
    is_main_green = _is_main_green_state(current_state)

    if not hasattr(traffic_signal, "last_signal_state"):
        traffic_signal.last_signal_state = current_state
    traffic_signal.last_signal_state = current_state

    if is_main_green:
        if not hasattr(traffic_signal, "my_green_start"):
            traffic_signal.my_green_start = _current_step(traffic_signal)
    elif hasattr(traffic_signal, "my_green_start"):
        delattr(traffic_signal, "my_green_start")

    return current_state, is_main_green


# 主路绿灯时长归一化：用于惩罚长时间空放主路。
# (Green-duration normalization: penalize excessive main green.)
def _green_duration_norm(traffic_signal) -> float:
    if not hasattr(traffic_signal, "my_green_start"):
        return 0.0
    return min((_current_step(traffic_signal) - traffic_signal.my_green_start) / 60.0, 1.0)


# 单条主路进口指标：统计 ETA 压力、近端车辆、自由流车辆和排队。
# (Edge metrics: ETA pressure, near vehicles, free-flow count, and queue.)
def _edge_metrics(sumo, edge_id: str) -> dict:
    lane_id = _lane_id(edge_id)
    lane_length = float(_safe(0.0, sumo.lane.getLength, lane_id))
    vehicle_ids = list(_safe([], sumo.lane.getLastStepVehicleIDs, lane_id))
    halting = float(_safe(0, sumo.lane.getLastStepHaltingNumber, lane_id))
    occupancy = float(_safe(0.0, sumo.lane.getLastStepOccupancy, lane_id)) / 100.0

    pressure = 0.0
    near_count = 0.0
    free_flow_count = 0.0
    eta_values = []

    for veh_id in vehicle_ids:
        lane_pos = float(_safe(0.0, sumo.vehicle.getLanePosition, veh_id))
        distance_to_stopline = max(lane_length - lane_pos, 0.0)
        speed = float(_safe(0.0, sumo.vehicle.getSpeed, veh_id))
        eta = distance_to_stopline / max(speed, MIN_ETA_SPEED)

        if eta <= ETA_HORIZON:
            eta_values.append(eta)
            pressure += np.exp(-eta / ETA_DECAY)
            if eta <= ETA_NEAR_WINDOW:
                near_count += 1.0

        if distance_to_stopline <= STOPLINE_DISTANCE and speed >= FREE_FLOW_SPEED:
            free_flow_count += 1.0

    min_eta = min(eta_values) if eta_values else ETA_HORIZON

    return {
        "vehicle_count": float(len(vehicle_ids)),
        "halting": halting,
        "occupancy": min(occupancy, 1.0),
        "pressure": float(pressure),
        "near_count": near_count,
        "free_flow_count": free_flow_count,
        "min_eta": float(min_eta),
    }


# 主路到达指标：聚合当前路口两侧主路进口的 ETA 信息。
# (Main arrival metrics: aggregate both arterial approaches.)
def _main_arrival_metrics(traffic_signal) -> dict:
    result = {
        "pressure": 0.0,
        "near_count": 0.0,
        "halting": 0.0,
        "free_flow_count": 0.0,
        "min_eta": ETA_HORIZON,
        "edge_metrics": [],
    }

    for edge_id in MAIN_APPROACH_EDGES.get(traffic_signal.id, []):
        metrics = _edge_metrics(traffic_signal.sumo, edge_id)
        result["edge_metrics"].append(metrics)
        result["pressure"] += metrics["pressure"]
        result["near_count"] += metrics["near_count"]
        result["halting"] += metrics["halting"]
        result["free_flow_count"] += metrics["free_flow_count"]
        result["min_eta"] = min(result["min_eta"], metrics["min_eta"])

    return result


# 支路排队统计：防止主路绿波长期压制支路。
# (Side queue: prevent arterial progression from starving side streets.)
def _side_queue(traffic_signal) -> float:
    side_queue = 0.0
    for lane_id in traffic_signal.lanes:
        if "top" in lane_id or "bottom" in lane_id:
            side_queue += float(_safe(0, traffic_signal.sumo.lane.getLastStepHaltingNumber, lane_id))
    return side_queue


# 主路排队统计：作为基础效率奖励的一部分。
# (Main queue: part of the base efficiency objective.)
def _main_queue(traffic_signal) -> float:
    main_queue = 0.0
    for lane_id in traffic_signal.lanes:
        if "top" not in lane_id and "bottom" not in lane_id:
            main_queue += float(_safe(0, traffic_signal.sumo.lane.getLastStepHaltingNumber, lane_id))
    return main_queue


# 下游阻塞检测：如果下游边占有率过高，减少继续放主路的奖励。
# (Downstream blockage: discourage releasing into congested downstream links.)
def _downstream_blockage(traffic_signal) -> float:
    blockage = 0.0
    for edge_id in MAIN_DOWNSTREAM_EDGES.get(traffic_signal.id, []):
        metrics = _edge_metrics(traffic_signal.sumo, edge_id)
        occupancy_pressure = max(0.0, metrics["occupancy"] - SPILLBACK_OCC_THRESHOLD)
        queue_pressure = min(metrics["halting"] / 12.0, 1.0)
        blockage += max(occupancy_pressure, queue_pressure)
    return min(blockage, 1.0)


# 双向 ETA 绿波奖励：同时考虑主路来车、红灯惩罚、支路队列和下游阻塞。
# (Bidirectional ETA reward: combine arrival pressure, red penalty, side queue, and spillback.)
def custom_bidirectional_green_wave_reward(traffic_signal):
    """
    奖励函数的总体目标：
    1. 保持基础通行效率：全局排队越长，奖励越低；
    2. 鼓励双向主路绿波：当 A-B-C 或 C-B-A 主路车辆即将到达停止线时，优先给主路绿灯；
    3. 避免错误绿波：如果主路车辆即将到达但信号仍为红灯，则给予惩罚；
    4. 保留支路公平性：主路绿灯持续过久且支路排队明显时，惩罚长期压制支路；
    5. 避免下游回溢：如果下游主路已经拥堵，不继续盲目奖励主路放行。

    (Overall goal: combine efficiency, bidirectional arterial progression,
    side-street fairness, and downstream spillback protection.)
    """

    # 读取当前信号灯状态，并根据真实红绿灯 state 字符串判断是否为主路绿灯。
    # 注意：这里不能只看 phase 编号，因为 phase 编号不一定可靠；主路绿灯应由 state 字符串识别。
    # (Read the actual signal state string; do not rely only on phase id.)
    current_state, is_main_green = _update_green_tracking(traffic_signal)
    is_yellow = _is_yellow_state(current_state)

    # 1. 基础排队惩罚：
    # traffic_signal.get_total_queued() 返回当前路口所有进口道的排队车辆数。
    # 队列越长，说明路口整体服务越差，因此给予负奖励。
    # QUEUE_WEIGHT 控制“降低排队”在总奖励中的重要程度。
    # (Base queue penalty: discourage long total queues.)
    total_queue = float(traffic_signal.get_total_queued())
    base_penalty = -QUEUE_WEIGHT * total_queue

    # 2. 势函数 shaping：
    # phi_current 是当前状态的势能，排队越少，势能越高。
    # shaping_reward = gamma * 当前势能 - 上一步势能。
    # 这样做的作用不是直接定义最终目标，而是让 PPO 更容易感知“排队正在变好还是变坏”。
    # 第一个 step 或新 episode 开始时没有上一状态，因此 shaping_reward 设为 0。
    # (Potential-based shaping: reward improvements in queue state over time.)
    phi_current = -total_queue / POTENTIAL_QUEUE_NORM

    if getattr(traffic_signal, "last_potential", None) is None or _is_new_episode(traffic_signal):
        traffic_signal.last_potential = phi_current
        shaping_reward = 0.0
    else:
        gamma = 0.99
        shaping_reward = gamma * phi_current - traffic_signal.last_potential
        traffic_signal.last_potential = phi_current

    # 3. 主路到达压力、支路队列、下游阻塞：
    # arrival 汇总当前路口两侧主路进口边的 ETA 信息，包括：
    # - pressure：车辆接近停止线的强度，ETA 越小权重越高；
    # - near_count：短时间窗口内即将到达的车辆数；
    # - free_flow_count：接近停止线且速度较高的主路车辆数；
    # - halting / occupancy：排队和占有率信息。
    # arrival_pressure 被截断到 6，是为了防止高流量时该项过大，压倒排队和公平性奖励。
    # side_q 用于判断支路是否被主路绿波长期压制。
    # downstream_block 用于判断下游是否已经拥堵，避免继续向堵塞区间放车。
    # (Collect arterial ETA pressure, side queue, and downstream blockage.)
    arrival = _main_arrival_metrics(traffic_signal)
    arrival_pressure = min(arrival["pressure"], 6.0)
    side_q = _side_queue(traffic_signal)
    downstream_block = _downstream_blockage(traffic_signal)

    # 4. ETA 主路绿波奖励/惩罚：
    # 如果主路车辆即将到达，并且当前是主路绿灯，则奖励。
    # 如果主路车辆即将到达，但当前不是主路绿灯且也不是黄灯，则惩罚。
    # 黄灯阶段不直接按红灯惩罚，因为黄灯是相位切换的过渡状态，强行惩罚会干扰正常换相。
    # 这一项是本文件“ETA 绿波”的核心：它鼓励信号灯在主路车队到达停止线前后保持主路放行。
    # (ETA green-wave term: reward green on arrival, penalize red on arrival.)
    eta_reward = 0.0
    if arrival_pressure > 0.0:
        if is_main_green:
            eta_reward += ETA_GREEN_WEIGHT * arrival_pressure
        elif not is_yellow:
            eta_reward -= ETA_RED_WEIGHT * arrival_pressure

    # 5. 自由流通过奖励：
    # 如果当前是主路绿灯，并且有车辆以较高速度接近停止线，则给予额外奖励。
    # 这不是奖励“车多”，而是奖励“车辆没有明显减速地通过路口”。
    # free_flow_count 最多按 4 辆计入，避免大车流时该奖励过强。
    # (Free-flow reward: encourage vehicles passing without significant slowing.)
    free_flow_reward = 0.0
    if is_main_green:
        free_flow_reward += FREE_FLOW_WEIGHT * min(arrival["free_flow_count"], 4.0)

    # 6. 空放主路惩罚与支路公平性惩罚：
    # idle_green_penalty：
    # 当主路几乎没有即将到达车辆 arrival_pressure < 0.2，但支路已经有排队 side_q > 0 时，
    # 继续保持主路绿灯属于“空放主路”，因此惩罚。
    # fairness_penalty：
    # 如果支路排队超过 SIDE_QUEUaE_THRESHOLD，并且主路绿灯持续时间超过 LONG_MAIN_GREEN_START，
    # 说明主路绿波可能正在长期压制支路。此时根据支路超载量和主路绿灯超时时长逐步加重惩罚。
    # 这一项的目的不是取消绿波，而是在主路绿波和支路可通行之间建立约束。
    # (Idle-green and fairness penalties: prevent starving side streets.)
    idle_green_penalty = 0.0
    fairness_penalty = 0.0
    if is_main_green:
        green_elapsed = 60.0 * _green_duration_norm(traffic_signal)
        if arrival_pressure < 0.2 and side_q > 0:
            idle_green_penalty -= IDLE_MAIN_GREEN_PENALTY
        if side_q > SIDE_QUEUE_THRESHOLD and green_elapsed > LONG_MAIN_GREEN_START:
            overload = side_q - SIDE_QUEUE_THRESHOLD
            duration_factor = min((green_elapsed - LONG_MAIN_GREEN_START) / 25.0, 1.0)
            fairness_penalty -= SIDE_QUEUE_WEIGHT * overload * duration_factor

    # 7. 下游回溢惩罚：
    # 如果当前是主路绿灯，但下游主路区间已经出现高占有率或排队，
    # 继续放主路车辆可能造成车辆进入下游后无法消散，形成 spillback。
    # 因此 downstream_block 越大、主路到达压力越大，惩罚越明显。
    # max(arrival_pressure, 1.0) 的作用是：即使 arrival_pressure 较小，
    # 只要检测到下游堵塞，也保留一个最低惩罚强度。
    # (Spillback penalty: discourage releasing vehicles into blocked downstream links.)
    spillback_penalty = 0.0
    if is_main_green and downstream_block > 0.0:
        spillback_penalty -= SPILLBACK_WEIGHT * downstream_block * max(arrival_pressure, 1.0)

    # 8. 总奖励：
    # 最终奖励由七部分相加得到：
    # - base_penalty：所有方向的基础排队惩罚；
    # - shaping_reward：排队改善趋势；
    # - eta_reward：主路车辆到达时是否给绿灯；
    # - free_flow_reward：主路车辆是否自由流通过；
    # - idle_green_penalty：主路无车时是否仍空放主路；
    # - fairness_penalty：支路是否被长期压制；
    # - spillback_penalty：下游是否已经堵塞仍继续放行。
    # 因此，这个奖励函数不是单纯追求绿波直线，而是尝试在“主路连续通行”和“整体等待时间”
    # 之间取得平衡。后续如果绿波明显但等待时间爆炸，通常说明 fairness_penalty 或支路释放机制不足；
    # 如果等待时间较低但绿波不明显，通常说明 eta_reward/free_flow_reward 对主路连续性的约束不足。
    # (Total reward: balance arterial progression and network-wide delay.)
    return (
        base_penalty
        + shaping_reward
        + eta_reward
        + free_flow_reward
        + idle_green_penalty
        + fairness_penalty
        + spillback_penalty
    )


# 双向 ETA 观测：基础观测 + 邻居排队 + 本路口 ETA/队列/下游阻塞。
# (Bidirectional ETA observation: base state plus neighbor and arterial features.)
class BidirectionalETAObservationFunction(DefaultObservationFunction):
    EXTRA_DIM = 18

    def __init__(self, ts):
        super().__init__(ts)

    def observation_space(self):
        # 1. 读取 SUMO-RL 默认观测空间。
        # 默认观测包含当前相位、min_green、车道密度和排队等基础信息。
        base_space = super().observation_space()
        base_dim = base_space.shape[0]
        # 2. 在默认观测后追加 18 个绿波相关特征。
        # EXTRA_DIM 必须和 _neighbor_features + _eta_features + _local_features 的总长度一致。
        new_dim = base_dim + self.EXTRA_DIM
        return spaces.Box(
            low=np.zeros(new_dim, dtype=np.float32),
            high=np.ones(new_dim, dtype=np.float32) * np.inf,
        )

    def _neighbor_features(self):
        # 邻居特征：每个邻居提供 [主路排队归一化, 是否主路绿灯, 主路绿灯持续时间归一化]。
        # 边界路口缺少邻居时用 0 填充，保证 A0/B0/C0 的观测维度一致。
        features = []
        for neighbor_id in NEIGHBOR_MAP.get(self.ts.id, [None, None]):
            if neighbor_id is None:
                features.extend([0.0, 0.0, 0.0])
                continue

            neighbor_ts = self.ts.env.traffic_signals.get(neighbor_id)
            if neighbor_ts is None:
                features.extend([0.0, 0.0, 0.0])
                continue

            queue_norm = min(_main_queue(neighbor_ts) / 50.0, 1.0)
            state = _signal_state(neighbor_ts.sumo, neighbor_id)
            # 邻居是否为主路绿灯也使用完整 state 字符串判断。
            is_main_green = 1.0 if _is_main_green_state(state) else 0.0
            green_duration_norm = _green_duration_norm(neighbor_ts)
            features.extend([queue_norm, is_main_green, green_duration_norm])

        return features

    def _eta_features(self):
        # ETA 特征：每个主路进口提供 [压力, 近端车辆数, 停车数, 最小 ETA]。
        # 当前路口最多取两个主路进口，因此总长度为 8。
        features = []
        edge_ids = MAIN_APPROACH_EDGES.get(self.ts.id, [])
        for edge_id in edge_ids[:2]:
            metrics = _edge_metrics(self.ts.sumo, edge_id)
            # 所有指标都缩放到 0-1 附近，避免某个特征数值过大主导神经网络输入。
            pressure_norm = min(metrics["pressure"] / 4.0, 1.0)
            near_norm = min(metrics["near_count"] / 6.0, 1.0)
            halt_norm = min(metrics["halting"] / 12.0, 1.0)
            min_eta_norm = min(metrics["min_eta"] / ETA_HORIZON, 1.0)
            features.extend([pressure_norm, near_norm, halt_norm, min_eta_norm])

        while len(features) < 8:
            # 如果某个路口缺少主路进口，用默认值填充。
            # min_eta_norm 填 1.0 表示“没有近端到达压力”。
            features.extend([0.0, 0.0, 0.0, 1.0])

        return features[:8]

    def _local_features(self):
        # 本地特征：当前路口主路绿灯状态、绿灯持续时间、支路排队、下游阻塞。
        # 这些特征帮助模型判断是否应该保持主路绿波，还是释放支路/避免回溢。
        state = _signal_state(self.ts.sumo, self.ts.id)
        is_main_green = 1.0 if _is_main_green_state(state) else 0.0
        side_queue_norm = min(_side_queue(self.ts) / 30.0, 1.0)
        downstream_block = _downstream_blockage(self.ts)
        return [is_main_green, _green_duration_norm(self.ts), side_queue_norm, downstream_block]

    def __call__(self):
        # 1. 获取默认观测，再拼接自定义 ETA/邻居/本地特征。
        base_obs = super().__call__()
        extra_obs = self._neighbor_features() + self._eta_features() + self._local_features()
        # 2. 转为 float32，匹配 Gymnasium/SB3 对 Box 观测的类型要求。
        final_obs = np.concatenate([base_obs, np.array(extra_obs, dtype=np.float32)])
        return np.array(final_obs, dtype=np.float32)


# 主程序：构建双向 ETA 环境、训练 PPO，并保存模型。
# (Main entry: build bidirectional ETA env, train PPO, and save model.)
if __name__ == "__main__":
    print("Initializing bidirectional ETA green-wave MARL training...")

    seed = 8848
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

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
    tensorboard_log_path = os.path.join(logs_base, "ppo_marl_tb")

    print("Generating initial random bidirectional traffic...")
    generate_route_file_in_root()

    env = parallel_env(
        net_file=net_path,
        route_file=route_path,
        out_csv_name=csv_base_path,
        use_gui=False,
        num_seconds=3600,
        reward_fn=custom_bidirectional_green_wave_reward,
        observation_class=BidirectionalETAObservationFunction,
        min_green=10,
        max_green=55,
    )

    original_reset = env.reset

    def custom_reset(*args, **kwargs):
        print("Regenerating random bidirectional traffic for this episode...")
        generate_route_file_in_root()
        return original_reset(*args, **kwargs)

    env.reset = custom_reset

    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(
        env,
        num_vec_envs=1,
        num_cpus=1,
        base_class="stable_baselines3",
    )

    env = SB3CompatibilityWrapper(env)
    env = VecMonitor(env)
    env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.0)

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=linear_schedule_with_min(3e-4, 3e-5),
        n_steps=2048,
        batch_size=256,
        n_epochs=10,
        clip_range=0.2,
        ent_coef=0.005,
        target_kl=0.05,
        verbose=1,
        device="cpu",
        tensorboard_log=tensorboard_log_path,
    )

    print("=" * 60)
    print("Bidirectional ETA observation space:", model.policy.observation_space)
    print("Action space:", model.policy.action_space)
    print("=" * 60)

    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path=os.path.join(run_save_dir, "checkpoints"),
        name_prefix="rl_model",
    )

    model.learn(
        total_timesteps=300000,
        callback=checkpoint_callback,
        tb_log_name=f"run_{run_idx}_bidirectional_eta",
    )

    model.save(os.path.join(run_save_dir, "ppo_marl_model_bidirectional_eta"))
    env.save(os.path.join(run_save_dir, "vec_normalize_marl_bidirectional_eta.pkl"))

    env.close()
    print("Bidirectional ETA green-wave MARL training finished.")
