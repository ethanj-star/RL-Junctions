"""
绿波模型诊断脚本。
Green-wave model diagnostic script.

本文件不参与训练，只加载指定 Run 的模型跑一轮仿真，并输出诊断 CSV。
It does not train; it runs one deterministic simulation and exports diagnostic CSV files.

诊断目标：用 route_summary 判断主路/支路等待是否失衡；用 state_summary 基于真实灯色字符串统计主路绿灯比例；
用 progression_summary 和 platoon_summary 判断主路是否连续通行、是否成组。
"""

import os
import re
from collections import defaultdict

import numpy as np
import pandas as pd
import traci
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecEnvWrapper, VecNormalize
from sumo_rl import parallel_env
from sumo_rl.environment.observations import DefaultObservationFunction
import supersuit as ss

"""
绿波诊断脚本：加载指定 Run 的模型，输出通行、排队、灯色和车队诊断表。
(English: Diagnose a selected model's traffic, queues, signal states, and platoons.)

本脚本不参与训练，只用于验证支路是否饿死、主路是否连续通过、车队是否成组。
(English: It does not train; it validates side service, progression, and platoons.)
"""

# 路径与模型选择：通过环境变量指定 Run、模型名、VecNormalize 名和是否打开 GUI。
# (Path/model selection: choose run, model, VecNormalize, and GUI by env variables.)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.environ.get("JUC_RL_ROOT") or os.path.dirname(CURRENT_DIR)

RUN_IDX = int(os.environ.get("JUC_RUN_IDX", "31"))
MODEL_BASENAME = os.environ.get("JUC_MODEL_BASENAME", "ppo_marl_model")
VEC_NORM_BASENAME = os.environ.get("JUC_VECNORM_BASENAME", "vec_normalize_marl")
USE_GUI = os.environ.get("JUC_USE_GUI", "0") == "1"

net_path = os.path.join(ROOT_DIR, "SUMOroutes.net.xml")
route_path = os.path.join(ROOT_DIR, "traffic.random.rou.xml")
run_dir = os.path.join(ROOT_DIR, "saved_models", f"marl_run_{RUN_IDX}")
log_dir = os.path.join(ROOT_DIR, "logs", f"marl_run_{RUN_IDX}")

model_path = os.path.join(run_dir, f"{MODEL_BASENAME}.zip")
vec_norm_path = os.path.join(run_dir, f"{VEC_NORM_BASENAME}.pkl")
diagnostic_dir = os.path.join(log_dir, f"diagnostics_{MODEL_BASENAME}")

# 信号灯状态：用完整灯色字符串判断主路绿灯。
# (Signal states: identify main green by full state strings.)
MAIN_GREEN_STATE = os.environ.get("JUC_MAIN_GREEN_STATE", "rrrrGGggrrrrGGgg")
SIDE_GREEN_STATE = os.environ.get("JUC_SIDE_GREEN_STATE", "GGggrrrrGGggrrrr")
SIGNALS = ("A0", "B0", "C0")

# 车道、路口和路线分组：用于统计主路/支路车辆表现。
# (Lane/signal/route groups: summarize main-road and side-street behavior.)
LANES_BY_SIGNAL = {
    "A0": ["top0A0_0", "B0A0_0", "bottom0A0_0", "left0A0_0"],
    "B0": ["top1B0_0", "C0B0_0", "bottom1B0_0", "A0B0_0"],
    "C0": ["top2C0_0", "right0C0_0", "bottom2C0_0", "B0C0_0"],
}

NEIGHBOR_MAP = {
    "A0": [None, "B0"],
    "B0": ["A0", "C0"],
    "C0": ["B0", None],
}

ROUTE_PREFIXES = (
    "WE_MAIN",
    "EW_MAIN",
    "NS_A0",
    "SN_A0",
    "NS_B0",
    "SN_B0",
    "NS_C0",
    "SN_C0",
)

# 绿波专项分段：统计 A-B、B-C、C-B、B-A 的无停车通过率。
# (Progression segments: measure no-stop rates on corridor links.)
PROGRESSION_SEGMENTS = {
    "WE_MAIN": (("A0_B0", "A0B0"), ("B0_C0", "B0C0")),
    "EW_MAIN": (("C0_B0", "C0B0"), ("B0_A0", "B0A0")),
}
SEGMENT_BY_EDGE = {
    edge_id: (route_prefix, segment_name)
    for route_prefix, segments in PROGRESSION_SEGMENTS.items()
    for segment_name, edge_id in segments
}
# 车队统计参数：用 10 秒窗口估计主路车辆是否成组通过。
# (Platoon settings: estimate grouped arterial passing within a 10-second window.)
PLATOON_WINDOW = 10.0
PLATOON_MIN_COUNT = 3


# 通信观测：与训练脚本保持相同观测维度，确保模型可以正常加载。
# (Communication observation: match training observation dimensions.)
class CommObservationFunction(DefaultObservationFunction):
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

            main_queue = 0
            for lane in neighbor_ts.lanes:
                if "top" not in lane and "bottom" not in lane:
                    main_queue += neighbor_ts.sumo.lane.getLastStepHaltingNumber(lane)
            extra_obs.append(float(main_queue))

        return np.array(np.concatenate([base_obs, extra_obs]), dtype=np.float32)


# 兼容包装器：统一 PettingZoo/Gymnasium/SB3 的接口返回格式。
# (Compatibility wrapper: normalize API return formats.)
class SB3CompatibilityWrapper(VecEnvWrapper):
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


# 车辆路线分组：根据车辆 ID 前缀识别主路或支路来源。
# (Route grouping: identify route group by vehicle ID prefix.)
def route_group(veh_id):
    for prefix in ROUTE_PREFIXES:
        if veh_id.startswith(prefix):
            return prefix
    return "OTHER"


# 路线类型判断：主路为 WE/EW，其他为支路。
# (Route type: WE/EW are main-road routes; others are side routes.)
def route_type(group):
    return "main" if group in ("WE_MAIN", "EW_MAIN") else "side"


# 车道类型判断：top/bottom 车道视为支路。
# (Lane type: top/bottom lanes are side-street lanes.)
def lane_type(lane_id):
    return "side" if "top" in lane_id or "bottom" in lane_id else "main"


# 主路绿灯判断：诊断统计统一使用灯色字符串。
# (Main green check: diagnostics use signal state strings.)
def is_main_green_state(state: str) -> bool:
    return state == MAIN_GREEN_STATE


# 环境构建：加载模型对应的 VecNormalize，并关闭训练模式。
# (Environment builder: load VecNormalize and disable training mode.)
def build_env():
    # 1. 创建测试环境。
    # reward_fn 在诊断中不重要，因为这里不训练，只需要模型根据观测输出动作。
    env = parallel_env(
        net_file=net_path,
        route_file=route_path,
        out_csv_name=None,
        use_gui=USE_GUI,
        num_seconds=3600,
        reward_fn=lambda ts: 0.0,
        observation_class=CommObservationFunction,
    )

    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(
        env,
        num_vec_envs=1,
        num_cpus=1,
        base_class="stable_baselines3",
    )
    env = SB3CompatibilityWrapper(env)

    if not os.path.exists(vec_norm_path):
        raise FileNotFoundError(f"VecNormalize file not found: {vec_norm_path}")

    # 2. 加载训练时的环境感知状态 (VecNormalize)。
    # VecNormalize 保存训练期间观测的均值和方差；测试必须用同一份统计量，否则模型输入尺度会失真。
    env = VecNormalize.load(vec_norm_path, env)
    # 3. 测试时关闭统计量更新，保证诊断结果稳定可复现。
    env.training = False
    # 4. 测试时保留真实奖励尺度，便于把结果和等待时间、排队等物理指标对应。
    env.norm_reward = False
    return env


# 安全调用工具：TraCI 查询失败时返回默认值，避免诊断中断。
# (Safe TraCI call: return defaults on query failure.)
def safe(default, func, *args):
    try:
        return func(*args)
    except Exception:
        return default


# 诊断主流程：运行一轮仿真，采集车辆、路线、车道、灯色和绿波分段数据。
# (Main diagnostic flow: run one simulation and collect all diagnostic data.)
def run_diagnostics():
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # 1. 创建诊断输出目录。
    # 所有 CSV 写入 diagnostics_模型名 文件夹，避免覆盖训练过程产生的 episode CSV。
    os.makedirs(diagnostic_dir, exist_ok=True)

    print(f"Project root: {ROOT_DIR}")
    print(f"Run: marl_run_{RUN_IDX}")
    print(f"Model: {model_path}")
    print(f"VecNormalize: {vec_norm_path}")
    print(f"USE_GUI: {USE_GUI}")
    print(f"Diagnostics will be saved to: {diagnostic_dir}")

    env = build_env()
    # 2. 加载 PPO 模型。
    # 后续 model.predict(..., deterministic=True) 使用确定性策略，适合做报告和复现实验。
    model = PPO.load(model_path, env=env)
    obs = env.reset()

    # 诊断缓存：分别保存车辆生命周期、路线时序、车道时序、信号状态和绿波分段。
    # (Diagnostic buffers: store vehicles, route steps, lane steps, signal states, and progression segments.)
    # 3. 初始化诊断缓存。
    # vehicle_records 记录单车生命周期；route_step_rows/lane_rows 记录时间步统计；
    # state_time 使用真实灯色字符串累计主路/支路绿灯时间。
    vehicle_records = {}
    route_step_rows = []
    lane_rows = []
    phase_rows = []
    phase_time = defaultdict(lambda: defaultdict(float))
    phase_changes = defaultdict(int)
    state_time = defaultdict(lambda: defaultdict(float))
    state_changes = defaultdict(int)
    progression_rows = []
    active_segments = {}
    last_phase = {}
    last_state = {}
    last_time = traci.simulation.getTime()

    # 仿真循环：使用确定性策略运行一整轮，逐步采集诊断数据。
    # (Simulation loop: run one deterministic episode and collect data step by step.)
    while True:
        # 4. 用训练好的策略预测动作，并推进 SUMO 一步。
        # deterministic=True 避免测试时随机采样动作，便于复现实验结果。
        action, _ = model.predict(obs, deterministic=True)
        obs, _, dones, _ = env.step(action)

        sim_time = traci.simulation.getTime()
        dt = max(sim_time - last_time, 0.0)
        last_time = sim_time

        # 信号灯记录：同时保存 phase 和 state，但主路绿灯判断只使用 state。
        # (Signal logging: save both phase and state; main-green logic uses state.)
        for signal_id in SIGNALS:
            # phase 只保留给人工审计；主路绿灯比例必须用 state 字符串判断。
            phase = traci.trafficlight.getPhase(signal_id)
            state = traci.trafficlight.getRedYellowGreenState(signal_id)
            phase_time[signal_id][phase] += dt
            state_time[signal_id][state] += dt
            if signal_id in last_phase and last_phase[signal_id] != phase:
                phase_changes[signal_id] += 1
            if signal_id in last_state and last_state[signal_id] != state:
                state_changes[signal_id] += 1
            last_phase[signal_id] = phase
            last_state[signal_id] = state
            phase_rows.append(
                {
                    "time": sim_time,
                    "signal": signal_id,
                    "phase": phase,
                    "state": state,
                    "is_main_green": int(is_main_green_state(state)),
                }
            )

        route_active = defaultdict(int)
        route_stopped = defaultdict(int)
        route_current_wait = defaultdict(float)
        route_speed_sum = defaultdict(float)

        # 车辆记录：统计每辆车的最大等待、累计等待、速度和所在路线。
        # (Vehicle logging: track each vehicle's waiting, accumulated waiting, speed, and route.)
        for veh_id in traci.vehicle.getIDList():
            # 5. 逐车采集速度、等待时间、累计等待、所在道路和车道。
            # 这些信息用于判断是否存在“主路几乎不等、支路长期不动”的失衡。
            group = route_group(veh_id)
            speed = safe(0.0, traci.vehicle.getSpeed, veh_id)
            waiting = safe(0.0, traci.vehicle.getWaitingTime, veh_id)
            accumulated_waiting = safe(waiting, traci.vehicle.getAccumulatedWaitingTime, veh_id)
            road_id = safe("", traci.vehicle.getRoadID, veh_id)
            lane_id = safe("", traci.vehicle.getLaneID, veh_id)

            route_active[group] += 1
            route_speed_sum[group] += speed
            route_current_wait[group] += waiting
            if speed < 0.1:
                route_stopped[group] += 1

            record = vehicle_records.setdefault(
                veh_id,
                {
                    "vehicle_id": veh_id,
                    "route_group": group,
                    "route_type": route_type(group),
                    "first_seen": sim_time,
                    "last_seen": sim_time,
                    "max_waiting": 0.0,
                    "max_accumulated_waiting": 0.0,
                    "last_road": road_id,
                    "last_lane": lane_id,
                },
            )
            record["last_seen"] = sim_time
            record["max_waiting"] = max(record["max_waiting"], waiting)
            record["max_accumulated_waiting"] = max(
                record["max_accumulated_waiting"],
                accumulated_waiting,
            )
            record["last_road"] = road_id
            record["last_lane"] = lane_id

            # 绿波分段记录：判断主路车辆通过 A-B、B-C、C-B、B-A 时是否停车。
            # (Progression segment logging: detect no-stop passage on arterial links.)
            segment_info = SEGMENT_BY_EDGE.get(road_id)
            active_info = active_segments.get(veh_id)
            if segment_info and segment_info[0] == group:
                # 6. 车辆进入某个主路分段时开始记录。
                # stopped 只要出现低速或等待时间大于 0，就认为该分段不是无停车通过。
                _, segment_name = segment_info
                if active_info is None or active_info["segment"] != segment_name:
                    active_segments[veh_id] = {
                        "vehicle_id": veh_id,
                        "route_group": group,
                        "segment": segment_name,
                        "entry_time": sim_time,
                        "exit_time": sim_time,
                        "min_speed": speed,
                        "stopped": int(speed < 0.1 or waiting > 0.0),
                    }
                else:
                    active_info["exit_time"] = sim_time
                    active_info["min_speed"] = min(active_info["min_speed"], speed)
                    active_info["stopped"] = int(active_info["stopped"] or speed < 0.1 or waiting > 0.0)
            elif active_info is not None:
                # 7. 车辆离开分段时结算该分段记录。
                duration = max(active_info["exit_time"] - active_info["entry_time"], 0.0)
                progression_rows.append(
                    {
                        **active_info,
                        "duration": duration,
                        "no_stop": int(active_info["stopped"] == 0),
                    }
                )
                del active_segments[veh_id]

        # 路线时序统计：按路线组记录每个仿真步的活跃车辆、停车车辆和速度。
        # (Route-step stats: active/stopped vehicles and speed by route group.)
        for group in ROUTE_PREFIXES:
            active = route_active[group]
            route_step_rows.append(
                {
                    "time": sim_time,
                    "route_group": group,
                    "route_type": route_type(group),
                    "active": active,
                    "stopped": route_stopped[group],
                    "current_waiting_sum": route_current_wait[group],
                    "mean_speed": route_speed_sum[group] / active if active else 0.0,
                }
            )

        # 车道时序统计：按车道记录车辆数、停车数和平均速度。
        # (Lane-step stats: vehicles, stopped vehicles, and mean speed by lane.)
        for signal_id, lanes in LANES_BY_SIGNAL.items():
            for lane_id in lanes:
                lane_rows.append(
                    {
                        "time": sim_time,
                        "signal": signal_id,
                        "lane": lane_id,
                        "lane_type": lane_type(lane_id),
                        "vehicles": safe(0, traci.lane.getLastStepVehicleNumber, lane_id),
                        "stopped": safe(0, traci.lane.getLastStepHaltingNumber, lane_id),
                        "mean_speed": safe(0.0, traci.lane.getLastStepMeanSpeed, lane_id),
                    }
                )

        if np.any(dones):
            break

    env.close()

    # 收尾处理：把仍在主路分段上的车辆也写入绿波分段记录。
    # (Finalization: record vehicles still inside a progression segment.)
    for active_info in active_segments.values():
        duration = max(active_info["exit_time"] - active_info["entry_time"], 0.0)
        progression_rows.append(
            {
                **active_info,
                "duration": duration,
                "no_stop": int(active_info["stopped"] == 0),
            }
        )

    df_vehicle = pd.DataFrame(vehicle_records.values())
    df_route_steps = pd.DataFrame(route_step_rows)
    df_lanes = pd.DataFrame(lane_rows)
    df_phases = pd.DataFrame(phase_rows)
    df_progression = pd.DataFrame(progression_rows)

    # 路线汇总：用于判断主路和支路是否存在明显等待或排队失衡。
    # (Route summary: detect imbalance between main-road and side-street service.)
    route_summary = (
        df_vehicle.groupby(["route_group", "route_type"], as_index=False)
        .agg(
            vehicles_seen=("vehicle_id", "count"),
            mean_max_waiting=("max_waiting", "mean"),
            max_max_waiting=("max_waiting", "max"),
            mean_max_accum_waiting=("max_accumulated_waiting", "mean"),
            max_accum_waiting=("max_accumulated_waiting", "max"),
        )
        .merge(
            df_route_steps.groupby("route_group", as_index=False).agg(
                avg_active=("active", "mean"),
                avg_stopped=("stopped", "mean"),
                max_stopped=("stopped", "max"),
                avg_current_wait_sum=("current_waiting_sum", "mean"),
                avg_speed=("mean_speed", "mean"),
            ),
            on="route_group",
            how="left",
        )
    )

    lane_summary = df_lanes.groupby(["signal", "lane", "lane_type"], as_index=False).agg(
        avg_vehicles=("vehicles", "mean"),
        avg_stopped=("stopped", "mean"),
        max_stopped=("stopped", "max"),
        avg_speed=("mean_speed", "mean"),
    )

    # phase 汇总：保留 phase 编号用于审计，但不再用它判断主路绿灯。
    # (Phase summary: keep phase IDs for audit, not for main-green decisions.)
    phase_summary_rows = []
    for signal_id in SIGNALS:
        total = sum(phase_time[signal_id].values())
        for phase, seconds in sorted(phase_time[signal_id].items()):
            phase_summary_rows.append(
                {
                    "signal": signal_id,
                    "phase": phase,
                    "seconds": seconds,
                    "ratio": seconds / total if total else 0.0,
                    "phase_changes": phase_changes[signal_id],
                }
            )
    phase_summary = pd.DataFrame(phase_summary_rows)

    # state 汇总：真正用于判断主路绿灯比例和信号切换次数。
    # (State summary: actual source for main-green ratio and state changes.)
    state_summary_rows = []
    for signal_id in SIGNALS:
        # 8. state_summary 是判断主路绿灯比例的正式证据。
        # is_main_green=1 的 state 对应主路绿灯；ratio 就是该状态占整轮仿真的时间比例。
        total = sum(state_time[signal_id].values())
        for state, seconds in sorted(state_time[signal_id].items()):
            state_summary_rows.append(
                {
                    "signal": signal_id,
                    "state": state,
                    "is_main_green": int(is_main_green_state(state)),
                    "seconds": seconds,
                    "ratio": seconds / total if total else 0.0,
                    "state_changes": state_changes[signal_id],
                }
            )
    state_summary = pd.DataFrame(state_summary_rows)

    # 绿波汇总：计算主路分段无停车通过率。
    # (Progression summary: calculate no-stop passage rates on arterial segments.)
    if df_progression.empty:
        progression_summary = pd.DataFrame(
            columns=[
                "route_group",
                "segment",
                "vehicles_seen",
                "no_stop_vehicles",
                "no_stop_rate",
                "mean_duration",
                "mean_min_speed",
            ]
        )
        platoon_summary = pd.DataFrame(
            columns=[
                "route_group",
                "segment",
                "vehicles_seen",
                "max_10s_passes",
                "platoon_windows",
                "platoon_window_rate",
            ]
        )
    else:
        progression_summary = df_progression.groupby(["route_group", "segment"], as_index=False).agg(
            vehicles_seen=("vehicle_id", "count"),
            no_stop_vehicles=("no_stop", "sum"),
            no_stop_rate=("no_stop", "mean"),
            mean_duration=("duration", "mean"),
            mean_min_speed=("min_speed", "mean"),
        )

        # 车队汇总：统计 10 秒窗口内主路车辆成组通过情况。
        # (Platoon summary: count grouped arterial passages within a 10-second window.)
        platoon_rows = []
        for (group, segment), df_segment in df_progression.groupby(["route_group", "segment"]):
            exit_times = sorted(df_segment["exit_time"].tolist())
            best_count = 0
            platoon_windows = 0
            for idx, start_time in enumerate(exit_times):
                count = sum(1 for t in exit_times[idx:] if t - start_time <= PLATOON_WINDOW)
                best_count = max(best_count, count)
                if count >= PLATOON_MIN_COUNT:
                    platoon_windows += 1
            platoon_rows.append(
                {
                    "route_group": group,
                    "segment": segment,
                    "vehicles_seen": len(exit_times),
                    "max_10s_passes": best_count,
                    "platoon_windows": platoon_windows,
                    "platoon_window_rate": platoon_windows / len(exit_times) if exit_times else 0.0,
                }
            )
        platoon_summary = pd.DataFrame(platoon_rows)

    # 结果保存：所有诊断表写入 diagnostics_xxx 文件夹。
    # (Save outputs: write all diagnostic tables to the diagnostics folder.)
    df_vehicle.to_csv(os.path.join(diagnostic_dir, "vehicle_records.csv"), index=False)
    df_route_steps.to_csv(os.path.join(diagnostic_dir, "route_steps.csv"), index=False)
    df_lanes.to_csv(os.path.join(diagnostic_dir, "lane_steps.csv"), index=False)
    df_phases.to_csv(os.path.join(diagnostic_dir, "phase_steps.csv"), index=False)
    df_progression.to_csv(os.path.join(diagnostic_dir, "progression_records.csv"), index=False)
    route_summary.to_csv(os.path.join(diagnostic_dir, "route_summary.csv"), index=False)
    lane_summary.to_csv(os.path.join(diagnostic_dir, "lane_summary.csv"), index=False)
    phase_summary.to_csv(os.path.join(diagnostic_dir, "phase_summary.csv"), index=False)
    state_summary.to_csv(os.path.join(diagnostic_dir, "state_summary.csv"), index=False)
    progression_summary.to_csv(os.path.join(diagnostic_dir, "progression_summary.csv"), index=False)
    platoon_summary.to_csv(os.path.join(diagnostic_dir, "platoon_summary.csv"), index=False)

    print("\nRoute summary:")
    print(route_summary.sort_values(["route_type", "mean_max_accum_waiting"], ascending=[True, False]).to_string(index=False))

    print("\nLane summary:")
    print(lane_summary.sort_values(["lane_type", "avg_stopped"], ascending=[True, False]).to_string(index=False))

    print("\nPhase summary:")
    print(phase_summary.to_string(index=False))

    print("\nState summary:")
    print(state_summary.to_string(index=False))

    print("\nProgression summary:")
    print(progression_summary.to_string(index=False))

    print("\nPlatoon summary:")
    print(platoon_summary.to_string(index=False))


# 主程序：执行诊断并保存 CSV 表格。
# (Main entry: run diagnostics and save CSV tables.)
if __name__ == "__main__":
    run_diagnostics()
