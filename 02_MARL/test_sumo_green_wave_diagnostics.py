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

MAIN_GREEN_PHASE = int(os.environ.get("JUC_MAIN_GREEN_PHASE", "0"))
SIGNALS = ("A0", "B0", "C0")

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


def route_group(veh_id):
    for prefix in ROUTE_PREFIXES:
        if veh_id.startswith(prefix):
            return prefix
    return "OTHER"


def route_type(group):
    return "main" if group in ("WE_MAIN", "EW_MAIN") else "side"


def lane_type(lane_id):
    return "side" if "top" in lane_id or "bottom" in lane_id else "main"


def build_env():
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

    env = VecNormalize.load(vec_norm_path, env)
    env.training = False
    env.norm_reward = False
    return env


def safe(default, func, *args):
    try:
        return func(*args)
    except Exception:
        return default


def run_diagnostics():
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    os.makedirs(diagnostic_dir, exist_ok=True)

    print(f"Project root: {ROOT_DIR}")
    print(f"Run: marl_run_{RUN_IDX}")
    print(f"Model: {model_path}")
    print(f"VecNormalize: {vec_norm_path}")
    print(f"USE_GUI: {USE_GUI}")
    print(f"Diagnostics will be saved to: {diagnostic_dir}")

    env = build_env()
    model = PPO.load(model_path, env=env)
    obs = env.reset()

    vehicle_records = {}
    route_step_rows = []
    lane_rows = []
    phase_rows = []
    phase_time = defaultdict(lambda: defaultdict(float))
    phase_changes = defaultdict(int)
    last_phase = {}
    last_time = traci.simulation.getTime()

    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, dones, _ = env.step(action)

        sim_time = traci.simulation.getTime()
        dt = max(sim_time - last_time, 0.0)
        last_time = sim_time

        for signal_id in SIGNALS:
            phase = traci.trafficlight.getPhase(signal_id)
            phase_time[signal_id][phase] += dt
            if signal_id in last_phase and last_phase[signal_id] != phase:
                phase_changes[signal_id] += 1
            last_phase[signal_id] = phase
            phase_rows.append(
                {
                    "time": sim_time,
                    "signal": signal_id,
                    "phase": phase,
                    "state": traci.trafficlight.getRedYellowGreenState(signal_id),
                    "is_main_green": int(phase == MAIN_GREEN_PHASE),
                }
            )

        route_active = defaultdict(int)
        route_stopped = defaultdict(int)
        route_current_wait = defaultdict(float)
        route_speed_sum = defaultdict(float)

        for veh_id in traci.vehicle.getIDList():
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

    df_vehicle = pd.DataFrame(vehicle_records.values())
    df_route_steps = pd.DataFrame(route_step_rows)
    df_lanes = pd.DataFrame(lane_rows)
    df_phases = pd.DataFrame(phase_rows)

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

    df_vehicle.to_csv(os.path.join(diagnostic_dir, "vehicle_records.csv"), index=False)
    df_route_steps.to_csv(os.path.join(diagnostic_dir, "route_steps.csv"), index=False)
    df_lanes.to_csv(os.path.join(diagnostic_dir, "lane_steps.csv"), index=False)
    df_phases.to_csv(os.path.join(diagnostic_dir, "phase_steps.csv"), index=False)
    route_summary.to_csv(os.path.join(diagnostic_dir, "route_summary.csv"), index=False)
    lane_summary.to_csv(os.path.join(diagnostic_dir, "lane_summary.csv"), index=False)
    phase_summary.to_csv(os.path.join(diagnostic_dir, "phase_summary.csv"), index=False)

    print("\nRoute summary:")
    print(route_summary.sort_values(["route_type", "mean_max_accum_waiting"], ascending=[True, False]).to_string(index=False))

    print("\nLane summary:")
    print(lane_summary.sort_values(["lane_type", "avg_stopped"], ascending=[True, False]).to_string(index=False))

    print("\nPhase summary:")
    print(phase_summary.to_string(index=False))


if __name__ == "__main__":
    run_diagnostics()
