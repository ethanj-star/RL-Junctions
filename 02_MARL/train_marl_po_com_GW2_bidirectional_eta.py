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


def linear_schedule_with_min(initial_value: float, min_value: float) -> Callable[[float], float]:
    def func(progress_remaining: float) -> float:
        return min_value + progress_remaining * (initial_value - min_value)

    return func


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CURRENT_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from generate_Random_Traffic import generate_route_file

net_path = os.path.join(ROOT_DIR, "SUMOroutes.net.xml")
route_path = os.path.join(ROOT_DIR, "traffic.random.rou.xml")

# In SUMOroutes.net.xml, phase 2 is the east-west corridor green phase.
# Phase 0 serves the north-south side approaches.
MAIN_GREEN_PHASE = 2
YELLOW_PHASES = {1, 3}

ETA_HORIZON = 25.0
ETA_NEAR_WINDOW = 8.0
ETA_DECAY = 8.0
MIN_ETA_SPEED = 2.0
FREE_FLOW_SPEED = 5.0
STOPLINE_DISTANCE = 25.0

QUEUE_WEIGHT = 0.04
POTENTIAL_QUEUE_NORM = 50.0
ETA_GREEN_WEIGHT = 0.18
ETA_RED_WEIGHT = 0.24
FREE_FLOW_WEIGHT = 0.08
SIDE_QUEUE_WEIGHT = 0.03
SIDE_QUEUE_THRESHOLD = 6.0
LONG_MAIN_GREEN_START = 35.0
IDLE_MAIN_GREEN_PENALTY = 0.04
SPILLBACK_WEIGHT = 0.18
SPILLBACK_OCC_THRESHOLD = 0.45

# Incoming main-road edges for each signal. These include both corridor directions.
MAIN_APPROACH_EDGES = {
    "A0": ["left0A0", "B0A0"],
    "B0": ["A0B0", "C0B0"],
    "C0": ["B0C0", "right0C0"],
}

# Main-road outgoing edges used to avoid releasing vehicles into blocked links.
MAIN_DOWNSTREAM_EDGES = {
    "A0": ["A0B0", "A0left0"],
    "B0": ["B0C0", "B0A0"],
    "C0": ["C0right0", "C0B0"],
}

NEIGHBOR_MAP = {
    "A0": [None, "B0"],
    "B0": ["A0", "C0"],
    "C0": ["B0", None],
}


@contextmanager
def pushd(path: str):
    old_cwd = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old_cwd)


def generate_route_file_in_root():
    with pushd(ROOT_DIR):
        generate_route_file()


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


def _safe(default, func, *args):
    try:
        return func(*args)
    except Exception:
        return default


def _lane_id(edge_id: str) -> str:
    return f"{edge_id}_0"


def _current_step(traffic_signal) -> float:
    return float(getattr(traffic_signal.env, "sim_step", 0.0))


def _is_new_episode(traffic_signal) -> bool:
    current_step = _current_step(traffic_signal)
    delta_time = float(getattr(traffic_signal.env, "delta_time", 5.0))
    return current_step <= delta_time


def _update_phase_tracking(traffic_signal):
    if _is_new_episode(traffic_signal):
        for attr in ("my_green_start", "last_phase", "last_potential"):
            if hasattr(traffic_signal, attr):
                delattr(traffic_signal, attr)

    current_phase = traffic_signal.sumo.trafficlight.getPhase(traffic_signal.id)
    is_main_green = current_phase == MAIN_GREEN_PHASE

    if not hasattr(traffic_signal, "last_phase"):
        traffic_signal.last_phase = current_phase
    traffic_signal.last_phase = current_phase

    if is_main_green:
        if not hasattr(traffic_signal, "my_green_start"):
            traffic_signal.my_green_start = _current_step(traffic_signal)
    elif hasattr(traffic_signal, "my_green_start"):
        delattr(traffic_signal, "my_green_start")

    return current_phase, is_main_green


def _green_duration_norm(traffic_signal) -> float:
    if not hasattr(traffic_signal, "my_green_start"):
        return 0.0
    return min((_current_step(traffic_signal) - traffic_signal.my_green_start) / 60.0, 1.0)


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


def _side_queue(traffic_signal) -> float:
    side_queue = 0.0
    for lane_id in traffic_signal.lanes:
        if "top" in lane_id or "bottom" in lane_id:
            side_queue += float(_safe(0, traffic_signal.sumo.lane.getLastStepHaltingNumber, lane_id))
    return side_queue


def _main_queue(traffic_signal) -> float:
    main_queue = 0.0
    for lane_id in traffic_signal.lanes:
        if "top" not in lane_id and "bottom" not in lane_id:
            main_queue += float(_safe(0, traffic_signal.sumo.lane.getLastStepHaltingNumber, lane_id))
    return main_queue


def _downstream_blockage(traffic_signal) -> float:
    blockage = 0.0
    for edge_id in MAIN_DOWNSTREAM_EDGES.get(traffic_signal.id, []):
        metrics = _edge_metrics(traffic_signal.sumo, edge_id)
        occupancy_pressure = max(0.0, metrics["occupancy"] - SPILLBACK_OCC_THRESHOLD)
        queue_pressure = min(metrics["halting"] / 12.0, 1.0)
        blockage += max(occupancy_pressure, queue_pressure)
    return min(blockage, 1.0)


def custom_bidirectional_green_wave_reward(traffic_signal):
    current_phase, is_main_green = _update_phase_tracking(traffic_signal)
    is_yellow = current_phase in YELLOW_PHASES

    total_queue = float(traffic_signal.get_total_queued())
    base_penalty = -QUEUE_WEIGHT * total_queue
    phi_current = -total_queue / POTENTIAL_QUEUE_NORM

    if getattr(traffic_signal, "last_potential", None) is None or _is_new_episode(traffic_signal):
        traffic_signal.last_potential = phi_current
        shaping_reward = 0.0
    else:
        gamma = 0.99
        shaping_reward = gamma * phi_current - traffic_signal.last_potential
        traffic_signal.last_potential = phi_current

    arrival = _main_arrival_metrics(traffic_signal)
    arrival_pressure = min(arrival["pressure"], 6.0)
    side_q = _side_queue(traffic_signal)
    downstream_block = _downstream_blockage(traffic_signal)

    eta_reward = 0.0
    if arrival_pressure > 0.0:
        if is_main_green:
            eta_reward += ETA_GREEN_WEIGHT * arrival_pressure
        elif not is_yellow:
            eta_reward -= ETA_RED_WEIGHT * arrival_pressure

    free_flow_reward = 0.0
    if is_main_green:
        free_flow_reward += FREE_FLOW_WEIGHT * min(arrival["free_flow_count"], 4.0)

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

    spillback_penalty = 0.0
    if is_main_green and downstream_block > 0.0:
        spillback_penalty -= SPILLBACK_WEIGHT * downstream_block * max(arrival_pressure, 1.0)

    return (
        base_penalty
        + shaping_reward
        + eta_reward
        + free_flow_reward
        + idle_green_penalty
        + fairness_penalty
        + spillback_penalty
    )


class BidirectionalETAObservationFunction(DefaultObservationFunction):
    EXTRA_DIM = 18

    def __init__(self, ts):
        super().__init__(ts)

    def observation_space(self):
        base_space = super().observation_space()
        base_dim = base_space.shape[0]
        new_dim = base_dim + self.EXTRA_DIM
        return spaces.Box(
            low=np.zeros(new_dim, dtype=np.float32),
            high=np.ones(new_dim, dtype=np.float32) * np.inf,
        )

    def _neighbor_features(self):
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
            current_phase = neighbor_ts.sumo.trafficlight.getPhase(neighbor_id)
            is_main_green = 1.0 if current_phase == MAIN_GREEN_PHASE else 0.0
            green_duration_norm = _green_duration_norm(neighbor_ts)
            features.extend([queue_norm, is_main_green, green_duration_norm])

        return features

    def _eta_features(self):
        features = []
        edge_ids = MAIN_APPROACH_EDGES.get(self.ts.id, [])
        for edge_id in edge_ids[:2]:
            metrics = _edge_metrics(self.ts.sumo, edge_id)
            pressure_norm = min(metrics["pressure"] / 4.0, 1.0)
            near_norm = min(metrics["near_count"] / 6.0, 1.0)
            halt_norm = min(metrics["halting"] / 12.0, 1.0)
            min_eta_norm = min(metrics["min_eta"] / ETA_HORIZON, 1.0)
            features.extend([pressure_norm, near_norm, halt_norm, min_eta_norm])

        while len(features) < 8:
            features.extend([0.0, 0.0, 0.0, 1.0])

        return features[:8]

    def _local_features(self):
        current_phase = self.ts.sumo.trafficlight.getPhase(self.ts.id)
        is_main_green = 1.0 if current_phase == MAIN_GREEN_PHASE else 0.0
        side_queue_norm = min(_side_queue(self.ts) / 30.0, 1.0)
        downstream_block = _downstream_blockage(self.ts)
        return [is_main_green, _green_duration_norm(self.ts), side_queue_norm, downstream_block]

    def __call__(self):
        base_obs = super().__call__()
        extra_obs = self._neighbor_features() + self._eta_features() + self._local_features()
        final_obs = np.concatenate([base_obs, np.array(extra_obs, dtype=np.float32)])
        return np.array(final_obs, dtype=np.float32)


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
        total_timesteps=1000000,
        callback=checkpoint_callback,
        tb_log_name=f"run_{run_idx}_bidirectional_eta",
    )

    model.save(os.path.join(run_save_dir, "ppo_marl_model_bidirectional_eta"))
    env.save(os.path.join(run_save_dir, "vec_normalize_marl_bidirectional_eta.pkl"))

    env.close()
    print("Bidirectional ETA green-wave MARL training finished.")
