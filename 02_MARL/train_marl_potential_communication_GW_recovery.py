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


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CURRENT_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from generate_Random_Traffic import generate_route_file


net_path = os.path.join(ROOT_DIR, "SUMOroutes.net.xml")
route_path = os.path.join(ROOT_DIR, "traffic.random.rou.xml")

SEED = 8848

# Recommended next experiment:
#   1) Keep LOAD_MODEL_RUN_IDX = 30.
#   2) Start from the model that visibly produced straight main-road trajectories.
#      If you want to start from the final model, use "ppo_marl_model" instead.
LOAD_MODEL_RUN_IDX = 30
LOAD_MODEL_BASENAME = "ppo_marl_model_gw_stage1"
LOAD_VECNORM_BASENAME = "vec_normalize_marl_gw_stage1"

TOTAL_TIMESTEPS = 450_000

MAIN_GREEN_PHASE = 2
SIDE_GREEN_PHASE = 0
YELLOW_PHASES = {1, 3}

ETA_HORIZON = 22.0
ETA_DECAY = 8.0
NEAR_ETA = 7.0
MIN_ETA_SPEED = 2.0
STOPLINE_DISTANCE = 30.0
FREE_FLOW_SPEED = 5.0
MAX_PRESSURE = 5.0

# The recovery profile keeps green-wave shaping, but PBRS and side recovery dominate.
PBRS_WEIGHT = 1.25
GREEN_PRESSURE_WEIGHT = 0.045
RED_PRESSURE_WEIGHT = 0.055
FREE_FLOW_WEIGHT = 0.025

SIDE_QUEUE_THRESHOLD = 5.0
SIDE_QUEUE_WEIGHT = 0.060
SIDE_RESCUE_REWARD = 0.035

MAX_MAIN_GREEN_SECONDS = 38.0
MAIN_OVERTIME_WEIGHT = 0.075
IDLE_MAIN_GREEN_WEIGHT = 0.080
MIN_MAIN_PRESSURE_FOR_HOLD = 0.35

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


def linear_schedule_with_min(initial_value: float, min_value: float) -> Callable[[float], float]:
    def func(progress_remaining: float) -> float:
        return min_value + progress_remaining * (initial_value - min_value)

    return func


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


def safe_call(default, func, *args):
    try:
        return func(*args)
    except Exception:
        return default


def current_step(traffic_signal) -> float:
    return float(getattr(traffic_signal.env, "sim_step", 0.0))


def is_new_episode(traffic_signal) -> bool:
    delta_time = float(getattr(traffic_signal.env, "delta_time", 5.0))
    return current_step(traffic_signal) <= delta_time


def update_main_green_timer(traffic_signal, phase):
    if is_new_episode(traffic_signal):
        for attr in ("main_green_start", "last_potential"):
            if hasattr(traffic_signal, attr):
                delattr(traffic_signal, attr)

    if phase == MAIN_GREEN_PHASE:
        if not hasattr(traffic_signal, "main_green_start"):
            traffic_signal.main_green_start = current_step(traffic_signal)
    elif hasattr(traffic_signal, "main_green_start"):
        delattr(traffic_signal, "main_green_start")


def main_green_elapsed(traffic_signal) -> float:
    if not hasattr(traffic_signal, "main_green_start"):
        return 0.0
    return max(0.0, current_step(traffic_signal) - traffic_signal.main_green_start)


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
        "free_flow_count": float(min(free_flow_count, 6.0)),
    }


def main_arrival_metrics(traffic_signal) -> dict:
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


def side_queue(traffic_signal) -> float:
    queue = 0.0
    for lane in traffic_signal.lanes:
        if "top" in lane or "bottom" in lane:
            queue += safe_call(0, traffic_signal.sumo.lane.getLastStepHaltingNumber, lane)
    return float(queue)


def recovery_reward(traffic_signal):
    phase = traffic_signal.sumo.trafficlight.getPhase(traffic_signal.id)
    update_main_green_timer(traffic_signal, phase)

    is_main_green = phase == MAIN_GREEN_PHASE
    is_side_green = phase == SIDE_GREEN_PHASE
    is_yellow = phase in YELLOW_PHASES

    reward = PBRS_WEIGHT * pbrs_reward(traffic_signal)

    metrics = main_arrival_metrics(traffic_signal)
    pressure = metrics["pressure"]
    sq = side_queue(traffic_signal)

    if pressure > 0:
        if is_main_green:
            reward += GREEN_PRESSURE_WEIGHT * pressure
            reward += FREE_FLOW_WEIGHT * metrics["free_flow_count"]
        elif not is_yellow:
            reward -= RED_PRESSURE_WEIGHT * pressure

    if is_main_green:
        elapsed = main_green_elapsed(traffic_signal)
        if elapsed > MAX_MAIN_GREEN_SECONDS:
            reward -= MAIN_OVERTIME_WEIGHT * (elapsed - MAX_MAIN_GREEN_SECONDS)

        if pressure < MIN_MAIN_PRESSURE_FOR_HOLD and sq > 0:
            reward -= IDLE_MAIN_GREEN_WEIGHT * min(sq, 10.0)

        if sq > SIDE_QUEUE_THRESHOLD:
            reward -= SIDE_QUEUE_WEIGHT * (sq - SIDE_QUEUE_THRESHOLD)

    if is_side_green and sq > SIDE_QUEUE_THRESHOLD:
        reward += SIDE_RESCUE_REWARD * min(sq - SIDE_QUEUE_THRESHOLD, 10.0)

    return reward


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


def build_env(csv_base_path, vec_norm_path):
    env = parallel_env(
        net_file=net_path,
        route_file=route_path,
        out_csv_name=csv_base_path,
        use_gui=False,
        num_seconds=3600,
        reward_fn=recovery_reward,
        observation_class=CommObservationFunction,
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
    env = ss.concat_vec_envs_v1(env, num_vec_envs=1, num_cpus=1, base_class="stable_baselines3")
    env = SB3CompatibilityWrapper(env)
    env = VecMonitor(env)

    if not os.path.exists(vec_norm_path):
        raise FileNotFoundError(f"VecNormalize file not found: {vec_norm_path}")

    env = VecNormalize.load(vec_norm_path, env)
    env.training = True
    env.norm_reward = False
    return env


def source_paths():
    source_dir = os.path.join(ROOT_DIR, "saved_models", f"marl_run_{LOAD_MODEL_RUN_IDX}")
    model_path = os.path.join(source_dir, f"{LOAD_MODEL_BASENAME}.zip")
    vec_norm_path = os.path.join(source_dir, f"{LOAD_VECNORM_BASENAME}.pkl")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not os.path.exists(vec_norm_path):
        raise FileNotFoundError(f"VecNormalize file not found: {vec_norm_path}")

    return model_path, vec_norm_path


if __name__ == "__main__":
    print("Initializing MARL green-wave recovery fine-tuning...")

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

    generate_route_file_in_root()
    env = build_env(csv_base_path, vec_norm_path)
    model = PPO.load(
        model_path,
        env=env,
        tensorboard_log=os.path.join(ROOT_DIR, "logs", "ppo_marl_tb"),
    )
    model.learning_rate = linear_schedule_with_min(1e-4, 2e-5)
    model.ent_coef = 0.01

    print("=" * 70)
    print("Observation space:", model.policy.observation_space)
    print("Action space:", model.policy.action_space)
    print(f"Source run: marl_run_{LOAD_MODEL_RUN_IDX}")
    print(f"Source model: {LOAD_MODEL_BASENAME}")
    print(f"Main overtime cap: {MAX_MAIN_GREEN_SECONDS}s")
    print(f"Side queue threshold: {SIDE_QUEUE_THRESHOLD}")
    print("=" * 70)

    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path=os.path.join(run_save_dir, "checkpoints"),
        name_prefix="rl_model",
    )

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=checkpoint_callback,
        tb_log_name=f"run_{run_idx}_gw_recovery",
        reset_num_timesteps=True,
    )

    model.save(os.path.join(run_save_dir, "ppo_marl_model"))
    env.save(os.path.join(run_save_dir, "vec_normalize_marl.pkl"))
    model.save(os.path.join(run_save_dir, "ppo_marl_model_gw_recovery"))
    env.save(os.path.join(run_save_dir, "vec_normalize_marl_gw_recovery.pkl"))

    env.close()
    print(f"Recovery fine-tuning finished. Outputs saved to: {run_save_dir}")
