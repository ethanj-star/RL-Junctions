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
TOTAL_TIMESTEPS = 300_000

# In SUMOroutes.net.xml, phase 2 is the east-west main-road phase.
MAIN_GREEN_PHASE = 2
YELLOW_PHASES = {1, 3}

# Keep green-wave shaping deliberately small. PBRS remains the main objective.
GREEN_WAVE_WEIGHT = 0.03
ETA_HORIZON = 20.0
ETA_DECAY = 8.0
MIN_ETA_SPEED = 2.0
MAX_ARRIVAL_PRESSURE = 4.0

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
        shaping_reward = (gamma * phi_current) - traffic_signal.last_potential
        traffic_signal.last_potential = phi_current

    beta = 100.0
    final_reward = base_reward + (beta * shaping_reward)
    return final_reward / 100.0


def edge_arrival_pressure(sumo, edge_id: str) -> float:
    lane_id = f"{edge_id}_0"
    lane_length = float(safe_call(0.0, sumo.lane.getLength, lane_id))
    vehicle_ids = list(safe_call([], sumo.lane.getLastStepVehicleIDs, lane_id))

    pressure = 0.0
    for veh_id in vehicle_ids:
        lane_pos = float(safe_call(0.0, sumo.vehicle.getLanePosition, veh_id))
        speed = float(safe_call(0.0, sumo.vehicle.getSpeed, veh_id))
        distance_to_stopline = max(lane_length - lane_pos, 0.0)
        eta = distance_to_stopline / max(speed, MIN_ETA_SPEED)

        if eta <= ETA_HORIZON:
            pressure += np.exp(-eta / ETA_DECAY)

    return float(min(pressure, MAX_ARRIVAL_PRESSURE))


def lightweight_green_wave_reward(traffic_signal) -> float:
    current_phase = traffic_signal.sumo.trafficlight.getPhase(traffic_signal.id)
    is_main_green = current_phase == MAIN_GREEN_PHASE
    is_yellow = current_phase in YELLOW_PHASES

    arrival_pressure = 0.0
    for edge_id in MAIN_APPROACH_EDGES.get(traffic_signal.id, []):
        arrival_pressure += edge_arrival_pressure(traffic_signal.sumo, edge_id)
    arrival_pressure = min(arrival_pressure, MAX_ARRIVAL_PRESSURE)

    if arrival_pressure <= 0.0:
        return 0.0

    if is_main_green:
        return GREEN_WAVE_WEIGHT * arrival_pressure

    if is_yellow:
        return 0.0

    return -0.5 * GREEN_WAVE_WEIGHT * arrival_pressure


def pbrs_green_wave_lite_reward(traffic_signal):
    return pbrs_reward(traffic_signal) + lightweight_green_wave_reward(traffic_signal)


class CommObservationFunction(DefaultObservationFunction):
    """
    Keep the same compact communication structure as train_marl_potential_communication.py:
    base observation + two neighbor main-road queue values.
    """

    def __init__(self, ts):
        super().__init__(ts)

    def observation_space(self):
        base_space = super().observation_space()
        base_dim = base_space.shape[0]
        new_dim = base_dim + 2

        return spaces.Box(
            low=np.zeros(new_dim, dtype=np.float32),
            high=np.ones(new_dim, dtype=np.float32) * np.inf,
        )

    def __call__(self):
        base_obs = super().__call__()
        my_id = self.ts.id
        neighbors = NEIGHBOR_MAP.get(my_id, [None, None])

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

        final_obs = np.concatenate([base_obs, extra_obs])
        return np.array(final_obs, dtype=np.float32)


if __name__ == "__main__":
    print("Initializing MARL PBRS + lightweight green-wave training...")

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
    tensorboard_log_path = os.path.join(logs_base, "ppo_marl_tb")

    print("Generating initial random bidirectional traffic...")
    generate_route_file_in_root()

    env = parallel_env(
        net_file=net_path,
        route_file=route_path,
        out_csv_name=csv_base_path,
        use_gui=False,
        num_seconds=3600,
        reward_fn=pbrs_green_wave_lite_reward,
        observation_class=CommObservationFunction,
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
        ent_coef=0.03,
        target_kl=0.05,
        verbose=1,
        device="cpu",
        tensorboard_log=tensorboard_log_path,
    )

    print("=" * 60)
    print("Observation space:", model.policy.observation_space)
    print("Action space:", model.policy.action_space)
    print(f"Main-road green phase: {MAIN_GREEN_PHASE}")
    print(f"Green-wave weight: {GREEN_WAVE_WEIGHT}")
    print("=" * 60)

    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path=os.path.join(run_save_dir, "checkpoints"),
        name_prefix="rl_model",
    )

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=checkpoint_callback,
        tb_log_name=f"run_{run_idx}_gw_lite",
    )

    # Generic names keep compatibility with existing plotting/testing scripts.
    model.save(os.path.join(run_save_dir, "ppo_marl_model"))
    env.save(os.path.join(run_save_dir, "vec_normalize_marl.pkl"))

    # Explicit names make this run easy to identify later.
    model.save(os.path.join(run_save_dir, "ppo_marl_model_gw_lite"))
    env.save(os.path.join(run_save_dir, "vec_normalize_marl_gw_lite.pkl"))

    env.close()
    print(f"Training finished. Outputs saved to: {run_save_dir}")
