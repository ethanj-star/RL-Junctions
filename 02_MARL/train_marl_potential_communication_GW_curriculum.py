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

# Optional warm start. Set to an integer, for example 29, to continue from a previous run.
LOAD_MODEL_RUN_IDX = None

# Stage 1 deliberately sacrifices some waiting-time optimality to make the corridor visibly green.
# Stage 2 restores PBRS as the main objective while keeping a small progression term.
STAGE1_TIMESTEPS = 250_000
STAGE2_TIMESTEPS = 350_000

MAIN_GREEN_PHASE = 2
YELLOW_PHASES = {1, 3}
SIGNALS = ("A0", "B0", "C0")

ETA_HORIZON = 22.0
ETA_DECAY = 8.0
NEAR_ETA = 7.0
MIN_ETA_SPEED = 2.0
STOPLINE_DISTANCE = 30.0
FREE_FLOW_SPEED = 5.0
MAX_PRESSURE = 5.0

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
        "free_flow_count": float(free_flow_count),
    }


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


def all_signals_main_green(traffic_signal) -> bool:
    for ts_id in SIGNALS:
        phase = traffic_signal.sumo.trafficlight.getPhase(ts_id)
        if phase != MAIN_GREEN_PHASE:
            return False
    return True


def side_queue(traffic_signal) -> float:
    queue = 0.0
    for lane in traffic_signal.lanes:
        if "top" in lane or "bottom" in lane:
            queue += safe_call(0, traffic_signal.sumo.lane.getLastStepHaltingNumber, lane)
    return float(queue)


def green_wave_curriculum_reward(traffic_signal):
    profile = REWARD_PROFILES[REWARD_STAGE]
    base = profile["pbrs"] * pbrs_reward(traffic_signal)

    phase = traffic_signal.sumo.trafficlight.getPhase(traffic_signal.id)
    is_main_green = phase == MAIN_GREEN_PHASE
    is_yellow = phase in YELLOW_PHASES

    metrics = main_arrival_pressure(traffic_signal)
    pressure = metrics["pressure"]

    progression = 0.0
    if pressure > 0:
        if is_main_green:
            progression += profile["green_pressure"] * pressure
            progression += profile["free_flow"] * metrics["free_flow_count"]
        elif not is_yellow:
            progression -= profile["red_pressure"] * pressure

    if all_signals_main_green(traffic_signal):
        corridor_pressure = 0.0
        for ts in traffic_signal.env.traffic_signals.values():
            corridor_pressure += main_arrival_pressure(ts)["pressure"]
        corridor_pressure = min(corridor_pressure / 6.0, 1.0)
        progression += profile["all_green"] * max(corridor_pressure, 0.25)

    fairness = -profile["side_queue"] * max(0.0, side_queue(traffic_signal) - 10.0)

    return base + progression + fairness


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


def build_env(csv_base_path, vec_norm_path=None):
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
        print("Regenerating random bidirectional traffic for this episode...")
        generate_route_file_in_root()
        return original_reset(*args, **kwargs)

    env.reset = custom_reset
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, num_vec_envs=1, num_cpus=1, base_class="stable_baselines3")
    env = SB3CompatibilityWrapper(env)
    env = VecMonitor(env)

    if vec_norm_path is not None:
        print(f"Loading VecNormalize statistics from: {vec_norm_path}")
        env = VecNormalize.load(vec_norm_path, env)
        env.training = True
        env.norm_reward = False
        return env

    return VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.0)


def maybe_load_model(env, model_path=None):
    if model_path is None:
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
    return PPO.load(model_path, env=env, tensorboard_log=os.path.join(ROOT_DIR, "logs", "ppo_marl_tb"))


if __name__ == "__main__":
    print("Initializing MARL green-wave curriculum training...")

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

    warm_model_path, warm_vec_norm_path = find_warm_start_paths()
    env = build_env(csv_base_path, warm_vec_norm_path)
    model = maybe_load_model(env, warm_model_path)

    print("=" * 70)
    print("Observation space:", model.policy.observation_space)
    print("Action space:", model.policy.action_space)
    print(f"Main-road green phase: {MAIN_GREEN_PHASE}")
    print(f"Stage 1 timesteps: {STAGE1_TIMESTEPS}, profile: aggressive")
    print(f"Stage 2 timesteps: {STAGE2_TIMESTEPS}, profile: balanced")
    print("=" * 70)

    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path=os.path.join(run_save_dir, "checkpoints"),
        name_prefix="rl_model",
    )

    REWARD_STAGE = "aggressive"
    print("Stage 1: aggressive green-wave shaping...")
    model.learn(
        total_timesteps=STAGE1_TIMESTEPS,
        callback=checkpoint_callback,
        tb_log_name=f"run_{run_idx}_gw_curriculum_stage1",
        reset_num_timesteps=True,
    )
    model.save(os.path.join(run_save_dir, "ppo_marl_model_gw_stage1"))
    env.save(os.path.join(run_save_dir, "vec_normalize_marl_gw_stage1.pkl"))

    REWARD_STAGE = "balanced"
    print("Stage 2: balanced fine-tuning back to PBRS...")
    model.learn(
        total_timesteps=STAGE2_TIMESTEPS,
        callback=checkpoint_callback,
        tb_log_name=f"run_{run_idx}_gw_curriculum_stage2",
        reset_num_timesteps=False,
    )

    model.save(os.path.join(run_save_dir, "ppo_marl_model"))
    env.save(os.path.join(run_save_dir, "vec_normalize_marl.pkl"))
    model.save(os.path.join(run_save_dir, "ppo_marl_model_gw_curriculum"))
    env.save(os.path.join(run_save_dir, "vec_normalize_marl_gw_curriculum.pkl"))

    env.close()
    print(f"Curriculum training finished. Outputs saved to: {run_save_dir}")
