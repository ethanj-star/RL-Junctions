from __future__ import annotations

import importlib.util
import os
import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import VecEnvWrapper, VecMonitor, VecNormalize

# 路径统一指向共享 envs，模型和日志保存在当前绿波模块中。
# Paths use the shared envs folder, while models and logs stay inside this module.
BASE_DIR = Path(__file__).resolve().parent
FORMAL_ROOT = BASE_DIR.parent.parent
ENV_DIR = FORMAL_ROOT / "envs"
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"
NET_FILE = ENV_DIR / "SUMOroutes.net.xml"
ROUTE_FILE = ENV_DIR / "traffic.random.rou.xml"
RUN_PREFIX = "fixlr_m06_morl_tunable_run_"

# 训练参数可由顺序控制器传入；未传入环境变量时保留手动运行默认值。
# The sequential controller can override these settings; manual defaults remain available.
RUN_IDX = int(os.environ["JUC_RUN_IDX"]) if os.environ.get("JUC_RUN_IDX") else None
SEED = int(os.environ.get("JUC_SEED", "8848"))
TOTAL_TIMESTEPS = int(os.environ.get("JUC_TOTAL_TIMESTEPS", "1000000"))
NUM_SECONDS = 3600
MIN_GREEN = 5
MAX_GREEN = 55

# 训练使用固定学习率 3e-5。 / Training uses a fixed learning rate of 3e-5.
# 随机车流脚本从共享 envs 目录加载，生成结果也写回该目录。
# The random traffic generator is loaded from shared envs and writes back there.
traffic_spec = importlib.util.spec_from_file_location(
    "formal_random_traffic",
    ENV_DIR / "generate_Random_Traffic.py",
)
assert traffic_spec is not None and traffic_spec.loader is not None
traffic_module = importlib.util.module_from_spec(traffic_spec)
traffic_spec.loader.exec_module(traffic_module)
generate_route_file = traffic_module.generate_route_file

# 信号状态和 ETA 奖励参数与原始双向绿波实验保持一致。
# Signal states and ETA reward settings match the prepared bidirectional experiment.
MAIN_GREEN_STATE = "rrrrGGggrrrrGGgg"
SIDE_GREEN_STATE = "GGggrrrrGGggrrrr"
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

# 每个路口同时使用两个主路进口、两个下游方向和左右邻居信息。
# Each signal uses two arterial approaches, two downstream links, and adjacent signals.
MAIN_APPROACH_EDGES = {
    "A0": ["left0A0", "B0A0"],
    "B0": ["A0B0", "C0B0"],
    "C0": ["B0C0", "right0C0"],
}
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

# 五个目标的顺序同时决定奖励标量化顺序和神经网络中的偏好输入顺序。
# The objective order controls both reward scalarization and preference-vector input.
MORL_OBJECTIVE_NAMES = (
    "delay",
    "green_wave",
    "free_flow",
    "side_fairness",
    "spillback",
)
PREFERENCE_RANGES = {
    "delay": (0.38, 0.55),
    "green_wave": (0.15, 0.32),
    "free_flow": (0.06, 0.14),
    "side_fairness": (0.15, 0.30),
    "spillback": (0.03, 0.12),
}
PREFERENCE_DIM = len(MORL_OBJECTIVE_NAMES)


# 偏好权重保持非负并归一化到总和为 1，便于不同偏好之间比较。
# Preference weights remain non-negative and are normalized to sum to one.
def normalize_morl_weights(values: dict) -> dict:
    weights = {name: float(values[name]) for name in MORL_OBJECTIVE_NAMES}
    if any(value < 0.0 for value in weights.values()):
        raise ValueError("MORL weights must be non-negative.")
    total = sum(weights.values())
    if total <= 0.0:
        raise ValueError("At least one MORL weight must be positive.")
    return {name: weights[name] / total for name in MORL_OBJECTIVE_NAMES}


def preference_midpoint() -> dict:
    return normalize_morl_weights(
        {name: sum(PREFERENCE_RANGES[name]) / 2.0 for name in MORL_OBJECTIVE_NAMES}
    )


# 每个 episode 采样一组偏好，奖励和观测始终读取同一组权重。
# One preference is sampled per episode and shared by reward and observation.
class PreferenceManager:
    def __init__(self):
        self.current_weights = preference_midpoint()

    def sample_episode(self) -> dict:
        raw = {
            name: random.uniform(*PREFERENCE_RANGES[name])
            for name in MORL_OBJECTIVE_NAMES
        }
        self.current_weights = normalize_morl_weights(raw)
        return self.current_weights

    def as_array(self) -> np.ndarray:
        return np.asarray(
            [self.current_weights[name] for name in MORL_OBJECTIVE_NAMES],
            dtype=np.float32,
        )


PREFERENCE_MANAGER = PreferenceManager()


# 该包装器统一 PettingZoo/Gymnasium 与 SB3 的 reset 和 step 返回格式。
# This wrapper aligns PettingZoo/Gymnasium reset and step results with SB3.
class SB3CompatibilityWrapper(VecEnvWrapper):
    def __init__(self, venv, reset_callback=None):
        super().__init__(venv)
        self.reset_callback = reset_callback

    def reset(self):
        if self.reset_callback is not None:
            self.reset_callback()
        obs = self.venv.reset()
        return obs[0] if isinstance(obs, tuple) and len(obs) == 2 else obs

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
            return obs, rewards, np.logical_or(terminated, truncated), infos
        return results


# 自动查找下一个 run 编号，同时保留手动指定 RUN_IDX 的能力。
# This finds the next run number while still allowing an explicit RUN_IDX.
def next_run_number() -> int:
    numbers = []
    for path in MODELS_DIR.glob(f"{RUN_PREFIX}*"):
        suffix = path.name.removeprefix(RUN_PREFIX)
        if suffix.isdigit():
            numbers.append(int(suffix))
    return max(numbers, default=0) + 1


# 以下工具从 TraCI 读取灯色、ETA、排队和下游占有率，供观测与奖励共同使用。
# These helpers read signal, ETA, queue, and downstream occupancy data for observations and rewards.
def _current_step(traffic_signal) -> float:
    return float(getattr(traffic_signal.env, "sim_step", 0.0))


def _is_new_episode(traffic_signal) -> bool:
    return _current_step(traffic_signal) <= float(getattr(traffic_signal.env, "delta_time", 5.0))


def _signal_state(sumo, signal_id: str) -> str:
    return sumo.trafficlight.getRedYellowGreenState(signal_id)


def _is_main_green_state(state: str) -> bool:
    return state == MAIN_GREEN_STATE


def _is_yellow_state(state: str) -> bool:
    return "y" in state.lower()


def _update_green_tracking(traffic_signal):
    if _is_new_episode(traffic_signal):
        for name in ("my_green_start", "last_potential"):
            if hasattr(traffic_signal, name):
                delattr(traffic_signal, name)

    state = _signal_state(traffic_signal.sumo, traffic_signal.id)
    is_main_green = _is_main_green_state(state)
    if is_main_green and not hasattr(traffic_signal, "my_green_start"):
        traffic_signal.my_green_start = _current_step(traffic_signal)
    elif not is_main_green and hasattr(traffic_signal, "my_green_start"):
        delattr(traffic_signal, "my_green_start")
    return state, is_main_green


def _green_duration_norm(traffic_signal) -> float:
    if not hasattr(traffic_signal, "my_green_start"):
        return 0.0
    elapsed = _current_step(traffic_signal) - traffic_signal.my_green_start
    return min(elapsed / 60.0, 1.0)


def _edge_metrics(sumo, edge_id: str) -> dict:
    lane_id = f"{edge_id}_0"
    lane_length = float(sumo.lane.getLength(lane_id))
    vehicle_ids = list(sumo.lane.getLastStepVehicleIDs(lane_id))
    eta_values = []
    pressure = near_count = free_flow_count = 0.0

    for vehicle_id in vehicle_ids:
        distance = max(lane_length - float(sumo.vehicle.getLanePosition(vehicle_id)), 0.0)
        speed = float(sumo.vehicle.getSpeed(vehicle_id))
        eta = distance / max(speed, MIN_ETA_SPEED)
        if eta <= ETA_HORIZON:
            eta_values.append(eta)
            pressure += np.exp(-eta / ETA_DECAY)
            near_count += float(eta <= ETA_NEAR_WINDOW)
        free_flow_count += float(distance <= STOPLINE_DISTANCE and speed >= FREE_FLOW_SPEED)

    return {
        "halting": float(sumo.lane.getLastStepHaltingNumber(lane_id)),
        "occupancy": min(float(sumo.lane.getLastStepOccupancy(lane_id)) / 100.0, 1.0),
        "pressure": float(pressure),
        "near_count": near_count,
        "free_flow_count": free_flow_count,
        "min_eta": float(min(eta_values, default=ETA_HORIZON)),
    }


def _main_arrival_metrics(traffic_signal) -> dict:
    result = {
        "pressure": 0.0,
        "near_count": 0.0,
        "halting": 0.0,
        "free_flow_count": 0.0,
        "min_eta": ETA_HORIZON,
    }
    for edge_id in MAIN_APPROACH_EDGES[traffic_signal.id]:
        metrics = _edge_metrics(traffic_signal.sumo, edge_id)
        for name in ("pressure", "near_count", "halting", "free_flow_count"):
            result[name] += metrics[name]
        result["min_eta"] = min(result["min_eta"], metrics["min_eta"])
    return result


def _side_queue(traffic_signal) -> float:
    return sum(
        float(traffic_signal.sumo.lane.getLastStepHaltingNumber(lane_id))
        for lane_id in traffic_signal.lanes
        if "top" in lane_id or "bottom" in lane_id
    )


def _main_queue(traffic_signal) -> float:
    return sum(
        float(traffic_signal.sumo.lane.getLastStepHaltingNumber(lane_id))
        for lane_id in traffic_signal.lanes
        if "top" not in lane_id and "bottom" not in lane_id
    )


def _downstream_blockage(traffic_signal) -> float:
    blockage = 0.0
    for edge_id in MAIN_DOWNSTREAM_EDGES[traffic_signal.id]:
        metrics = _edge_metrics(traffic_signal.sumo, edge_id)
        occupancy = max(0.0, metrics["occupancy"] - SPILLBACK_OCC_THRESHOLD)
        queue = min(metrics["halting"] / 12.0, 1.0)
        blockage += max(occupancy, queue)
    return min(blockage, 1.0)


# 五维奖励分别表示延误、绿波、自由流、支路公平和下游回溢。
# The five reward components represent delay, green wave, free flow, side fairness, and spillback.
def morl_reward_components(traffic_signal) -> dict:
    state, is_main_green = _update_green_tracking(traffic_signal)
    total_queue = float(traffic_signal.get_total_queued())
    phi_current = -total_queue / POTENTIAL_QUEUE_NORM

    if getattr(traffic_signal, "last_potential", None) is None or _is_new_episode(traffic_signal):
        shaping_reward = 0.0
    else:
        shaping_reward = 0.99 * phi_current - traffic_signal.last_potential
    traffic_signal.last_potential = phi_current

    arrival = _main_arrival_metrics(traffic_signal)
    arrival_pressure = min(arrival["pressure"], 6.0)
    side_queue = _side_queue(traffic_signal)
    downstream_block = _downstream_blockage(traffic_signal)

    green_wave_component = 0.0
    if arrival_pressure > 0.0:
        green_wave_component = (
            ETA_GREEN_WEIGHT * arrival_pressure
            if is_main_green
            else -ETA_RED_WEIGHT * arrival_pressure * float(not _is_yellow_state(state))
        )

    free_flow_component = (
        FREE_FLOW_WEIGHT * min(arrival["free_flow_count"], 4.0) if is_main_green else 0.0
    )
    side_fairness_component = 0.0
    if is_main_green:
        green_elapsed = 60.0 * _green_duration_norm(traffic_signal)
        if arrival_pressure < 0.2 and side_queue > 0.0:
            side_fairness_component -= IDLE_MAIN_GREEN_PENALTY
        if side_queue > SIDE_QUEUE_THRESHOLD and green_elapsed > LONG_MAIN_GREEN_START:
            overload = side_queue - SIDE_QUEUE_THRESHOLD
            duration = min((green_elapsed - LONG_MAIN_GREEN_START) / 25.0, 1.0)
            side_fairness_component -= SIDE_QUEUE_WEIGHT * overload * duration

    spillback_component = 0.0
    if is_main_green and downstream_block > 0.0:
        spillback_component = -SPILLBACK_WEIGHT * downstream_block * max(arrival_pressure, 1.0)

    return {
        "delay": -QUEUE_WEIGHT * total_queue + shaping_reward,
        "green_wave": green_wave_component,
        "free_flow": free_flow_component,
        "side_fairness": side_fairness_component,
        "spillback": spillback_component,
    }


# 当前 episode 的偏好向量将五维奖励线性标量化为 PPO 使用的单一奖励。
# The current episode preference linearly scalarizes the five objectives for PPO.
def tunable_morl_reward(traffic_signal) -> float:
    components = morl_reward_components(traffic_signal)
    return float(
        sum(
            PREFERENCE_MANAGER.current_weights[name] * components[name]
            for name in MORL_OBJECTIVE_NAMES
        )
    )


# 观测在默认维度后追加 18 维交通特征和 5 维当前偏好向量。
# The observation appends 18 traffic features and the current five-value preference vector.
def get_observation_class():
    from gymnasium import spaces
    from sumo_rl.environment.observations import DefaultObservationFunction

    class MORLBidirectionalETAObservationFunction(DefaultObservationFunction):
        def observation_space(self):
            base_dim = super().observation_space().shape[0]
            return spaces.Box(
                low=np.zeros(base_dim + 18 + PREFERENCE_DIM, dtype=np.float32),
                high=np.full(base_dim + 18 + PREFERENCE_DIM, np.inf, dtype=np.float32),
            )

        def _neighbor_features(self):
            features = []
            for neighbor_id in NEIGHBOR_MAP[self.ts.id]:
                if neighbor_id is None:
                    features.extend([0.0, 0.0, 0.0])
                    continue
                neighbor = self.ts.env.traffic_signals[neighbor_id]
                state = _signal_state(neighbor.sumo, neighbor_id)
                features.extend(
                    [
                        min(_main_queue(neighbor) / 50.0, 1.0),
                        float(_is_main_green_state(state)),
                        _green_duration_norm(neighbor),
                    ]
                )
            return features

        def _eta_features(self):
            features = []
            for edge_id in MAIN_APPROACH_EDGES[self.ts.id]:
                metrics = _edge_metrics(self.ts.sumo, edge_id)
                features.extend(
                    [
                        min(metrics["pressure"] / 4.0, 1.0),
                        min(metrics["near_count"] / 6.0, 1.0),
                        min(metrics["halting"] / 12.0, 1.0),
                        min(metrics["min_eta"] / ETA_HORIZON, 1.0),
                    ]
                )
            return features

        def __call__(self):
            state = _signal_state(self.ts.sumo, self.ts.id)
            local = [
                float(_is_main_green_state(state)),
                _green_duration_norm(self.ts),
                min(_side_queue(self.ts) / 30.0, 1.0),
                _downstream_blockage(self.ts),
            ]
            extra = self._neighbor_features() + self._eta_features() + local
            return np.concatenate(
                [
                    super().__call__(),
                    np.asarray(extra, dtype=np.float32),
                    PREFERENCE_MANAGER.as_array(),
                ]
            ).astype(np.float32)

    return MORLBidirectionalETAObservationFunction


# 每个 episode 采样一组偏好并重生成随机车流，再构建三智能体环境。
# Each episode samples one preference and regenerates traffic before building the three-agent environment.
def make_pettingzoo_env(csv_base_path: Path, use_gui: bool = False):
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
        reward_fn=tunable_morl_reward,
        observation_class=get_observation_class(),
        min_green=MIN_GREEN,
        max_green=MAX_GREEN,
        sumo_seed=SEED,
    )
    # 每次 reset 更新偏好；首次使用预生成车流，之后按递增 seed 更新车流。
    # Every reset samples a preference; traffic is prepared once and then regenerated with increasing seeds.
    def prepare_episode():
        nonlocal episode_index, first_reset
        preference = PREFERENCE_MANAGER.sample_episode()
        print(f"Episode preference: {preference}")
        if first_reset:
            first_reset = False
            return
        generate_route_file(seed=SEED + episode_index, output_file=route_file)
        episode_index += 1

    return env, prepare_episode


def make_vec_env(csv_base_path: Path, use_gui: bool = False):
    import supersuit as ss

    env, reset_callback = make_pettingzoo_env(csv_base_path, use_gui)
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(
        env,
        num_vec_envs=1,
        num_cpus=1,
        base_class="stable_baselines3",
    )
    return VecMonitor(SB3CompatibilityWrapper(env, reset_callback))


# 每个 run 保存偏好范围和关键训练参数，便于复现实验。
# Each run stores its preference ranges and key training settings for reproducibility.
def write_morl_config(model_dir: Path) -> None:
    lines = [
        "Tunable MORL green-wave training configuration",
        "objective_order=" + ",".join(MORL_OBJECTIVE_NAMES),
        *(f"preference_range_{name}={low:.6f}:{high:.6f}" for name, (low, high) in PREFERENCE_RANGES.items()),
        f"total_timesteps={TOTAL_TIMESTEPS}",
        f"min_green={MIN_GREEN}",
        f"max_green={MAX_GREEN}",
        f"preference_dim={PREFERENCE_DIM}",
        "observation_layout=default_plus_18_traffic_features_plus_5_preferences",
        "feature_extractor=default_sb3_mlp_policy",
    ]
    (model_dir / "morl_tunable_preference_config.txt").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


# 主训练流程保存模型、VecNormalize、checkpoint、TensorBoard 和 SUMO CSV。
# The main training flow saves the model, VecNormalize, checkpoints, TensorBoard data, and SUMO CSV files.
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
    write_morl_config(model_dir)

    env = VecNormalize(
        make_vec_env(log_dir / "marl_output"),
        norm_obs=True,
        norm_reward=False,
        clip_obs=10.0,
    )
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-5,
        n_steps=2048,
        batch_size=256,
        n_epochs=10,
        clip_range=0.2,
        ent_coef=0.005,
        target_kl=0.05,
        verbose=1,
        device="cpu",
        seed=SEED,
        tensorboard_log=str(LOGS_DIR / "ppo_marl_tb"),
    )
    checkpoint = CheckpointCallback(
        save_freq=50000,
        save_path=str(model_dir / "checkpoints"),
        name_prefix="rl_model_morl_tunable",
        save_vecnormalize=True,
    )
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=checkpoint,
        tb_log_name=f"run_{run_idx}_bidirectional_eta_morl_tunable",
    )
    model.save(str(model_dir / "ppo_marl_model_bidirectional_eta_morl_tunable"))
    env.save(str(model_dir / "vec_normalize_marl_bidirectional_eta_morl_tunable.pkl"))
    env.close()


if __name__ == "__main__":
    train()
