"""Evaluate M04/M05/M06 on one shared traffic seed and record time-space data."""

import csv
import importlib.util
import os
import sys
from collections import defaultdict
from pathlib import Path
from xml.etree import ElementTree

import numpy as np
import torch
import traci
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecMonitor, VecNormalize


BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent.parent
MODEL_ROOT = REPO_ROOT / "MARL"
DATA_DIR = BASE_DIR / "data"

RUN_IDX = int(os.environ.get("JUC_RUN_IDX", "1"))
EVAL_SEED = int(os.environ.get("JUC_EVAL_SEED", "8848"))
SIM_SECONDS = int(os.environ.get("JUC_SIM_SECONDS", "3600"))
USE_GUI = os.environ.get("JUC_USE_GUI", "0") == "1"
ACTIVE_CONTROLLERS = tuple(
    item.strip().upper()
    for item in os.environ.get("JUC_CONTROLLERS", "M04,M05,M06").split(",")
    if item.strip()
)

MAIN_GREEN_STATE = "rrrrGGggrrrrGGgg"
SIGNALS = ("A0", "B0", "C0")
MAIN_ROUTES = ("WE_MAIN", "EW_MAIN")
CORRIDOR_EDGES = {
    "left0A0", "A0B0", "B0C0", "C0right0",
    "right0C0", "C0B0", "B0A0", "A0left0",
}
SEGMENT_BY_EDGE = {
    "A0B0": ("WE_MAIN", "A0_B0"),
    "B0C0": ("WE_MAIN", "B0_C0"),
    "C0B0": ("EW_MAIN", "C0_B0"),
    "B0A0": ("EW_MAIN", "B0_A0"),
}

CONTROLLERS = {
    "M04": {
        "folder": "M04_PBRS_Communication_FixLR",
        "run_prefix": "fixlr_m04_com_run_",
        "model": "ppo_marl_model.zip",
        "vec": "vec_normalize_marl.pkl",
    },
    "M05": {
        "folder": "M05_ETA_GreenWave_FixLR",
        "run_prefix": "fixlr_m05_com_gw_run_",
        "model": "ppo_marl_model_bidirectional_eta.zip",
        "vec": "vec_normalize_marl_bidirectional_eta.pkl",
    },
    "M06": {
        "folder": "M06_Tunable_MORL_FixLR",
        "run_prefix": "fixlr_m06_morl_tunable_run_",
        "model": "ppo_marl_model_bidirectional_eta_morl_tunable.zip",
        "vec": "vec_normalize_marl_bidirectional_eta_morl_tunable.pkl",
    },
}


# 动态加载各正式模块，直接复用其观测、奖励和环境包装逻辑。
# Load each formal module so evaluation reuses its observation, reward, and wrappers.
def load_train_module(controller: str):
    path = MODEL_ROOT / CONTROLLERS[controller]["folder"] / "train.py"
    spec = importlib.util.spec_from_file_location(f"timespace_{controller.lower()}_train", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.SEED = EVAL_SEED
    module.NUM_SECONDS = SIM_SECONDS
    return module


# 按各模块训练配置构建一次性评估环境，三个控制器使用同一评价车流。
# Build one-shot evaluation envs from each training config with the same traffic demand.
def build_env(controller: str, module, runtime_dir: Path, vec_path: Path):
    import supersuit as ss
    from sumo_rl import parallel_env

    route_file = runtime_dir / "traffic_eval.rou.xml"
    module.generate_route_file(seed=EVAL_SEED, output_file=route_file)
    if controller == "M04":
        reward_fn = module.pbrs_reward
        observation_class = module.get_comm_observation_class()
        signal_kwargs = {}
    elif controller == "M05":
        reward_fn = module.green_wave_reward
        observation_class = module.get_observation_class()
        signal_kwargs = {"min_green": module.MIN_GREEN, "max_green": module.MAX_GREEN}
    else:
        module.PREFERENCE_MANAGER.current_weights = module.preference_midpoint()
        reward_fn = module.tunable_morl_reward
        observation_class = module.get_observation_class()
        signal_kwargs = {"min_green": module.MIN_GREEN, "max_green": module.MAX_GREEN}

    env = parallel_env(
        net_file=str(module.NET_FILE),
        route_file=str(route_file),
        out_csv_name=str(runtime_dir / "sumo_output"),
        use_gui=USE_GUI,
        num_seconds=SIM_SECONDS,
        reward_fn=reward_fn,
        observation_class=observation_class,
        sumo_seed=EVAL_SEED,
        **signal_kwargs,
    )
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(
        env, num_vec_envs=1, num_cpus=1, base_class="stable_baselines3"
    )
    env = VecMonitor(module.SB3CompatibilityWrapper(env))

    env = VecNormalize.load(str(vec_path), env)
    env.training = False
    env.norm_reward = False
    return env


# 车辆编号由共享车流生成器添加方向前缀，用它区分双向主路车流。
# Vehicle IDs carry route prefixes that identify both arterial directions.
def route_group(vehicle_id: str) -> str:
    for group in MAIN_ROUTES:
        if vehicle_id.startswith(group):
            return group
    return "SIDE"


def safe(default, function, *args):
    try:
        return function(*args)
    except (traci.TraCIException, KeyError):
        return default


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# 运行一个确定性 episode，并同步记录车辆轨迹、真实灯色与网络指标。
# Run one deterministic episode and record trajectories, true signal states, and metrics.
def evaluate(controller: str) -> dict:
    config = CONTROLLERS[controller]
    module = load_train_module(controller)
    run_dir = MODEL_ROOT / config["folder"] / "models" / f"{config['run_prefix']}{RUN_IDX}"
    model_path = run_dir / config["model"]
    vec_path = run_dir / config["vec"]
    missing = [path for path in (model_path, vec_path) if not path.exists()]
    if missing:
        raise FileNotFoundError("\n".join(str(path) for path in missing))

    runtime_dir = DATA_DIR / "runtime" / f"{controller.lower()}_run{RUN_IDX}_seed{EVAL_SEED}"
    runtime_dir.mkdir(parents=True, exist_ok=True)

    # SUMO-RL 每个内部仿真秒都会调用此方法；临时计数可得到完整通行量。
    # SUMO-RL calls this method every simulated second, allowing exact trip counts.
    from sumo_rl.environment.env import SumoEnvironment

    original_sumo_step = SumoEnvironment._sumo_step
    traffic_counts = {"departed": 0, "arrived": 0, "running": 0, "finished": False}

    def counted_sumo_step(sumo_env):
        original_sumo_step(sumo_env)
        if traffic_counts["finished"]:
            return
        traffic_counts["departed"] += int(sumo_env.sumo.simulation.getDepartedNumber())
        traffic_counts["arrived"] += int(sumo_env.sumo.simulation.getArrivedNumber())
        traffic_counts["running"] = int(sumo_env.sumo.vehicle.getIDCount())
        if float(sumo_env.sumo.simulation.getTime()) >= SIM_SECONDS:
            traffic_counts["finished"] = True

    SumoEnvironment._sumo_step = counted_sumo_step
    env = build_env(controller, module, runtime_dir, vec_path)
    model = PPO.load(str(model_path), env=env)
    obs = env.reset()

    trajectories = []
    signals = []
    waiting_by_step = []
    queue_by_step = []
    vehicle_max_waiting = defaultdict(float)
    active_segments = {}
    completed_segments = []

    try:
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, dones, _ = env.step(action)
            if np.any(dones):
                break
            sim_time = float(safe(0.0, traci.simulation.getTime))

            # 灯色取 TraCI 完整 RYG 字符串，不使用可能因环境而变化的 phase 编号。
            # Read complete TraCI RYG states instead of environment-dependent phase indices.
            for signal_id in SIGNALS:
                state = safe("", traci.trafficlight.getRedYellowGreenState, signal_id)
                status = "yellow" if "y" in state.lower() else (
                    "main_green" if state == MAIN_GREEN_STATE else "main_red"
                )
                signals.append({
                    "controller": controller,
                    "time_s": sim_time,
                    "signal_id": signal_id,
                    "state": state,
                    "status": status,
                })

            active_ids = list(safe([], traci.vehicle.getIDList))
            current_segment_keys = set()
            step_waiting = 0.0
            step_queue = 0

            for vehicle_id in active_ids:
                group = route_group(vehicle_id)
                waiting = float(safe(0.0, traci.vehicle.getWaitingTime, vehicle_id))
                speed = float(safe(0.0, traci.vehicle.getSpeed, vehicle_id))
                edge_id = safe("", traci.vehicle.getRoadID, vehicle_id)
                step_waiting += waiting
                step_queue += speed < 0.1
                vehicle_max_waiting[vehicle_id] = max(vehicle_max_waiting[vehicle_id], waiting)

                if group in MAIN_ROUTES and (
                    edge_id in CORRIDOR_EDGES or edge_id.startswith(":A0")
                    or edge_id.startswith(":B0") or edge_id.startswith(":C0")
                ):
                    x_m, _ = safe((np.nan, np.nan), traci.vehicle.getPosition, vehicle_id)
                    trajectories.append({
                        "controller": controller,
                        "time_s": sim_time,
                        "vehicle_id": vehicle_id,
                        "direction": group,
                        "edge_id": edge_id,
                        "x_m": float(x_m),
                        "speed_mps": speed,
                    })

                if edge_id in SEGMENT_BY_EDGE:
                    segment_direction, segment_name = SEGMENT_BY_EDGE[edge_id]
                    key = (vehicle_id, segment_name)
                    current_segment_keys.add(key)
                    record = active_segments.setdefault(
                        key,
                        {"direction": segment_direction, "segment": segment_name, "had_stop": False},
                    )
                    record["had_stop"] = record["had_stop"] or speed < 0.1 or waiting > 0.0

            for key in list(active_segments):
                if key not in current_segment_keys:
                    completed_segments.append(active_segments.pop(key))

            waiting_by_step.append(step_waiting)
            queue_by_step.append(step_queue)
    finally:
        completed_segments.extend(active_segments.values())
        env.close()
        SumoEnvironment._sumo_step = original_sumo_step

    prefix = f"{controller.lower()}_run{RUN_IDX}_seed{EVAL_SEED}"
    write_csv(
        DATA_DIR / f"{prefix}_trajectories.csv",
        trajectories,
        ["controller", "time_s", "vehicle_id", "direction", "edge_id", "x_m", "speed_mps"],
    )
    write_csv(
        DATA_DIR / f"{prefix}_signals.csv",
        signals,
        ["controller", "time_s", "signal_id", "state", "status"],
    )

    side_waits = [
        value for vehicle_id, value in vehicle_max_waiting.items()
        if route_group(vehicle_id) == "SIDE"
    ]
    no_stop_rate = (
        float(np.mean([not row["had_stop"] for row in completed_segments]))
        if completed_segments else np.nan
    )
    route_file = runtime_dir / "traffic_eval.rou.xml"
    scheduled_vehicles = sum(1 for _ in ElementTree.parse(route_file).iter("vehicle"))
    summary = {
        "controller": controller,
        "run_idx": RUN_IDX,
        "evaluation_seed": EVAL_SEED,
        "average_waiting_pressure": float(np.mean(waiting_by_step)),
        "average_queue_length": float(np.mean(queue_by_step)),
        "main_no_stop_rate": no_stop_rate,
        "progression_segments": len(completed_segments),
        "side_mean_max_waiting": float(np.mean(side_waits)) if side_waits else np.nan,
        "side_max_waiting": float(np.max(side_waits)) if side_waits else np.nan,
        "inserted_vehicles": traffic_counts["departed"],
        "completed_trips": traffic_counts["arrived"],
        "running_vehicles_at_end": traffic_counts["running"],
        "waiting_to_insert_at_end": scheduled_vehicles - traffic_counts["departed"],
    }
    return summary


def main() -> None:
    unknown = set(ACTIVE_CONTROLLERS) - set(CONTROLLERS)
    if unknown:
        raise ValueError(f"Unknown controllers: {sorted(unknown)}")
    assert route_group("WE_MAIN_straight_1") == "WE_MAIN"
    assert route_group("NS_A0_straight_1") == "SIDE"

    torch.set_num_threads(1)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    summaries = []
    for controller in ACTIVE_CONTROLLERS:
        print(f"Evaluating {controller}: run={RUN_IDX}, traffic seed={EVAL_SEED}")
        summary = evaluate(controller)
        summaries.append(summary)
        print(
            "  waiting={average_waiting_pressure:.1f}, queue={average_queue_length:.2f}, "
            "no-stop={main_no_stop_rate:.1%}, completed={completed_trips}".format(**summary)
        )

    write_csv(DATA_DIR / "summary.csv", summaries, list(summaries[0]))
    print(f"Saved evaluation data to {DATA_DIR}")


if __name__ == "__main__":
    main()
