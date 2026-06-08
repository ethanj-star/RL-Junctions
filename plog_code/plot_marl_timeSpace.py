import os
import re

import matplotlib.patches as patches
import matplotlib.pyplot as plt
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

# If this script is copied into another project root, leave this as None.
# If you run this copy while the trained model is in another folder, set:
#   PowerShell: $env:JUC_RL_ROOT="C:\Users\DJI\Desktop\dissertation\3JucRL"
ROOT_DIR = os.environ.get("JUC_RL_ROOT") or os.path.dirname(CURRENT_DIR)

RUN_IDX_OVERRIDE = 31
MODEL_BASENAME_OVERRIDE = None
# Examples:
# MODEL_BASENAME_OVERRIDE = "ppo_marl_model_gw_stage1"
# MODEL_BASENAME_OVERRIDE = "ppo_marl_model_gw_curriculum"

MAIN_GREEN_PHASE = 2
TIME_WINDOWS = [
    (400, 800, "offpeak_400_800", "Off-Peak 400s-800s"),
    (1600, 2000, "peak_1600_2000", "Peak 1600s-2000s"),
]

net_path = os.path.join(ROOT_DIR, "SUMOroutes.net.xml")
route_path = os.path.join(ROOT_DIR, "traffic.random.rou.xml")

NEIGHBOR_MAP = {
    "A0": [None, "B0"],
    "B0": ["A0", "C0"],
    "C0": ["B0", None],
}


def find_latest_run():
    saved_models_dir = os.path.join(ROOT_DIR, "saved_models")
    candidates = []

    if not os.path.exists(saved_models_dir):
        return None

    for folder in os.listdir(saved_models_dir):
        match = re.fullmatch(r"marl_run_(\d+)", folder)
        if not match:
            continue

        run_idx = int(match.group(1))
        run_dir = os.path.join(saved_models_dir, folder)
        has_model = (
            os.path.exists(os.path.join(run_dir, "ppo_marl_model.zip"))
            or os.path.exists(os.path.join(run_dir, "ppo_marl_model_gw_lite.zip"))
        )
        has_vecnorm = (
            os.path.exists(os.path.join(run_dir, "vec_normalize_marl.pkl"))
            or os.path.exists(os.path.join(run_dir, "vec_normalize_marl_gw_lite.pkl"))
        )

        if has_model and has_vecnorm:
            candidates.append(run_idx)

    return max(candidates) if candidates else None


RUN_IDX = RUN_IDX_OVERRIDE if RUN_IDX_OVERRIDE is not None else find_latest_run()
if RUN_IDX is None:
    raise FileNotFoundError(f"No valid marl_run_* model was found under {ROOT_DIR}")

RUN_DIR = os.path.join(ROOT_DIR, "saved_models", f"marl_run_{RUN_IDX}")
LOG_DIR = os.path.join(ROOT_DIR, "logs", f"marl_run_{RUN_IDX}")

if MODEL_BASENAME_OVERRIDE is None:
    MODEL_PATH = os.path.join(RUN_DIR, "ppo_marl_model.zip")
    VEC_NORM_PATH = os.path.join(RUN_DIR, "vec_normalize_marl.pkl")

    if not os.path.exists(MODEL_PATH):
        MODEL_PATH = os.path.join(RUN_DIR, "ppo_marl_model_gw_lite.zip")
    if not os.path.exists(VEC_NORM_PATH):
        VEC_NORM_PATH = os.path.join(RUN_DIR, "vec_normalize_marl_gw_lite.pkl")
else:
    MODEL_PATH = os.path.join(RUN_DIR, f"{MODEL_BASENAME_OVERRIDE}.zip")
    vec_name = MODEL_BASENAME_OVERRIDE.replace("ppo_marl_model", "vec_normalize_marl")
    VEC_NORM_PATH = os.path.join(RUN_DIR, f"{vec_name}.pkl")


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

            main_arterial_queue = 0
            for lane in neighbor_ts.lanes:
                if "top" not in lane and "bottom" not in lane:
                    main_arterial_queue += neighbor_ts.sumo.lane.getLastStepHaltingNumber(lane)

            extra_obs.append(float(main_arterial_queue))

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


def build_env():
    env = parallel_env(
        net_file=net_path,
        route_file=route_path,
        out_csv_name=None,
        use_gui=False,
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

    if not os.path.exists(VEC_NORM_PATH):
        raise FileNotFoundError(f"VecNormalize file not found: {VEC_NORM_PATH}")

    env = VecNormalize.load(VEC_NORM_PATH, env)
    env.training = False
    env.norm_reward = False
    return env


def run_test_and_harvest_data():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

    print(f"Project root: {ROOT_DIR}")
    print(f"Run index: {RUN_IDX}")
    print(f"Model: {MODEL_PATH}")
    print(f"VecNormalize: {VEC_NORM_PATH}")
    print(f"Main-road green phase: {MAIN_GREEN_PHASE}")

    env = build_env()
    model = PPO.load(MODEL_PATH, env=env)
    obs = env.reset()

    trajectory_data = []
    signal_states = {"A0": [], "B0": [], "C0": []}

    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, dones, _ = env.step(action)

        current_sim_time = traci.simulation.getTime()

        for ts_id in ["A0", "B0", "C0"]:
            current_phase = traci.trafficlight.getPhase(ts_id)
            signal_states[ts_id].append(
                {
                    "time": current_sim_time,
                    "phase": current_phase,
                    "main_green": 1 if current_phase == MAIN_GREEN_PHASE else 0,
                }
            )

        for veh_id in traci.vehicle.getIDList():
            if "WE_MAIN_straight" in veh_id:
                direction = "A-B-C"
            elif "EW_MAIN_straight" in veh_id:
                direction = "C-B-A"
            else:
                continue

            x_pos = traci.vehicle.getPosition(veh_id)[0]
            trajectory_data.append(
                {
                    "time": current_sim_time,
                    "veh_id": veh_id,
                    "position": x_pos,
                    "direction": direction,
                }
            )

        if np.any(dones):
            break

    env.close()
    return pd.DataFrame(trajectory_data), signal_states


def draw_signal_bands(ax, signal_states, time_start, time_end):
    junctions = {"A0": 100, "B0": 300, "C0": 500}
    band_width = 15

    for ts_id, y_pos in junctions.items():
        df_sig = pd.DataFrame(signal_states[ts_id])
        df_sig = df_sig[(df_sig["time"] >= time_start) & (df_sig["time"] <= time_end)]

        if df_sig.empty:
            continue

        current_color = None
        start_t = time_start

        for _, row in df_sig.iterrows():
            t = row["time"]
            color = "#2ecc71" if row["main_green"] == 1 else "#e74c3c"

            if current_color is None:
                current_color = color
                start_t = t
            elif color != current_color:
                ax.add_patch(
                    patches.Rectangle(
                        (start_t, y_pos - band_width / 2),
                        t - start_t,
                        band_width,
                        linewidth=0,
                        facecolor=current_color,
                        alpha=0.55,
                    )
                )
                current_color = color
                start_t = t

        ax.add_patch(
            patches.Rectangle(
                (start_t, y_pos - band_width / 2),
                time_end - start_t,
                band_width,
                linewidth=0,
                facecolor=current_color,
                alpha=0.55,
            )
        )

        ax.axhline(y=y_pos, color="gray", linestyle="--", linewidth=1, alpha=0.45)
        ax.text(time_start + 5, y_pos + 10, f"Intersection {ts_id}", color="black")


def plot_green_wave(df_traj, signal_states, time_start, time_end, file_suffix, title_desc):
    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, ax = plt.subplots(figsize=(16, 8))
    ax.set_xlim(time_start, time_end)
    ax.set_ylim(0, 650)

    draw_signal_bands(ax, signal_states, time_start, time_end)

    df_window = df_traj[(df_traj["time"] >= time_start) & (df_traj["time"] <= time_end)]
    colors = {"A-B-C": "black", "C-B-A": "#1f77b4"}

    for _, group in df_window.groupby("veh_id"):
        if len(group) <= 5:
            continue
        direction = group["direction"].iloc[0]
        ax.plot(
            group["time"],
            group["position"],
            color=colors.get(direction, "black"),
            alpha=0.42,
            linewidth=1.15,
        )

    ax.set_title(
        f"Bidirectional Time-Space Diagram ({title_desc}) - Run {RUN_IDX}",
        fontsize=18,
        fontweight="bold",
        pad=20,
    )
    ax.set_xlabel("Simulation Time (seconds)", fontsize=14)
    ax.set_ylabel("Absolute X Position / Corridor (meters)", fontsize=14)
    ax.grid(True, linestyle=":", alpha=0.6)

    legend_handles = [
        patches.Patch(color="black", label="A-B-C vehicles"),
        patches.Patch(color="#1f77b4", label="C-B-A vehicles"),
        patches.Patch(color="#2ecc71", label="Main-road green phase"),
        patches.Patch(color="#e74c3c", label="Other phase"),
    ]
    ax.legend(handles=legend_handles, loc="upper right")

    os.makedirs(LOG_DIR, exist_ok=True)
    model_suffix = MODEL_BASENAME_OVERRIDE or "final"
    save_path = os.path.join(LOG_DIR, f"marl_run_{RUN_IDX}_{model_suffix}_green_wave_tsd_{file_suffix}.png")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"Saved: {save_path}")


if __name__ == "__main__":
    traj_data, sig_data = run_test_and_harvest_data()

    for start, end, suffix, desc in TIME_WINDOWS:
        plot_green_wave(
            df_traj=traj_data,
            signal_states=sig_data,
            time_start=start,
            time_end=end,
            file_suffix=suffix,
            title_desc=desc,
        )
