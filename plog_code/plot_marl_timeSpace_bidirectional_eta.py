import os
import sys

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import traci
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize
from sumo_rl import parallel_env
import supersuit as ss


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CURRENT_DIR)
MARL_DIR = os.path.join(ROOT_DIR, "02_MARL")
if MARL_DIR not in sys.path:
    sys.path.insert(0, MARL_DIR)

from train_marl_po_com_GW2_bidirectional_eta import (
    BidirectionalETAObservationFunction,
    MAIN_GREEN_PHASE,
    SB3CompatibilityWrapper,
    custom_bidirectional_green_wave_reward,
)


RUN_IDX_OVERRIDE = 32


def find_latest_bidirectional_run():
    saved_models_dir = os.path.join(ROOT_DIR, "saved_models")
    candidates = []
    if not os.path.exists(saved_models_dir):
        return None

    for folder in os.listdir(saved_models_dir):
        if not folder.startswith("marl_run_"):
            continue
        try:
            run_idx = int(folder.replace("marl_run_", ""))
        except ValueError:
            continue
        model_file = os.path.join(
            saved_models_dir,
            folder,
            "ppo_marl_model_bidirectional_eta.zip",
        )
        if os.path.exists(model_file):
            candidates.append(run_idx)

    return max(candidates) if candidates else None


RUN_IDX = RUN_IDX_OVERRIDE if RUN_IDX_OVERRIDE is not None else find_latest_bidirectional_run()
if RUN_IDX is None:
    raise FileNotFoundError("No bidirectional ETA model run was found in saved_models.")

RUN_DIR = os.path.join(ROOT_DIR, "saved_models", f"marl_run_{RUN_IDX}")
MODEL_PATH = os.path.join(RUN_DIR, "ppo_marl_model_bidirectional_eta.zip")
VEC_NORM_PATH = os.path.join(RUN_DIR, "vec_normalize_marl_bidirectional_eta.pkl")

net_path = os.path.join(ROOT_DIR, "SUMOroutes.net.xml")
route_path = os.path.join(ROOT_DIR, "traffic.random.rou.xml")
output_dir = os.path.join(ROOT_DIR, "logs", f"marl_run_{RUN_IDX}")


def run_test_and_harvest_data():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    if not os.path.exists(VEC_NORM_PATH):
        raise FileNotFoundError(f"VecNormalize file not found: {VEC_NORM_PATH}")

    env = parallel_env(
        net_file=net_path,
        route_file=route_path,
        out_csv_name=None,
        use_gui=False,
        num_seconds=3600,
        reward_fn=custom_bidirectional_green_wave_reward,
        observation_class=BidirectionalETAObservationFunction,
        min_green=10,
        max_green=55,
    )

    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(
        env,
        num_vec_envs=1,
        num_cpus=1,
        base_class="stable_baselines3",
    )
    env = SB3CompatibilityWrapper(env)
    env = VecNormalize.load(VEC_NORM_PATH, env)
    env.training = False
    env.norm_reward = False

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
                    "main_green": 1 if current_phase == MAIN_GREEN_PHASE else 0,
                    "phase": current_phase,
                }
            )

        for veh_id in traci.vehicle.getIDList():
            if "WE_MAIN_straight" in veh_id or "EW_MAIN_straight" in veh_id:
                x_pos = traci.vehicle.getPosition(veh_id)[0]
                direction = "A-B-C" if "WE_MAIN_straight" in veh_id else "C-B-A"
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


def _draw_signal_bands(ax, signal_states, time_start, time_end):
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

        ax.axhline(y=y_pos, color="gray", linestyle="--", linewidth=1, alpha=0.5)
        ax.text(time_start + 5, y_pos + 10, f"Intersection {ts_id}", color="black")


def plot_time_space(df_traj, signal_states, time_start, time_end, file_suffix, title_desc):
    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, ax = plt.subplots(figsize=(16, 8))
    ax.set_xlim(time_start, time_end)
    ax.set_ylim(0, 650)

    _draw_signal_bands(ax, signal_states, time_start, time_end)

    df_window = df_traj[(df_traj["time"] >= time_start) & (df_traj["time"] <= time_end)]
    colors = {"A-B-C": "black", "C-B-A": "#1f77b4"}

    for veh_id, group in df_window.groupby("veh_id"):
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

    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"marl_run_{RUN_IDX}_bidirectional_tsd_{file_suffix}.png")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved: {save_path}")


if __name__ == "__main__":
    traj_data, sig_data = run_test_and_harvest_data()

    plot_time_space(
        df_traj=traj_data,
        signal_states=sig_data,
        time_start=400,
        time_end=800,
        file_suffix="offpeak_400_800",
        title_desc="Off-Peak 400s-800s",
    )

    plot_time_space(
        df_traj=traj_data,
        signal_states=sig_data,
        time_start=1600,
        time_end=2000,
        file_suffix="peak_1600_2000",
        title_desc="Peak 1600s-2000s",
    )
