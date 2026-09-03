from pathlib import Path
from typing import Optional
import os
import re

import matplotlib.pyplot as plt
import pandas as pd

# RUN_IDX 用于选择要绘制的训练日志，WINDOW_SIZE 控制曲线平滑窗口。
# RUN_IDX selects the training logs to plot, and WINDOW_SIZE controls the smoothing window.
RUN_IDX = int(os.environ.get("JUC_RUN_IDX", "1"))
RUN_PREFIX = "multi_queue_dynamic_lr_run_"
WINDOW_SIZE = 5

BASE_DIR = Path(__file__).resolve().parent
LOG_DIR = BASE_DIR / "logs" / f"{RUN_PREFIX}{RUN_IDX}"


# 读取当前 run 的 SUMO CSV 文件，并按 episode 汇总等待时间和排队长度。
# This reads SUMO CSV files for the selected run and aggregates waiting time and queue length by episode.
def load_episode_stats() -> Optional[pd.DataFrame]:
    frames = []
    for csv_path in sorted(LOG_DIR.glob("*.csv")):
        if "test" in csv_path.name:
            continue
        match = re.search(r"ep(\d+)\.csv$", csv_path.name)
        if not match:
            continue
        df = pd.read_csv(csv_path)
        df["episode"] = int(match.group(1))
        frames.append(df)
    if not frames:
        print(f"No SUMO csv files found in {LOG_DIR}")
        return None

    data = pd.concat(frames, ignore_index=True)
    stats = data.groupby("episode").agg(
        waiting_time=("system_total_waiting_time", "mean"),
        queue_length=("system_total_stopped", "mean"),
    ).reset_index()
    stats = stats.sort_values("episode")
    stats["waiting_smooth"] = stats["waiting_time"].rolling(WINDOW_SIZE, min_periods=1).mean()
    stats["queue_smooth"] = stats["queue_length"].rolling(WINDOW_SIZE, min_periods=1).mean()
    stats["waiting_std"] = stats["waiting_time"].rolling(WINDOW_SIZE, min_periods=1).std().fillna(0)
    stats["queue_std"] = stats["queue_length"].rolling(WINDOW_SIZE, min_periods=1).std().fillna(0)
    return stats


# 在平滑曲线的最低点标记最佳 episode，并根据横轴位置自动选择标签方向。
# Mark the lowest smoothed value and place its label toward the plot interior.
def annotate_best(axis, episodes, best_row, value_column, color, unit) -> None:
    episode = int(best_row["episode"])
    value = float(best_row[value_column])
    place_right = episode <= (episodes.min() + episodes.max()) / 2
    offset_x = 14 if place_right else -14
    axis.scatter(episode, value, s=58, color=color, edgecolor="white", linewidth=0.8, zorder=6)
    axis.annotate(
        f"Best: {value:.1f} {unit}\n@ Ep {episode}",
        xy=(episode, value),
        xytext=(offset_x, 22),
        textcoords="offset points",
        ha="left" if place_right else "right",
        va="bottom",
        color=color,
        fontsize=9,
        fontweight="bold",
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": color, "alpha": 0.95},
        arrowprops={"arrowstyle": "->", "color": color, "linewidth": 1.2},
        zorder=7,
    )


# 绘制等待时间和排队长度两张训练曲线，并保存到当前 run 的日志文件夹。
# This plots waiting time and queue length curves, then saves the figure into the log folder of the selected run.
def plot_learning_curve() -> None:
    stats = load_episode_stats()
    if stats is None:
        return

    episodes = stats["episode"].to_numpy()
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    axes[0].scatter(episodes, stats["waiting_time"], color="0.65", s=18, alpha=0.45, label="raw")
    axes[0].plot(episodes, stats["waiting_smooth"], color="#d62728", linewidth=2, label="smoothed")
    axes[0].fill_between(
        episodes,
        (stats["waiting_smooth"] - stats["waiting_std"]).to_numpy(),
        (stats["waiting_smooth"] + stats["waiting_std"]).to_numpy(),
        color="#d62728",
        alpha=0.15,
    )
    waiting_best = stats.loc[stats["waiting_smooth"].idxmin()]
    annotate_best(axes[0], episodes, waiting_best, "waiting_smooth", "#d62728", "s")
    axes[0].set_ylabel("Average waiting time (s)")
    axes[0].set_title(f"MARL queue dynamic LR waiting time, run {RUN_IDX}")
    axes[0].legend()

    axes[1].scatter(episodes, stats["queue_length"], color="0.65", s=18, alpha=0.45, label="raw")
    axes[1].plot(episodes, stats["queue_smooth"], color="#2ca02c", linewidth=2, label="smoothed")
    axes[1].fill_between(
        episodes,
        (stats["queue_smooth"] - stats["queue_std"]).to_numpy(),
        (stats["queue_smooth"] + stats["queue_std"]).to_numpy(),
        color="#2ca02c",
        alpha=0.15,
    )
    queue_best = stats.loc[stats["queue_smooth"].idxmin()]
    annotate_best(axes[1], episodes, queue_best, "queue_smooth", "#2ca02c", "veh")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Average queue length (vehicles)")
    axes[1].legend()

    fig.tight_layout()
    output_path = LOG_DIR / f"{RUN_PREFIX}{RUN_IDX}_learning_curve.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    pd.DataFrame(
        [
            {"metric": "waiting_time", "episode": int(waiting_best["episode"]), "best_value": waiting_best["waiting_smooth"], "rolling_sd": waiting_best["waiting_std"], "unit": "s"},
            {"metric": "queue_length", "episode": int(queue_best["episode"]), "best_value": queue_best["queue_smooth"], "rolling_sd": queue_best["queue_std"], "unit": "vehicles"},
        ]
    ).to_csv(LOG_DIR / f"{RUN_PREFIX}{RUN_IDX}_best_points.csv", index=False, encoding="utf-8-sig")
    print(f"Saved figure to {output_path}")


if __name__ == "__main__":
    plot_learning_curve()
