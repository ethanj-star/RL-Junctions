"""Plot bidirectional time-space diagrams from test.py CSV output."""

import csv
import os
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
FIGURE_DIR = BASE_DIR / "figures"
RUN_IDX = int(os.environ.get("JUC_RUN_IDX", "1"))
EVAL_SEED = int(os.environ.get("JUC_EVAL_SEED", "8848"))
CONTROLLERS = ("M04", "M05", "M06")
CONTROLLER_TITLES = {
    "M04": "M04: PBRS + communication",
    "M05": "M05: ETA green wave",
    "M06": "M06: tunable MORL (midpoint)",
}
SIGNAL_X = {"A0": 100.0, "B0": 300.0, "C0": 500.0}
STATUS_COLORS = {"main_green": "#2ca25f", "yellow": "#e6ab02", "main_red": "#de2d26"}


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


# 把连续灯色采样合并成区间，避免在图上重复绘制大量散点。
# Merge consecutive signal samples into intervals before plotting.
def signal_intervals(rows: list[dict], signal_id: str, start: float, end: float):
    points = sorted(
        (float(row["time_s"]), row["status"])
        for row in rows
        if row["signal_id"] == signal_id and start <= float(row["time_s"]) <= end
    )
    if not points:
        return []
    intervals = []
    interval_start, status = points[0]
    previous_time = interval_start
    for time_s, next_status in points[1:]:
        if next_status != status:
            intervals.append((interval_start, previous_time + 5.0, status))
            interval_start, status = time_s, next_status
        previous_time = time_s
    intervals.append((interval_start, min(previous_time + 5.0, end), status))
    return intervals


# 每行对应一个控制器，两列分别展示西向东和东向西的主路车辆轨迹。
# Each row is one controller; columns show westbound/eastbound arterial trajectories.
def plot_window(start: float, end: float, output_name: str, title: str) -> Path:
    fig, axes = plt.subplots(3, 2, figsize=(13, 10), sharex=True, sharey=True)
    directions = ("WE_MAIN", "EW_MAIN")
    direction_titles = ("West to east (A0-B0-C0)", "East to west (C0-B0-A0)")
    line_colors = ("#2166ac", "#b35806")

    for row_index, controller in enumerate(CONTROLLERS):
        prefix = f"{controller.lower()}_run{RUN_IDX}_seed{EVAL_SEED}"
        trajectory_path = DATA_DIR / f"{prefix}_trajectories.csv"
        signal_path = DATA_DIR / f"{prefix}_signals.csv"
        if not trajectory_path.exists() or not signal_path.exists():
            raise FileNotFoundError(f"Run test.py first; missing {trajectory_path} or {signal_path}")
        trajectories = read_csv(trajectory_path)
        signals = read_csv(signal_path)

        for column_index, (direction, direction_title, line_color) in enumerate(
            zip(directions, direction_titles, line_colors)
        ):
            ax = axes[row_index, column_index]
            vehicle_points = defaultdict(list)
            for point in trajectories:
                time_s = float(point["time_s"])
                if point["direction"] == direction and start <= time_s <= end:
                    vehicle_points[point["vehicle_id"]].append((time_s, float(point["x_m"])))

            for points in vehicle_points.values():
                if len(points) >= 2:
                    points.sort()
                    ax.plot(
                        [point[0] for point in points],
                        [point[1] for point in points],
                        color=line_color,
                        linewidth=0.65,
                        alpha=0.45,
                    )

            # 三个路口的真实主路绿/黄/红状态画在对应空间位置上。
            # Draw true main-road green/yellow/red intervals at each junction position.
            for signal_id, x_m in SIGNAL_X.items():
                for interval_start, interval_end, status in signal_intervals(
                    signals, signal_id, start, end
                ):
                    ax.hlines(
                        x_m,
                        interval_start,
                        interval_end,
                        color=STATUS_COLORS[status],
                        linewidth=5.0,
                        alpha=0.75,
                        zorder=3,
                    )

            if row_index == 0:
                ax.set_title(direction_title)
            if column_index == 0:
                ax.set_ylabel(f"{CONTROLLER_TITLES[controller]}\nCorridor position")
            ax.set_xlim(start, end)
            ax.set_ylim(-110, 710)
            ax.set_yticks([100, 300, 500], labels=["A0", "B0", "C0"])
            ax.grid(axis="x", color="#d9d9d9", linewidth=0.6)

    axes[-1, 0].set_xlabel("Simulation time (s)")
    axes[-1, 1].set_xlabel("Simulation time (s)")
    fig.suptitle(title, fontsize=14, y=0.995)
    fig.legend(
        handles=[
            Line2D([0], [0], color="#2166ac", label="West-to-east vehicle"),
            Line2D([0], [0], color="#b35806", label="East-to-west vehicle"),
            Line2D([0], [0], color=STATUS_COLORS["main_green"], linewidth=5, label="Main-road green"),
            Line2D([0], [0], color=STATUS_COLORS["yellow"], linewidth=5, label="Yellow"),
            Line2D([0], [0], color=STATUS_COLORS["main_red"], linewidth=5, label="Main-road not green"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.962),
        ncol=5,
        frameon=False,
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    output_path = FIGURE_DIR / output_name
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> None:
    outputs = (
        plot_window(
            1600.0,
            2000.0,
            "01_M04_M05_M06_Peak_TimeSpace.png",
            f"Bidirectional time-space diagrams: peak period (run {RUN_IDX}, seed {EVAL_SEED})",
        ),
        plot_window(
            400.0,
            800.0,
            "02_M04_M05_M06_OffPeak_TimeSpace.png",
            f"Bidirectional time-space diagrams: off-peak period (run {RUN_IDX}, seed {EVAL_SEED})",
        ),
    )
    for output in outputs:
        print(f"Saved {output}")


if __name__ == "__main__":
    main()
