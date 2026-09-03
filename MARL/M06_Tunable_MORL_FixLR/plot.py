"""
可调偏好多目标绿波模型的权重扫描结果绘图脚本。
(Plot preference sweep results for the tunable multi-objective green-wave model.)

本脚本读取 eval_tunable_preference_sweep.py 生成的 CSV，不重新运行 SUMO。
它用于回答两个问题：
1. 在当前 preference 范围内，哪些权重区域等待时间更低；
2. 哪些权重区域主路 no-stop rate 更高，但是否牺牲了支路等待。
(This script visualizes trade-offs from the preference sweep CSV.)

输入和图片均位于当前模块对应 run 的 logs/tunable_preference_sweep 目录。
Inputs and figures stay in the selected run's logs/tunable_preference_sweep folder.
"""

import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# RUN_IDX 选择要绘制的扫描结果，路径固定在当前正式模块中。
# RUN_IDX selects the sweep result, with paths fixed inside this formal module.
RUN_IDX = int(os.environ.get("JUC_RUN_IDX", "1"))
RUN_PREFIX = "fixlr_m06_morl_tunable_run_"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "logs",
    f"{RUN_PREFIX}{RUN_IDX}",
    "tunable_preference_sweep",
)
OFAT_CSV = os.path.join(OUTPUT_DIR, "tunable_ofat_results_raw.csv")
OFAT_SEED_GLOB = os.path.join(OUTPUT_DIR, "tunable_ofat_results_seed_*.csv")
LEGACY_CSV = os.path.join(OUTPUT_DIR, "tunable_sweep_results.csv")
RESULT_CSV = os.environ.get(
    "JUC_SWEEP_CSV",
    OFAT_CSV if os.path.exists(OFAT_CSV) else LEGACY_CSV,
)

OBJECTIVE_NAMES = ("delay", "green_wave", "free_flow", "side_fairness", "spillback")


def ensure_input_exists():
    if not os.path.exists(RESULT_CSV) and not glob.glob(OFAT_SEED_GLOB):
        print(f"Preference sweep CSV not found: {RESULT_CSV}")
        return False
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    return True


def load_results():
    seed_paths = sorted(glob.glob(OFAT_SEED_GLOB))
    if not seed_paths:
        return pd.read_csv(RESULT_CSV)

    frames = [pd.read_csv(path) for path in seed_paths]
    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates(["preference_id", "evaluation_seed"], keep="last")
    objective_order = {name: index for index, name in enumerate(OBJECTIVE_NAMES)}
    df["_objective_order"] = df["sweep_objective"].map(objective_order)
    df = df.sort_values(
        ["_objective_order", "sweep_level_index", "evaluation_seed"]
    ).drop(columns="_objective_order")
    df.to_csv(OFAT_CSV, index=False, encoding="utf-8-sig")
    print(f"Merged {len(seed_paths)} seed files into {OFAT_CSV} ({len(df)} rows)")
    return df


def clean_numeric_columns(df):
    # CSV 可能被 Excel 打开保存过，显式转数值可以避免绘图时类型错误。
    # (Coerce numeric columns in case the CSV has been edited in Excel.)
    numeric_candidates = [
        *(f"raw_{name}" for name in OBJECTIVE_NAMES),
        *(f"weight_{name}" for name in OBJECTIVE_NAMES),
        "average_waiting_time",
        "average_queue_length",
        "final_waiting_time",
        "final_queue_length",
        "side_max_waiting",
        "side_mean_max_waiting",
        "main_mean_max_waiting",
        "main_green_ratio",
        "side_green_ratio",
        "mean_no_stop_rate",
        "sweep_level",
        "sweep_level_index",
        "target_raw_weight",
        "target_normalized_weight",
        "evaluation_seed",
    ]
    for column in numeric_candidates:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


def minmax(series, invert=False):
    # 指标归一化只用于综合排序，不改变原始物理指标。
    # (Metric normalization is only for ranking and does not change raw metrics.)
    values = pd.to_numeric(series, errors="coerce").astype(float)
    min_value = values.min()
    max_value = values.max()
    if not np.isfinite(min_value) or not np.isfinite(max_value) or max_value == min_value:
        scaled = pd.Series(np.zeros(len(values)), index=values.index)
    else:
        scaled = (values - min_value) / (max_value - min_value)
    return 1.0 - scaled if invert else scaled


def add_ranking_columns(df):
    # balance_score 越低越好：低等待、低支路最大等待、高 no-stop rate。
    # (Lower balance_score means lower delay, lower side wait, and higher no-stop rate.)
    df = df.copy()
    df["rank_waiting_norm"] = minmax(df["average_waiting_time"], invert=False)
    df["rank_side_wait_norm"] = minmax(df["side_max_waiting"], invert=False)
    df["rank_no_stop_norm"] = minmax(df["mean_no_stop_rate"], invert=True)
    df["balance_score"] = (
        0.45 * df["rank_waiting_norm"]
        + 0.35 * df["rank_side_wait_norm"]
        + 0.20 * df["rank_no_stop_norm"]
    )
    return df


def pareto_front(df):
    # 三目标 Pareto 判断：等待越低越好，支路最大等待越低越好，no-stop rate 越高越好。
    # (Pareto front: minimize waiting and side wait, maximize no-stop rate.)
    metric_df = df[["average_waiting_time", "side_max_waiting", "mean_no_stop_rate"]].copy()
    metric_df = metric_df.replace([np.inf, -np.inf], np.nan).dropna()
    if metric_df.empty:
        return pd.Series(False, index=df.index)

    is_front = pd.Series(False, index=df.index)
    valid_indices = list(metric_df.index)
    for idx in valid_indices:
        row = metric_df.loc[idx]
        dominated = False
        for other_idx in valid_indices:
            if other_idx == idx:
                continue
            other = metric_df.loc[other_idx]
            no_worse = (
                other["average_waiting_time"] <= row["average_waiting_time"]
                and other["side_max_waiting"] <= row["side_max_waiting"]
                and other["mean_no_stop_rate"] >= row["mean_no_stop_rate"]
            )
            strictly_better = (
                other["average_waiting_time"] < row["average_waiting_time"]
                or other["side_max_waiting"] < row["side_max_waiting"]
                or other["mean_no_stop_rate"] > row["mean_no_stop_rate"]
            )
            if no_worse and strictly_better:
                dominated = True
                break
        is_front.loc[idx] = not dominated
    return is_front


def save_best_table(df):
    # 输出一张候选权重表，后续代表性时空图脚本可以直接读取。
    # (Save selected candidate preferences for representative time-space plotting.)
    selectors = [
        ("best_waiting", df["average_waiting_time"].idxmin()),
        ("best_no_stop", df["mean_no_stop_rate"].idxmax()),
        ("best_side_protection", df["side_max_waiting"].idxmin()),
        ("best_balance", df["balance_score"].idxmin()),
    ]

    selected_rows = []
    used_indices = set()
    for label, idx in selectors:
        if idx in used_indices:
            continue
        row = df.loc[idx].copy()
        row["selection_label"] = label
        selected_rows.append(row)
        used_indices.add(idx)

    best_df = pd.DataFrame(selected_rows)
    best_path = os.path.join(OUTPUT_DIR, "tunable_sweep_best_candidates.csv")
    best_df.to_csv(best_path, index=False, encoding="utf-8-sig")
    print(f"Saved best candidate table: {best_path}")
    return best_df


def plot_tradeoff_scatter(df):
    # 散点图：横轴等待时间，纵轴 no-stop rate，颜色表示支路最大等待。
    # (Trade-off scatter: waiting vs no-stop rate, colored by side-street max wait.)
    fig, ax = plt.subplots(figsize=(10, 7))
    color_values = df["side_max_waiting"].astype(float)
    size_values = 80 + 260 * minmax(df["main_green_ratio"]).fillna(0.0)

    scatter = ax.scatter(
        df["average_waiting_time"],
        df["mean_no_stop_rate"] * 100.0,
        c=color_values,
        s=size_values,
        cmap="viridis_r",
        alpha=0.82,
        edgecolors="black",
        linewidths=0.5,
    )

    front_df = df[df["is_pareto_front"]]
    ax.scatter(
        front_df["average_waiting_time"],
        front_df["mean_no_stop_rate"] * 100.0,
        facecolors="none",
        edgecolors="#e74c3c",
        linewidths=2.0,
        s=220,
        label="Pareto front",
    )

    best_idx = df["balance_score"].idxmin()
    best = df.loc[best_idx]
    ax.annotate(
        "best balance",
        xy=(best["average_waiting_time"], best["mean_no_stop_rate"] * 100.0),
        xytext=(12, 12),
        textcoords="offset points",
        arrowprops={"arrowstyle": "->", "color": "#c0392b"},
        color="#c0392b",
        fontsize=10,
    )

    ax.set_title(f"Run {RUN_IDX}: Tunable Preference Trade-off")
    ax.set_xlabel("Average waiting-time proxy (s)")
    ax.set_ylabel("Mean arterial no-stop rate (%)")
    ax.grid(True, linestyle=":", alpha=0.55)
    ax.legend(loc="best")
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Side-street max waiting (s)")
    save_path = os.path.join(OUTPUT_DIR, "tunable_sweep_tradeoff_scatter.png")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.savefig(os.path.splitext(save_path)[0] + ".pdf")
    plt.close()
    print(f"Saved: {save_path}")


def plot_heatmap(df, value_column, title, file_name, colorbar_label):
    # 网格扫描结果可以画成 delay-green_wave 热力图；非网格结果会自动跳过。
    # (Grid sweep results can be plotted as delay-green_wave heatmaps.)
    if "weight_delay" not in df.columns or "weight_green_wave" not in df.columns:
        return

    pivot = df.pivot_table(
        index="weight_green_wave",
        columns="weight_delay",
        values=value_column,
        aggfunc="mean",
    ).sort_index(ascending=True)

    if pivot.shape[0] < 2 or pivot.shape[1] < 2:
        print(f"Skip heatmap {file_name}: not enough grid points.")
        return

    fig, ax = plt.subplots(figsize=(10, 7))
    image = ax.imshow(
        pivot.values,
        origin="lower",
        aspect="auto",
        cmap="magma_r" if "waiting" in value_column else "viridis",
    )
    ax.set_title(title)
    ax.set_xlabel("Normalized delay weight")
    ax.set_ylabel("Normalized green-wave weight")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels([f"{value:.2f}" for value in pivot.columns], rotation=45, ha="right")
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels([f"{value:.2f}" for value in pivot.index])

    for y_idx in range(pivot.shape[0]):
        for x_idx in range(pivot.shape[1]):
            value = pivot.values[y_idx, x_idx]
            if np.isfinite(value):
                text = f"{value:.1f}" if value_column != "mean_no_stop_rate" else f"{value * 100:.1f}%"
                ax.text(x_idx, y_idx, text, ha="center", va="center", fontsize=8, color="white")

    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label(colorbar_label)
    save_path = os.path.join(OUTPUT_DIR, file_name)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.savefig(os.path.splitext(save_path)[0] + ".pdf")
    plt.close()
    print(f"Saved: {save_path}")


def plot_parallel_metrics(df):
    # 平行坐标风格图：把不同权重点在关键指标上的相对表现放在一张图里。
    # (Parallel-style normalized plot for comparing candidate preferences.)
    metric_columns = [
        "average_waiting_time",
        "side_max_waiting",
        "average_queue_length",
        "mean_no_stop_rate",
        "main_green_ratio",
    ]
    available = [column for column in metric_columns if column in df.columns]
    if len(available) < 3:
        return

    normalized = pd.DataFrame(index=df.index)
    for column in available:
        invert = column in ("mean_no_stop_rate", "main_green_ratio")
        normalized[column] = minmax(df[column], invert=invert)

    fig, ax = plt.subplots(figsize=(11, 6))
    x_positions = np.arange(len(available))
    for _, row in df.iterrows():
        alpha = 0.25 if not row["is_pareto_front"] else 0.9
        linewidth = 1.0 if not row["is_pareto_front"] else 2.2
        color = "#7f8c8d" if not row["is_pareto_front"] else "#e74c3c"
        ax.plot(x_positions, normalized.loc[row.name, available], color=color, alpha=alpha, linewidth=linewidth)

    ax.set_xticks(x_positions)
    ax.set_xticklabels(
        [
            "waiting\nlow good",
            "side max wait\nlow good",
            "queue\nlow good",
            "no-stop\nhigh good",
            "main green\nhigh good",
        ][: len(available)]
    )
    ax.set_ylabel("Normalized score, lower is better")
    ax.set_title(f"Run {RUN_IDX}: Normalized Metric Profiles")
    ax.grid(True, axis="y", linestyle=":", alpha=0.55)
    save_path = os.path.join(OUTPUT_DIR, "tunable_sweep_metric_profiles.png")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.savefig(os.path.splitext(save_path)[0] + ".pdf")
    plt.close()
    print(f"Saved: {save_path}")


OFAT_METRICS = (
    "average_waiting_time",
    "average_queue_length",
    "final_waiting_time",
    "final_queue_length",
    "vehicles_seen",
    "main_vehicles_seen",
    "side_vehicles_seen",
    "main_mean_max_waiting",
    "side_mean_max_waiting",
    "side_max_waiting",
    "main_green_ratio",
    "side_green_ratio",
    "mean_no_stop_rate",
    "progression_segment_count",
)

OBJECTIVE_LABELS = {
    "delay": "Delay",
    "green_wave": "Green wave",
    "free_flow": "Free flow",
    "side_fairness": "Side fairness",
    "spillback": "Spillback",
}

OBJECTIVE_COLORS = {
    "delay": "#1f4e79",
    "green_wave": "#2e8b57",
    "free_flow": "#e67e22",
    "side_fairness": "#8e44ad",
    "spillback": "#c0392b",
}


def summarize_ofat(df):
    """Create one plotting row per objective/weight level with seed statistics."""
    static_columns = [
        *(f"raw_{name}" for name in OBJECTIVE_NAMES),
        *(f"weight_{name}" for name in OBJECTIVE_NAMES),
        "target_raw_weight",
        "target_normalized_weight",
    ]
    rows = []
    for (_objective, _level), group in df.groupby(
        ["sweep_objective", "sweep_level_index"], sort=False
    ):
        first = group.iloc[0]
        row = {
            "preference_id": first["preference_id"],
            "sweep_objective": first["sweep_objective"],
            "sweep_level_index": int(first["sweep_level_index"]),
            "sweep_level": float(first["sweep_level"]),
            "evaluation_count": int(len(group)),
            "evaluation_seeds": ",".join(
                str(int(value)) for value in sorted(group["evaluation_seed"].dropna().unique())
            ),
        }
        for column in static_columns:
            if column in group.columns:
                row[column] = float(first[column])
        for metric in OFAT_METRICS:
            if metric not in group.columns:
                continue
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            row[f"{metric}_mean"] = float(values.mean()) if len(values) else np.nan
            row[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            row[f"{metric}_sem"] = (
                float(values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else 0.0
            )
            row[f"{metric}_min"] = float(values.min()) if len(values) else np.nan
            row[f"{metric}_max"] = float(values.max()) if len(values) else np.nan
        rows.append(row)

    summary = pd.DataFrame(rows)
    order = {name: index for index, name in enumerate(OBJECTIVE_NAMES)}
    summary["_objective_order"] = summary["sweep_objective"].map(order)
    summary = summary.sort_values(["_objective_order", "sweep_level_index"]).drop(
        columns="_objective_order"
    )
    return summary


def save_ofat_config(df):
    seeds = ",".join(str(int(value)) for value in sorted(df["evaluation_seed"].unique()))
    rows = []
    for objective in OBJECTIVE_NAMES:
        target_rows = df[df["sweep_objective"] == objective]
        midpoint_rows = df[df["sweep_objective"] != objective]
        midpoint = midpoint_rows[f"raw_{objective}"].median()
        rows.append(
            {
                "run": f"{RUN_PREFIX}{RUN_IDX}",
                "objective": objective,
                "raw_range_min": target_rows[f"raw_{objective}"].min(),
                "raw_midpoint_fixed_when_not_swept": midpoint,
                "raw_range_max": target_rows[f"raw_{objective}"].max(),
                "sweep_steps": target_rows["sweep_level_index"].nunique(),
                "evaluation_repetitions": target_rows["evaluation_seed"].nunique(),
                "evaluation_seeds": seeds,
                "x_axis": "relative position within each objective's raw training range",
                "fixed_values": "other four raw weights at their training-range midpoints",
                "policy_input": "full five-dimensional vector normalized to sum to one",
                "uncertainty_band": "mean plus/minus one sample standard deviation across SUMO seeds",
                "traffic_demand": "same traffic.random.rou.xml for all evaluations",
            }
        )
    config_path = os.path.join(OUTPUT_DIR, "tunable_ofat_experiment_config.csv")
    pd.DataFrame(rows).to_csv(config_path, index=False, encoding="utf-8-sig")
    print(f"Saved OFAT experiment config: {config_path}")


def plot_ofat_response(summary, metric, ylabel, title, file_name):
    fig, ax = plt.subplots(figsize=(10, 6.5))
    for objective in OBJECTIVE_NAMES:
        group = summary[summary["sweep_objective"] == objective].sort_values(
            "sweep_level_index"
        )
        x = group["sweep_level"].to_numpy(dtype=float) * 100.0
        mean = group[f"{metric}_mean"].to_numpy(dtype=float)
        std = group[f"{metric}_std"].fillna(0.0).to_numpy(dtype=float)
        color = OBJECTIVE_COLORS[objective]
        ax.plot(
            x,
            mean,
            color=color,
            marker="o",
            markersize=5,
            linewidth=2.0,
            label=OBJECTIVE_LABELS[objective],
        )
        ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.14, linewidth=0)

    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Position within the swept objective's training range (%)")
    ax.set_ylabel(ylabel)
    ax.set_xticks(np.linspace(0, 100, 5))
    ax.set_xlim(-2, 102)
    ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.45)
    ax.legend(title="Swept objective", ncol=2, frameon=True)
    ax.text(
        0.01,
        0.01,
        "Other four raw weights fixed at midpoint; line = mean, band = +/-1 SD (5 SUMO seeds)",
        transform=ax.transAxes,
        fontsize=8.5,
        color="#555555",
    )
    save_path = os.path.join(OUTPUT_DIR, file_name)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    fig.savefig(os.path.splitext(save_path)[0] + ".pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


def main():
    if not ensure_input_exists():
        return
    df = load_results()
    df = clean_numeric_columns(df)

    if "sweep_objective" in df.columns:
        summary = summarize_ofat(df)
        summary_path = os.path.join(OUTPUT_DIR, "tunable_ofat_results_summary.csv")
        summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
        print(f"Saved OFAT summary: {summary_path}")
        save_ofat_config(df)
        plot_ofat_response(
            summary,
            metric="average_waiting_time",
            ylabel="Average network waiting-time proxy (s)",
            title=f"Run {RUN_IDX}: Waiting-Time Response to Objective Preferences",
            file_name="tunable_ofat_waiting_response.png",
        )
        plot_ofat_response(
            summary,
            metric="average_queue_length",
            ylabel="Average queue length (vehicles)",
            title=f"Run {RUN_IDX}: Queue-Length Response to Objective Preferences",
            file_name="tunable_ofat_queue_response.png",
        )
        return

    df = add_ranking_columns(df)
    df["is_pareto_front"] = pareto_front(df)

    enriched_csv = os.path.join(OUTPUT_DIR, "tunable_sweep_results_ranked.csv")
    df.to_csv(enriched_csv, index=False, encoding="utf-8-sig")
    print(f"Saved ranked sweep CSV: {enriched_csv}")

    save_best_table(df)
    plot_tradeoff_scatter(df)
    plot_heatmap(
        df,
        value_column="average_waiting_time",
        title=f"Run {RUN_IDX}: Average Waiting across Tunable Preferences",
        file_name="tunable_sweep_waiting_heatmap.png",
        colorbar_label="Average waiting-time proxy (s)",
    )
    plot_heatmap(
        df,
        value_column="mean_no_stop_rate",
        title=f"Run {RUN_IDX}: No-stop Rate across Tunable Preferences",
        file_name="tunable_sweep_no_stop_heatmap.png",
        colorbar_label="Mean arterial no-stop rate",
    )
    plot_heatmap(
        df,
        value_column="side_max_waiting",
        title=f"Run {RUN_IDX}: Side-street Max Waiting across Tunable Preferences",
        file_name="tunable_sweep_side_wait_heatmap.png",
        colorbar_label="Side-street max waiting (s)",
    )
    plot_parallel_metrics(df)


if __name__ == "__main__":
    main()
