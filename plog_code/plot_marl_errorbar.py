import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os
import re

# ====== 解决 Matplotlib 中文显示问题 ======
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# ================= 1. 核心配置区 =================
# 填入你多次训练 (不同 Seed 或 Run) 的 logs 文件夹绝对/相对路径
RUN_DIRS = [
    r"C:\Users\DJI\Desktop\dissertation\3JucRL\logs\single_queue_run_1",
    r"C:\Users\DJI\Desktop\dissertation\3JucRL\logs\single_queue_run_2",
r"C:\Users\DJI\Desktop\dissertation\3JucRL\logs\single_queue_run_3"
]

# 实验名称（将显示在图例中）multiagent
#EXP_NAME = "MARL PPO"
# 如果画单智能体，改为 "Single-Agent PPO"
EXP_NAME = "Single-Agent PPO"

# 曲线颜色配置 (保持和之前一致)
# MARL 推荐色: 等待时间 '#e74c3c'(红), 排队长度 '#2ecc71'(绿)
# Single 推荐色: 等待时间 '#9b59b6'(紫), 排队长度 '#3498db'(蓝)
COLOR_WAIT = '#e74c3c'
COLOR_QUEUE = '#2ecc71'

# 平滑窗口大小
WINDOW_SIZE = 5


def process_multiple_runs():
    all_runs_data = []
    print(f"正在读取 {len(RUN_DIRS)} 个实验批次的数据，请稍候...")

    for run_idx, run_dir in enumerate(RUN_DIRS):
        if not os.path.exists(run_dir):
            print(f" 警告：找不到文件夹 {run_dir}，已跳过。")
            continue

        # 抓取当前 Seed 文件夹下所有的 CSV
        csv_files = glob.glob(os.path.join(run_dir, "*.csv"))
        # 排除测试阶段生成的日志
        csv_files = [f for f in csv_files if "test" not in os.path.basename(f).lower()]

        df_list = []
        for f in csv_files:
            basename = os.path.basename(f)
            try:
                # 精准提取 Episode
                match = re.search(r'ep(\d+)\.csv', basename)
                if match:
                    ep_num = int(match.group(1))
                    temp_df = pd.read_csv(f)
                    temp_df['episode'] = ep_num
                    df_list.append(temp_df)
            except Exception as e:
                pass

        if not df_list:
            continue

        # 拼接当前 Seed 下的所有文件
        df_run = pd.concat(df_list, ignore_index=True)

        # 1. 聚合当前 Seed 下多进程的统一表现
        run_ep_stats = df_run.groupby('episode').agg({
            'system_total_waiting_time': 'mean',
            'system_total_stopped': 'mean'
        }).reset_index()

        run_ep_stats = run_ep_stats.sort_values(by='episode').reset_index(drop=True)

        # 2. 对当前 Seed 的曲线进行时间序列平滑处理
        run_ep_stats['smoothed_waiting'] = run_ep_stats['system_total_waiting_time'].rolling(window=WINDOW_SIZE,
                                                                                             min_periods=1).mean()
        run_ep_stats['smoothed_stopped'] = run_ep_stats['system_total_stopped'].rolling(window=WINDOW_SIZE,
                                                                                        min_periods=1).mean()

        # 3. 打上身份标签，证明它属于哪个 Seed
        run_ep_stats['Run_ID'] = f"Seed_{run_idx}"
        all_runs_data.append(run_ep_stats)
        print(f" 成功加载 {run_dir} (最大回合: {run_ep_stats['episode'].max()})")

    if not all_runs_data:
        return pd.DataFrame()

    # 将所有 Seed 的数据合并成一个用来画 Error Bar 的大表
    return pd.concat(all_runs_data, ignore_index=True)


if __name__ == '__main__':
    df_all = process_multiple_runs()

    if df_all.empty:
        print(" 提取到的数据为空，请检查路径是否正确！")
        exit()

    print(f"\n 数据处理完毕！开始绘制跨越 {df_all['Run_ID'].nunique()} 个种子的 Error Bar 图表...")

    # ====== 3. 开始画图 (融合原版高级样式) ======
    sns.set_theme(style="whitegrid", font="SimHei")
    fig, axes = plt.subplots(2, 1, figsize=(10, 10), dpi=300)

    # === 图 1: Episode vs Waiting Time ===
    ax1 = axes[0]
    # 散点：把所有 run 的真实原始波动值打上去（作为朦胧的底色）
    sns.scatterplot(data=df_all, x='episode', y='system_total_waiting_time',
                    color='gray', alpha=0.15, s=15, ax=ax1, label='Raw Data (各 Seed 原始波动)')

    # 核心：利用 Seaborn 的 errorbar 自动计算多个 Seed 在同一 Episode 的均值和标准差！
    sns.lineplot(data=df_all, x='episode', y='smoothed_waiting',
                 errorbar='sd',  # 计算标准差作为阴影
                 color=COLOR_WAIT, linewidth=2.5, ax=ax1,
                 label=f'{EXP_NAME} (Mean ± SD across seeds)')

    ax1.set_title(f'{EXP_NAME} 多种子汇总：回合数 vs 平均等待时间', fontsize=15, fontweight='bold')
    ax1.set_xlabel('Number of Episodes (训练回合)', fontsize=12)
    ax1.set_ylabel('Waiting Time (seconds)', fontsize=12)
    ax1.legend()

    # === 图 2: Episode vs Queue Length (Stopped Vehicles) ===
    ax2 = axes[1]
    sns.scatterplot(data=df_all, x='episode', y='system_total_stopped',
                    color='gray', alpha=0.15, s=15, ax=ax2, label='Raw Data (各 Seed 原始波动)')

    sns.lineplot(data=df_all, x='episode', y='smoothed_stopped',
                 errorbar='sd',
                 color=COLOR_QUEUE, linewidth=2.5, ax=ax2,
                 label=f'{EXP_NAME} (Mean ± SD across seeds)')

    ax2.set_title(f'{EXP_NAME} 多种子汇总：回合数 vs 平均排队长度', fontsize=15, fontweight='bold')
    ax2.set_xlabel('Number of Episodes (训练回合)', fontsize=12)
    ax2.set_ylabel('Queue Length (vehicles)', fontsize=12)
    ax2.legend()

    plt.tight_layout()

    # 自动保存在脚本运行的当前目录下
    save_name = f"aggregated_errorbar_{EXP_NAME.replace(' ', '_')}.png"
    plt.savefig(save_name, bbox_inches='tight')
    plt.close()

    print(f" 完美！高级学术对比图已保存至:\n {os.path.abspath(save_name)}")