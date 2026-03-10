import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os
import numpy as np

# 解决 Matplotlib 中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


def plot_sumorl_learning_curve():
    print("正在查找 logs 文件夹下的训练数据...")

    # 查找所有匹配前缀的 CSV 文件 (处理 sumo-rl 可能输出多个 episode 文件的情况)
    #csv_files = glob.glob("logs/output*.csv")
    # 查找多智能体 (MARL) 的训练日志
    #csv_files = glob.glob("logs/marl_output*.csv")
    # 查找单智能体 (Single Agent) 的训练日志
    # 精准锁定你外层文件夹里的单智能体日志
    csv_files = glob.glob(r"C:\Users\DJI\Desktop\dissertation\3JucRL\logs\output_process_single_*.csv")


    if not csv_files:
        print("未找到训练日志，请确认 train.py 已经运行并生成了 logs/output*.csv 文件！")
        return

    # 读取并合并所有 CSV 数据
    df_list = []
    for f in csv_files:
        temp_df = pd.read_csv(f)
        # 如果是单文件追加模式，文件里会有 step 列循环，我们需要自己划分 Episode
        df_list.append(temp_df)

    df_all = pd.concat(df_list, ignore_index=True)

    # 如果原始数据里没有 episode 列，我们通过 step 的循环来推断 Episode
    if 'episode' not in df_all.columns:
        # 当 step 突然变小（回到 0 或 1），说明开始了新的 Episode
        df_all['episode'] = (df_all['step'].diff() < 0).cumsum() + 1
        # 处理第一行
        df_all.loc[0, 'episode'] = 1

    print(f"共解析到 {df_all['episode'].max()} 个 Episode 的数据。")

    # 动态检查存在哪些列，防止因为找不到 reward 报错
    agg_dict = {
        'system_total_waiting_time': 'mean',
        'system_total_stopped': 'mean'
    }
    if 'reward' in df_all.columns:
        agg_dict['reward'] = 'sum'

    # 按 Episode 分组，计算每个回合的【平均等待时间】和【平均排队长度】
    episode_stats = df_all.groupby('episode').agg(agg_dict).reset_index()

    # ---- 新增平滑处理代码 ----
    window_size = 5  # 窗口大小，数字越大曲线越平滑，建议设为 5 或 10

    # 计算滑动平均值
    episode_stats['smoothed_waiting'] = episode_stats['system_total_waiting_time'].rolling(window=window_size,
                                                                                           min_periods=1).mean()
    episode_stats['smoothed_stopped'] = episode_stats['system_total_stopped'].rolling(window=window_size,
                                                                                      min_periods=1).mean()

    # 计算波动的标准差（用于画阴影），使用 fillna(0) 防止第一行报错
    episode_stats['std_waiting'] = episode_stats['system_total_waiting_time'].rolling(window=window_size,
                                                                                      min_periods=1).std().fillna(0)
    episode_stats['std_stopped'] = episode_stats['system_total_stopped'].rolling(window=window_size,
                                                                                 min_periods=1).std().fillna(0)

    # ---- 开始画图 (论文同款风格) ----
    sns.set_theme(style="whitegrid", font="SimHei")

    # 我们画两张图：一张等待时间，一张排队长度
    fig, axes = plt.subplots(2, 1, figsize=(10, 10))

    # === 图 1: Episode vs Waiting Time ===
    ax1 = axes[0]

    # 把原始散点画在底层（浅灰色）
    sns.scatterplot(data=episode_stats, x='episode', y='system_total_waiting_time',
                    color='gray', alpha=0.3, s=25, ax=ax1, label='Raw Data (原始数据)')

    # 画平滑后的主线
    sns.lineplot(data=episode_stats, x='episode', y='smoothed_waiting',
                 color='#9b59b6', linewidth=2.5, ax=ax1, label='PPO Agent (Smoothed)')

    # 画半透明的波动阴影带（完美还原论文误差棒）
    ax1.fill_between(episode_stats['episode'],
                     episode_stats['smoothed_waiting'] - episode_stats['std_waiting'],
                     episode_stats['smoothed_waiting'] + episode_stats['std_waiting'],
                     color='#9b59b6', alpha=0.2, label='Variance (波动范围)')

    ax1.set_title('图 4-6 风格：回合数 vs 平均等待时间', fontsize=15, fontweight='bold')
    ax1.set_xlabel('Number of Episodes (训练回合)', fontsize=12)
    ax1.set_ylabel('Waiting Time (seconds)', fontsize=12)
    ax1.legend()

    # === 图 2: Episode vs Queue Length (Stopped Vehicles) ===
    ax2 = axes[1]

    # 原始散点
    sns.scatterplot(data=episode_stats, x='episode', y='system_total_stopped',
                    color='gray', alpha=0.3, s=25, ax=ax2, label='Raw Data (原始数据)')

    # 画平滑后的主线
    sns.lineplot(data=episode_stats, x='episode', y='smoothed_stopped',
                 color='#3498db', linewidth=2.5, ax=ax2, label='PPO Agent (Smoothed)')

    # 波动阴影
    ax2.fill_between(episode_stats['episode'],
                     episode_stats['smoothed_stopped'] - episode_stats['std_stopped'],
                     episode_stats['smoothed_stopped'] + episode_stats['std_stopped'],
                     color='#3498db', alpha=0.2, label='Variance (波动范围)')

    ax2.set_title('图 7 风格：回合数 vs 平均排队长度', fontsize=15, fontweight='bold')
    ax2.set_xlabel('Number of Episodes (训练回合)', fontsize=12)
    ax2.set_ylabel('Queue Length (vehicles)', fontsize=12)
    ax2.legend()

    plt.tight_layout()
    save_path = r"C:\Users\DJI\Desktop\dissertation\3JucRL\paper_style_learning_curves.png"
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"绘图完成！已保存为: {save_path}")


if __name__ == "__main__":
    plot_sumorl_learning_curve()