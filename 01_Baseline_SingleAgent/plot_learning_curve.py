import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os
import re
import numpy as np  # 新增：用于生成等距刻度

# 解决 Matplotlib 中文显示问题  # Fix Matplotlib Chinese display issues
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 核心路径配置 (绝对路径最稳妥)  # Core path configuration
RUN_IDX = 1  # 你想画第几次的图！！！！每次修改  # Which run to plot !!!! Modify this every time
LOG_DIR = rf"C:\Users\DJI\Desktop\dissertation\3JucRL\logs\single_queue_run_{RUN_IDX}"


def plot_sumorl_learning_curve():
    print(f"正在读取文件夹: {LOG_DIR}")

    # 抓取所有符合 output_*.csv 的文件  # Fetch all files matching output_*.csv
    csv_pattern = os.path.join(LOG_DIR, "output_*.csv")
    csv_files = glob.glob(csv_pattern)

    if not csv_files:
        print(f" 未找到任何文件，请检查路径是否正确: \n{LOG_DIR}")
        return

    # 读取并提取文件名中的 Episode
    df_list = []
    for f in csv_files:
        basename = os.path.basename(f)
        try:
            # 使用正则表达式提取 'ep' 后面的数字
            match = re.search(r'ep(\d+)\.csv', basename)
            if match:
                ep_num = int(match.group(1))
            else:
                continue

            temp_df = pd.read_csv(f)
            temp_df['episode'] = ep_num
            df_list.append(temp_df)

        except Exception as e:
            print(f"解析文件 {basename} 时出错: {e}")

    if not df_list:
        print(" 解析失败，提取到的数据为空！")
        return

    # 把所有小碎片数据合并成一个大表
    df_all = pd.concat(df_list, ignore_index=True)
    print(f" 成功解析 {len(df_list)} 个碎片文件，最大训练回合数为: {df_all['episode'].max()}")

    # 数据聚合与平滑计算
    agg_dict = {
        'system_total_waiting_time': 'mean',
        'system_total_stopped': 'mean'
    }
    if 'reward' in df_all.columns:
        agg_dict['reward'] = 'sum'

    # 按 Episode 分组，计算【平均等待时间】和【平均排队长度】
    episode_stats = df_all.groupby('episode').agg(agg_dict).reset_index()

    # 按照 episode 从小到大排序 (非常重要)
    episode_stats = episode_stats.sort_values(by='episode').reset_index(drop=True)

    # 平滑处理代码 (Window Size 与代码1保持一致为5)
    window_size = 5
    episode_stats['smoothed_waiting'] = episode_stats['system_total_waiting_time'].rolling(window=window_size,
                                                                                           min_periods=1).mean()
    episode_stats['smoothed_stopped'] = episode_stats['system_total_stopped'].rolling(window=window_size,
                                                                                      min_periods=1).mean()

    # 计算波动的标准差（用于画阴影带）
    episode_stats['std_waiting'] = episode_stats['system_total_waiting_time'].rolling(window=window_size,
                                                                                      min_periods=1).std().fillna(0)
    episode_stats['std_stopped'] = episode_stats['system_total_stopped'].rolling(window=window_size,
                                                                                 min_periods=1).std().fillna(0)

    # 开始画图
    sns.set_theme(style="whitegrid", font="SimHei")
    fig, axes = plt.subplots(2, 1, figsize=(10, 10))

    # ==========================================
    # 图 1: Episode vs Waiting Time (同步代码1视觉：红色主调)
    # ==========================================
    ax1 = axes[0]
    sns.scatterplot(data=episode_stats, x='episode', y='system_total_waiting_time',
                    color='gray', alpha=0.3, s=25, ax=ax1, label='Raw Data (原始值)')
    sns.lineplot(data=episode_stats, x='episode', y='smoothed_waiting',
                 color='#e74c3c', linewidth=2.5, ax=ax1, label='PPO Agent (Smoothed)')
    ax1.fill_between(episode_stats['episode'],
                     episode_stats['smoothed_waiting'] - episode_stats['std_waiting'],
                     episode_stats['smoothed_waiting'] + episode_stats['std_waiting'],
                     color='#e74c3c', alpha=0.2, label='Variance (波动范围)')

    ax1.set_title(f'Run {RUN_IDX}：Episodes vs Average Waiting Time', fontsize=15, fontweight='bold')
    ax1.set_xlabel('Number of Episodes', fontsize=12)
    ax1.set_ylabel('Average Waiting Time (seconds)', fontsize=12)

    # 【同步坐标轴锁死】
    ax1.set_xlim(0, 500)
    ax1.set_ylim(0, 300)
    ax1.set_xticks(np.arange(0, 501, 20))  # X轴 0-160，步长 20
    ax1.set_yticks(np.arange(0, 301, 30))  # Y轴 30-300，步长 30
    ax1.legend()

    # 【同步：寻找并标注最佳等待时间气泡框】
    best_wait_idx = episode_stats['smoothed_waiting'].idxmin()
    best_wait_val = episode_stats.loc[best_wait_idx, 'smoothed_waiting']
    best_wait_ep = episode_stats.loc[best_wait_idx, 'episode']

    ax1.scatter(best_wait_ep, best_wait_val, color='#c0392b', s=80, zorder=5)
    ax1.annotate(
        f'Best: {best_wait_val:.1f}s\n@ Ep {best_wait_ep}',
        xy=(best_wait_ep, best_wait_val),
        xytext=(20, 30), textcoords='offset points',
        bbox=dict(boxstyle="round,pad=0.4", fc="#fadbd8", ec="#e74c3c", lw=1.5, alpha=0.9),
        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2", color='#c0392b', lw=1.5),
        fontsize=11, fontweight='bold', color='#922b21'
    )

    # ==========================================
    # 图 2: Episode vs Queue Length (同步代码1视觉：绿色主调)
    # ==========================================
    ax2 = axes[1]
    sns.scatterplot(data=episode_stats, x='episode', y='system_total_stopped',
                    color='gray', alpha=0.3, s=25, ax=ax2, label='Raw Data (原始值)')
    sns.lineplot(data=episode_stats, x='episode', y='smoothed_stopped',
                 color='#2ecc71', linewidth=2.5, ax=ax2, label='PPO Agent (Smoothed)')
    ax2.fill_between(episode_stats['episode'],
                     episode_stats['smoothed_stopped'] - episode_stats['std_stopped'],
                     episode_stats['smoothed_stopped'] + episode_stats['std_stopped'],
                     color='#2ecc71', alpha=0.2, label='Variance (波动范围)')

    ax2.set_title(f'Run {RUN_IDX}：Episodes vs Average Queue Length', fontsize=15, fontweight='bold')
    ax2.set_xlabel('Number of Episodes', fontsize=12)
    ax2.set_ylabel('Average Queue Length (vehicles)', fontsize=12)

    # 【同步坐标轴锁死】
    ax2.set_xlim(0, 500)
    ax2.set_xticks(np.arange(0, 501, 20))
    ax2.set_ylim(0, 38)
    ax2.set_yticks(np.arange(0, 39, 6))
    ax2.legend()

    # 【同步：寻找并标注最佳排队长度气泡框】
    best_queue_idx = episode_stats['smoothed_stopped'].idxmin()
    best_queue_val = episode_stats.loc[best_queue_idx, 'smoothed_stopped']
    best_queue_ep = episode_stats.loc[best_queue_idx, 'episode']

    ax2.scatter(best_queue_ep, best_queue_val, color='#27ae60', s=80, zorder=5)
    ax2.annotate(
        f'Best: {best_queue_val:.1f} veh\n@ Ep {best_queue_ep}',
        xy=(best_queue_ep, best_queue_val),
        xytext=(20, 30), textcoords='offset points',
        bbox=dict(boxstyle="round,pad=0.4", fc="#d5f5e3", ec="#2ecc71", lw=1.5, alpha=0.9),
        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2", color='#27ae60', lw=1.5),
        fontsize=11, fontweight='bold', color='#145a32'
    )

    plt.tight_layout()

    # 保持原有的保存逻辑
    save_path = os.path.join(LOG_DIR, f"run_{RUN_IDX}_learning_curves.png")
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f" 绘图完成！图片已自动保存至:\n {save_path}")


if __name__ == "__main__":
    plot_sumorl_learning_curve()