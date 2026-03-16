import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os
import re

# ====== 解决 Matplotlib 中文显示问题 ======
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# ====== 1. 核心路径配置 (直接使用绝对路径最稳妥) ======
# 根据你的报错信息，你的 logs 真实路径是在 01_Baseline_SingleAgent 文件夹下
RUN_IDX = 1  # 你想画第几次的图
LOG_DIR = rf"C:\Users\DJI\Desktop\dissertation\3JucRL\logs\single_queue_run_{RUN_IDX}"


def plot_sumorl_learning_curve():
    print(f"正在读取文件夹: {LOG_DIR}")

    # 抓取所有符合 output_*.csv 的文件
    csv_pattern = os.path.join(LOG_DIR, "output_*.csv")
    csv_files = glob.glob(csv_pattern)

    if not csv_files:
        print(f"❌ 未找到任何文件，请检查路径是否正确: \n{LOG_DIR}")
        return

    # ====== 2. 读取并提取文件名中的 Episode ======
    df_list = []
    for f in csv_files:
        basename = os.path.basename(f)  # 例如: output_0_conn0_ep1.csv
        try:
            # 使用正则表达式提取 'ep' 后面的数字
            match = re.search(r'ep(\d+)\.csv', basename)
            if match:
                ep_num = int(match.group(1))
            else:
                continue  # 如果匹配不到，跳过该文件

            temp_df = pd.read_csv(f)
            # 直接将提取出的真实回合数作为新列加入！
            temp_df['episode'] = ep_num
            df_list.append(temp_df)

        except Exception as e:
            print(f"解析文件 {basename} 时出错: {e}")

    # 把所有小碎片数据合并成一个大表
    df_all = pd.concat(df_list, ignore_index=True)
    print(f"✅ 成功解析 {len(csv_files)} 个碎片文件，最大训练回合数为: {df_all['episode'].max()}")

    # ====== 3. 数据聚合与平滑计算 ======
    agg_dict = {
        'system_total_waiting_time': 'mean',
        'system_total_stopped': 'mean'
    }
    if 'reward' in df_all.columns:
        agg_dict['reward'] = 'sum'

    # 按 Episode 分组，计算【平均等待时间】和【平均排队长度】
    episode_stats = df_all.groupby('episode').agg(agg_dict).reset_index()

    # 按照 episode 从小到大排序 (非常重要，防止曲线乱飞)
    episode_stats = episode_stats.sort_values(by='episode').reset_index(drop=True)

    # 平滑处理代码 (Window Size 推荐 5 到 10)
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

    # ====== 4. 开始画图 (论文同款风格) ======
    sns.set_theme(style="whitegrid", font="SimHei")
    fig, axes = plt.subplots(2, 1, figsize=(10, 10))

    # === 图 1: Episode vs Waiting Time ===
    ax1 = axes[0]
    sns.scatterplot(data=episode_stats, x='episode', y='system_total_waiting_time',
                    color='gray', alpha=0.3, s=25, ax=ax1, label='Raw Data (各进程汇总原始值)')
    sns.lineplot(data=episode_stats, x='episode', y='smoothed_waiting',
                 color='#9b59b6', linewidth=2.5, ax=ax1, label='PPO Agent (Smoothed)')
    ax1.fill_between(episode_stats['episode'],
                     episode_stats['smoothed_waiting'] - episode_stats['std_waiting'],
                     episode_stats['smoothed_waiting'] + episode_stats['std_waiting'],
                     color='#9b59b6', alpha=0.2, label='Variance (波动范围)')

    ax1.set_title(f'Run {RUN_IDX}：回合数 vs 平均等待时间', fontsize=15, fontweight='bold')
    ax1.set_xlabel('Number of Episodes (训练回合)', fontsize=12)
    ax1.set_ylabel('Waiting Time (seconds)', fontsize=12)
    ax1.legend()

    # === 图 2: Episode vs Queue Length (Stopped Vehicles) ===
    ax2 = axes[1]
    sns.scatterplot(data=episode_stats, x='episode', y='system_total_stopped',
                    color='gray', alpha=0.3, s=25, ax=ax2, label='Raw Data (各进程汇总原始值)')
    sns.lineplot(data=episode_stats, x='episode', y='smoothed_stopped',
                 color='#3498db', linewidth=2.5, ax=ax2, label='PPO Agent (Smoothed)')
    ax2.fill_between(episode_stats['episode'],
                     episode_stats['smoothed_stopped'] - episode_stats['std_stopped'],
                     episode_stats['smoothed_stopped'] + episode_stats['std_stopped'],
                     color='#3498db', alpha=0.2, label='Variance (波动范围)')

    ax2.set_title(f'Run {RUN_IDX}：回合数 vs 平均排队长度', fontsize=15, fontweight='bold')
    ax2.set_xlabel('Number of Episodes (训练回合)', fontsize=12)
    ax2.set_ylabel('Queue Length (vehicles)', fontsize=12)
    ax2.legend()

    plt.tight_layout()

    # 图片自动保存在 logs 文件夹下
    save_path = os.path.join(LOG_DIR, f"run_{RUN_IDX}_learning_curves.png")
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"🎉 绘图完成！图片已自动保存至:\n👉 {save_path}")


if __name__ == "__main__":
    plot_sumorl_learning_curve()