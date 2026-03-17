import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os
import re

# ====== 解决 Matplotlib 中文显示问题 ======
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# ====== 1. 核心路径动态配置 (修复版) ======
# 获取当前脚本所在目录 (例如: 3JucRL/02_MARL)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# 向上一级，获取根目录 (例如: 3JucRL)
ROOT_DIR = os.path.dirname(CURRENT_DIR)

# 指定你想画第几次 MARL 训练的图
RUN_IDX = 1
# 现在统一去根目录下的 logs 里找数据！
LOG_DIR = os.path.join(ROOT_DIR, 'logs', f'marl_run_{RUN_IDX}')


def plot_marl_learning_curve():
    print(f"正在读取 MARL 文件夹: \n👉 {LOG_DIR}")

    # 诊断 1: 检查文件夹到底存不存在
    if not os.path.exists(LOG_DIR):
        print("\n❌ 致命错误：找不到这个文件夹！请检查 RUN_IDX 是否正确。")
        return

    # 抓取所有符合的 csv 文件
    csv_pattern = os.path.join(LOG_DIR, "*.csv")
    csv_files = glob.glob(csv_pattern)

    # 【核心防御】：排除掉 test_marl.py 生成的测试日志，防止污染训练曲线
    csv_files = [f for f in csv_files if "test" not in os.path.basename(f)]

    if not csv_files:
        print(f"❌ 未找到任何训练 CSV 文件，请检查路径: \n{LOG_DIR}")
        return

    # ====== 2. 读取并提取文件名中的 Episode ======
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
        print("❌ 解析失败，提取到的数据为空！")
        return

    # 把所有小碎片数据合并成一个大表
    df_all = pd.concat(df_list, ignore_index=True)
    print(f"✅ 成功解析 {len(df_list)} 个碎片文件，最大训练回合数为: {df_all['episode'].max()}")

    # ====== 3. 数据聚合与平滑计算 ======
    # 保持和单智能体完全一致的度量指标
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
    episode_stats['smoothed_waiting'] = episode_stats['system_total_waiting_time'].rolling(window=window_size, min_periods=1).mean()
    episode_stats['smoothed_stopped'] = episode_stats['system_total_stopped'].rolling(window=window_size, min_periods=1).mean()

    # 计算波动的标准差（用于画阴影带）
    episode_stats['std_waiting'] = episode_stats['system_total_waiting_time'].rolling(window=window_size, min_periods=1).std().fillna(0)
    episode_stats['std_stopped'] = episode_stats['system_total_stopped'].rolling(window=window_size, min_periods=1).std().fillna(0)

    # ====== 4. 开始画图 (论文同款风格) ======
    sns.set_theme(style="whitegrid", font="SimHei")
    fig, axes = plt.subplots(2, 1, figsize=(10, 10))

    # === 图 1: Episode vs Waiting Time ===
    ax1 = axes[0]
    sns.scatterplot(data=episode_stats, x='episode', y='system_total_waiting_time',
                    color='gray', alpha=0.3, s=25, ax=ax1, label='Raw Data (各进程汇总原始值)')
    # MARL 曲线采用红色区分
    sns.lineplot(data=episode_stats, x='episode', y='smoothed_waiting',
                 color='#e74c3c', linewidth=2.5, ax=ax1, label='MARL PPO (Smoothed)')
    ax1.fill_between(episode_stats['episode'],
                     episode_stats['smoothed_waiting'] - episode_stats['std_waiting'],
                     episode_stats['smoothed_waiting'] + episode_stats['std_waiting'],
                     color='#e74c3c', alpha=0.2, label='Variance (波动范围)')

    ax1.set_title(f'MARL Run {RUN_IDX}：回合数 vs 平均等待时间', fontsize=15, fontweight='bold')
    ax1.set_xlabel('Number of Episodes (训练回合)', fontsize=12)
    ax1.set_ylabel('Waiting Time (seconds)', fontsize=12)
    ax1.legend()

    # === 图 2: Episode vs Queue Length (Stopped Vehicles) ===
    ax2 = axes[1]
    sns.scatterplot(data=episode_stats, x='episode', y='system_total_stopped',
                    color='gray', alpha=0.3, s=25, ax=ax2, label='Raw Data (各进程汇总原始值)')
    # MARL 曲线采用绿色区分
    sns.lineplot(data=episode_stats, x='episode', y='smoothed_stopped',
                 color='#2ecc71', linewidth=2.5, ax=ax2, label='MARL PPO (Smoothed)')
    ax2.fill_between(episode_stats['episode'],
                     episode_stats['smoothed_stopped'] - episode_stats['std_stopped'],
                     episode_stats['smoothed_stopped'] + episode_stats['std_stopped'],
                     color='#2ecc71', alpha=0.2, label='Variance (波动范围)')

    ax2.set_title(f'MARL Run {RUN_IDX}：回合数 vs 平均排队长度', fontsize=15, fontweight='bold')
    ax2.set_xlabel('Number of Episodes (训练回合)', fontsize=12)
    ax2.set_ylabel('Queue Length (vehicles)', fontsize=12)
    ax2.legend()

    plt.tight_layout()

    # 图片自动保存在 logs/marl_run_X 文件夹下
    save_path = os.path.join(LOG_DIR, f"marl_run_{RUN_IDX}_learning_curves.png")
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"🎉 绘图完成！图片已自动保存至:\n👉 {save_path}")

if __name__ == "__main__":
    plot_marl_learning_curve()