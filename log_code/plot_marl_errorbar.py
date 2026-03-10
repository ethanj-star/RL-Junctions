import os
import glob
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ================= 1. 配置你的 5 个 Seed 文件夹路径 =================
# 注意路径前面的 r 不要删，防止转义报错
folders = [
    r"C:\Users\DJI\Desktop\dissertation\3JucRL\03 queue 500000 s42",
    r"C:\Users\DJI\Desktop\dissertation\3JucRL\04 queue 500000 s2026",
    r"C:\Users\DJI\Desktop\dissertation\3JucRL\05 queue 500000 s99",
    r"C:\Users\DJI\Desktop\dissertation\3JucRL\06 queue 500000 s666"
]

# 解决 Matplotlib 中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


def process_data():
    all_data = []
    print("正在努力读取数百个 CSV 文件，请稍候...")

    # 遍历每个 seed 的文件夹
    for seed_idx, folder in enumerate(folders):
        if not os.path.exists(folder):
            print(f" 警告：找不到文件夹 {folder}")
            continue

        # 寻找该文件夹下所有的 _epXXX.csv 文件
        csv_files = glob.glob(os.path.join(folder, "*_ep*.csv"))

        for file in csv_files:
            # 用正则表达式提取文件名中的回合数 (Episode)
            match = re.search(r'_ep(\d+)\.csv', file)
            if match:
                ep_num = int(match.group(1))

                try:
                    df = pd.read_csv(file)
                    # sumo-rl 的 csv 包含每一步的数据，我们计算整个回合的平均表现
                    # 提取系统平均等待时间和系统总排队数 (如果不叫这两个名字，请打开你的csv看一眼表头并修改这里)
                    ep_mean_wait = df['system_total_waiting_time'].mean()
                    ep_mean_queue = df['system_total_stopped'].mean()

                    all_data.append({
                        'Seed': f"Seed_{seed_idx}",
                        'Episode': ep_num,
                        'Waiting Time (s)': ep_mean_wait,
                        'Queue Length': ep_mean_queue
                    })
                except Exception as e:
                    print(f"读取文件出错 {file}: {e}")

    # 把收集到的所有数据变成一个 Pandas 巨表
    return pd.DataFrame(all_data)


if __name__ == '__main__':
    # 1. 提取合并数据
    df_all = process_data()

    if df_all.empty:
        print("没有读取到任何数据，请检查文件夹路径！")
        exit()

    print(f"数据处理完毕，共读取了 {len(df_all)} 个回合的数据。开始绘制高级学术图表...")

    # 2. 开始画图 (Seaborn 会自动帮我们处理阴影)
    sns.set_theme(style="whitegrid", font="SimHei")
    fig, axes = plt.subplots(2, 1, figsize=(10, 12), dpi=300)

    # ============ 图 1：平均等待时间 Error Bar ============
    # errorbar='sd' 表示绘制标准差阴影；也可以换成 errorbar=('ci', 95) 绘制95%置信区间
    sns.lineplot(
        ax=axes[0],
        data=df_all,
        x='Episode',
        y='Waiting Time (s)',
        errorbar='sd',  # <=== 魔法在这里：自动画出 Error bar 阴影！
        linewidth=2,
        color='#9b59b6',  # 紫色系
        label='MARL (Mean ± SD)'
    )
    axes[0].set_title('图 A：训练回合数 vs 系统平均等待时间', fontsize=15, fontweight='bold')
    axes[0].set_xlabel('Number of Episodes (训练回合)', fontsize=12)
    axes[0].set_ylabel('Waiting Time (seconds)', fontsize=12)
    axes[0].legend()

    # ============ 图 2：排队长度 Error Bar ============
    sns.lineplot(
        ax=axes[1],
        data=df_all,
        x='Episode',
        y='Queue Length',
        errorbar='sd',
        linewidth=2,
        color='#3498db',  # 蓝色系
        label='MARL (Mean ± SD)'
    )
    axes[1].set_title('图 B：训练回合数 vs 系统平均排队长度', fontsize=15, fontweight='bold')
    axes[1].set_xlabel('Number of Episodes (训练回合)', fontsize=12)
    axes[1].set_ylabel('Queue Length (vehicles)', fontsize=12)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig('marl_errorbar_results.png', bbox_inches='tight')
    plt.show()
    print(" 绘制成功！请查看当前目录下的 marl_errorbar_results.png")