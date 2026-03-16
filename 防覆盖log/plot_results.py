import xml.etree.ElementTree as ET
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 解决 Matplotlib 中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


def parse_stats(file_path):
    """解析宏观统计文件 stats.xml"""
    if not os.path.exists(file_path):
        return None
    tree = ET.parse(file_path)
    root = tree.getroot()
    trip_stats = root.find('vehicleTripStatistics')
    if trip_stats is not None:
        return {
            'count': float(trip_stats.get('count')),
            'waitingTime': float(trip_stats.get('waitingTime')),
            'timeLoss': float(trip_stats.get('timeLoss')),
            'speed': float(trip_stats.get('speed')),
            'duration': float(trip_stats.get('duration'))
        }
    return None


def parse_tripinfo(file_path, label):
    """解析微观车辆数据 tripinfo.xml"""
    if not os.path.exists(file_path):
        return pd.DataFrame()
    tree = ET.parse(file_path)
    root = tree.getroot()

    data = []
    for trip in root.findall('tripinfo'):
        data.append({
            'depart': float(trip.get('depart')),
            'waitingTime': float(trip.get('waitingTime')),
            'timeLoss': float(trip.get('timeLoss')),
            'routeLength': float(trip.get('routeLength')),
            'Group': label
        })
    return pd.DataFrame(data)


def main():
    print("正在读取数据...")
    # 1. 提取宏观数据
    baseline_stats = parse_stats("../logs/baseline_stats.xml")
    ai_stats = parse_stats("../logs/ai_stats.xml")

    # 2. 提取微观数据
    df_baseline = parse_tripinfo("../logs/baseline_tripinfo.xml", "固定配时 (Baseline)")
    df_ai = parse_tripinfo("../logs/ai_tripinfo.xml", "强化学习 (AI)")

    df_all = pd.concat([df_baseline, df_ai], ignore_index=True)

    if baseline_stats is None or ai_stats is None or df_all.empty:
        print("未找到数据文件！请确保 logs 文件夹下有 baseline 和 ai 的 xml 文件。")
        return

    print("数据读取成功，开始绘制图表...")
    sns.set_theme(style="whitegrid", font="SimHei")  # 使用 Seaborn 美化

    # ================= 图 1：宏观核心指标柱状图 =================
    plt.figure(figsize=(10, 6))
    categories = ['平均排队等待时间 (s)', '平均时间损耗 (s)', '单车平均总耗时 (s)']
    baseline_values = [baseline_stats['waitingTime'], baseline_stats['timeLoss'], baseline_stats['duration']]
    ai_values = [ai_stats['waitingTime'], ai_stats['timeLoss'], ai_stats['duration']]

    x = range(len(categories))
    width = 0.35

    plt.bar([i - width / 2 for i in x], baseline_values, width=width, label='固定配时 (Baseline)', color='#e74c3c')
    plt.bar([i + width / 2 for i in x], ai_values, width=width, label='强化学习 (AI)', color='#2ecc71')

    plt.xticks(x, categories, fontsize=12)
    plt.ylabel('时间 (秒)', fontsize=12)
    plt.title('整体路网通行核心指标对比', fontsize=15, fontweight='bold')
    plt.legend(fontsize=12)

    # 在柱子上添加具体数值
    for i, v in enumerate(baseline_values):
        plt.text(i - width / 2, v + 1, str(v), ha='center', fontsize=11)
    for i, v in enumerate(ai_values):
        plt.text(i + width / 2, v + 1, str(v), ha='center', fontsize=11)

    plt.tight_layout()
    plt.savefig('logs/1_macro_comparison_bar.png', dpi=300)
    plt.close()

    # ================= 图 2：微观排队时间密度分布图 =================
    plt.figure(figsize=(10, 6))
    sns.kdeplot(data=df_all, x="waitingTime", hue="Group", fill=True, common_norm=False, palette=['#e74c3c', '#2ecc71'],
                alpha=.5)
    plt.title('车辆排队等待时间密度分布', fontsize=15, fontweight='bold')
    plt.xlabel('单车排队等待时间 (秒)', fontsize=12)
    plt.ylabel('车辆密度 (概率)', fontsize=12)
    plt.xlim(0, max(df_all['waitingTime']) * 0.8)  # 截断极端极值以保证图表美观
    plt.tight_layout()
    plt.savefig('logs/2_waiting_time_density.png', dpi=300)
    plt.close()

    # ================= 图 3：随时间变化的拥堵散点图 (展示应对突发流的能力) =================
    plt.figure(figsize=(12, 6))
    sns.scatterplot(data=df_all, x="depart", y="waitingTime", hue="Group", palette=['#e74c3c', '#2ecc71'], s=15,
                    alpha=0.6)

    # 标注出我们设定的特殊事件时间轴
    plt.axvline(x=300, color='gray', linestyle='--', label='B0路口突发拥堵开始 (300s)')
    plt.axvline(x=1500, color='gray', linestyle='-.', label='B0路口突发拥堵结束 (1500s)')
    plt.axvline(x=1800, color='purple', linestyle=':', label='主干道潮汐流反转 (1800s)')

    plt.title('发车时间与单车排队时间的关系 (动态响应测试)', fontsize=15, fontweight='bold')
    plt.xlabel('车辆出发时间 (仿真秒)', fontsize=12)
    plt.ylabel('排队等待时间 (秒)', fontsize=12)
    plt.legend(loc='upper right', fontsize=10)
    plt.tight_layout()
    plt.savefig('logs/3_time_series_scatter.png', dpi=300)
    plt.close()

    print("🎉 绘图完成！请去 logs 文件夹下查看 3 张 PNG 图片。")


if __name__ == "__main__":
    main()