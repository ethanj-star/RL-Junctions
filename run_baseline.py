import traci
import os


def run_fixed_time_baseline():
    print("开始运行固定配时 (Baseline) 仿真...")

    # 确保 logs 文件夹存在
    if not os.path.exists('logs'):
        os.makedirs('logs')

    # 配置纯 SUMO 的启动命令
    # 这里用 "sumo" 而不是 "sumo-gui"，因为后台计算速度极快，几秒钟就能跑完 1 小时的数据
    sumo_cmd = [
        "sumo",
        "-n", "SUMOroutes.net.xml",
        "-r", "traffic.rou.rou.xml",
        "--statistic-output", "logs/baseline_stats.xml",  # 核心输出 1：整体宏观统计
        "--tripinfo-output", "logs/baseline_tripinfo.xml",  # 核心输出 2：每辆车的微观数据
    ]

    # 启动 TraCI 连接
    traci.start(sumo_cmd)

    # 运行 3600 秒
    step = 0
    while step < 3600:
        traci.simulationStep()
        step += 1

    # 关闭连接
    traci.close()
    print("✅ Baseline 仿真结束！")
    print("请去 logs 文件夹下查看 baseline_stats.xml 文件。")


if __name__ == "__main__":
    run_fixed_time_baseline()