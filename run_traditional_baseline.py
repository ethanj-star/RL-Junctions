import os
import traci
import numpy as np

# ====== 1. 核心路径配置 ======
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# 如果你把这个文件建在根目录，ROOT_DIR 就是 CURRENT_DIR；如果是子目录，请用 os.path.dirname(CURRENT_DIR)
ROOT_DIR = CURRENT_DIR

net_path = os.path.join(ROOT_DIR, 'SUMOroutes.net.xml')
route_path = os.path.join(ROOT_DIR, 'traffic.rou.rou.xml')

# SUMO 启动命令 (纯后台运行，速度极快)
sumo_cmd = [
    "sumo",
    "-n", net_path,
    "-r", route_path,
    "--no-warnings", "true"
]


def run_baseline():
    print("🚀 正在启动 SUMO 原生交通灯基线测试 (不带 AI)...")
    traci.start(sumo_cmd)

    step = 0
    total_stopped_list = []
    total_waiting_time_list = []

    # 运行 3600 秒 (与你的 RL 仿真时间保持绝对一致)
    while step < 3600:
        traci.simulationStep()

        # 每隔 5 秒（与 RL 的默认 delta_time 一致）采样一次数据
        if step % 5 == 0:
            vehicles = traci.vehicle.getIDList()

            # 【口径对齐 1】：统计全网停滞车辆数 (速度 < 0.1m/s)
            halt_num = sum([1 for veh in vehicles if traci.vehicle.getSpeed(veh) < 0.1])
            total_stopped_list.append(halt_num)

            # 【口径对齐 2】：统计全网单次连续等待时间（彻底与 sumo-rl 源码对齐！）
            wait_time = sum([traci.vehicle.getWaitingTime(veh) for veh in vehicles])
            total_waiting_time_list.append(wait_time)
        step += 1

    traci.close()

    # ====== 3. 计算理论最优/基线的平均值 ======
    best_stopped = np.mean(total_stopped_list)
    best_waiting = np.mean(total_waiting_time_list)

    print("\n🎉 基线测试完成！这是你路网的原生实力：")
    print("=" * 60)
    print("请将以下两行代码直接复制填入你的画图脚本中：")
    print(f"BEST_QUEUE_LENGTH = {best_stopped:.2f}  # 对应图 2")
    print(f"BEST_WAITING_TIME = {best_waiting:.2f}  # 对应图 1")
    print("=" * 60)


if __name__ == "__main__":
    run_baseline()