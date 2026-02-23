import os
import sys
import traci

# 1. 检查环境变量是否配置正确，并将 SUMO 的工具库路径加入 Python 搜索路径
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("请先配置环境变量 'SUMO_HOME'")


def run_simulation():
    # 2. 设置启动命令
    # 'sumo-gui' 会打开可视化界面；如果不想看动画只求速度，可以换成 'sumo'
    sumoCmd = ["sumo-gui", "-c", "start.sumocfg"]

    # 3. 启动 TraCI 与 SUMO 的连接
    traci.start(sumoCmd)
    print("成功连接到 SUMO！")

    # 4. 让仿真跑起来（这里测试跑 1000 步）
    step = 0
    while step < 1000:
        traci.simulationStep()  # 让 SUMO 往前推进一秒（或一个步长）

        # --- 这里就是未来你写 RL 算法的地方 ---
        # 比如：读取当前红绿灯状态、读取车辆排队长度、改变红绿灯颜色等

        step += 1

    # 5. 仿真结束，关闭连接
    traci.close()
    print("仿真测试结束。")


if __name__ == "__main__":
    run_simulation()