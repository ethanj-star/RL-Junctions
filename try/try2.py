import gymnasium as gym
from sumo_rl import SumoEnvironment

# 1. 让 sumo-rl 自动加载你的路网和车流
env = SumoEnvironment(
    net_file='C:/Users/DJI/Desktop/dissertation/3JucRL/SUMOroutes.net.xml',
    route_file='C:/Users/DJI/Desktop/dissertation/3JucRL/traffic.rou.xml',
    out_csv_name='logs/output',
    use_gui=True,
    num_seconds=3600,
)

# 2. 打印环境自带的观测空间（State）和动作空间（Action）
print("动作空间:", env.action_space)
print("状态空间:", env.observation_space)

# 3. 启动环境，提取第 0 秒的初始状态
initial_state = env.reset()

print("初始状态数据是:", initial_state)

# 跑完先关闭，不往下循环
env.close()