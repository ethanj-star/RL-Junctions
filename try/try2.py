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

print("初始状态数据是-", initial_state)

# ... 前面的 SumoEnvironment 初始化代码保持不变 ...

# 1. 重置环境，获取初始状态
obs = env.reset()
print("初始状态:", obs)

# 2. 让仿真跑 100 步测试一下
for step in range(100):
    # A. 为每个路口生成一个随机动作 (0 或 1)
    # 在实际训练中，这里将变成你的神经网络输出的预测动作
    actions = {
        'A0': env.action_space.sample(),
        'B0': env.action_space.sample(),
        'C0': env.action_space.sample()
    }

    # B. 把动作字典传给环境，往前推进一秒
    next_obs, rewards, dones, infos = env.step(actions)

    # C. 每隔 10 步打印一次奖励，看看表现如何
    if step % 10 == 0:
        print(f"--- 第 {step} 步 ---")
        print(f"动作: {actions}")
        print(f"获得奖励: {rewards}")

# 3. 跑完关闭环境
print("测试运行结束！")
env.close()