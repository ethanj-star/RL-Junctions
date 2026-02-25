from stable_baselines3 import PPO # 或者 DQN
from sumo_rl import SumoEnvironment
from envs.wrappers import ThreeJunctionCentralizedWrapper

# 1. 创建底层多智能体环境
raw_env = SumoEnvironment(
    net_file='SUMOroutes.net.xml',
    route_file='traffic.rou.rou.xml',
    out_csv_name='logs/output',
    use_gui=False,  # 训练时建议关掉 GUI，速度快100倍！
    num_seconds=3600
)

# 2. 套上你的包装类
env = ThreeJunctionCentralizedWrapper(raw_env)

# 3. 直接喂给 SB3 训练
model = PPO("MlpPolicy", env, verbose=1)
print("开始训练...")
model.learn(total_timesteps=100000)

# 4. 保存模型
model.save("saved_models/ppo_3juc")
env.close()