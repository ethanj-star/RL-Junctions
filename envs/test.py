import gymnasium as gym
from stable_baselines3 import PPO
from sumo_rl import SumoEnvironment
from envs.wrappers import ThreeJunctionCentralizedWrapper


def run_test():
    print("正在加载环境和模型...")

    # 1. 创建底层的 SUMO 环境 (这次一定要把 use_gui 改为 True，为了看动画)
    raw_env = SumoEnvironment(
        net_file='../SUMOroutes.net.xml',
        route_file='../traffic.rou.rou.xml',
        out_csv_name='logs/test_output',
        use_gui=True,  # 开启可视化界面
        num_seconds=3600  # 跑 1 个小时的仿真
    )

    # 2. 套上我们之前写的包装类
    env = ThreeJunctionCentralizedWrapper(raw_env)

    # 3. 加载你训练好的模型
    # 注意路径要和你 train.py 里保存的路径完全一致
    model_path = "../防覆盖log/ppo_3juc_multi.zip"
    model = PPO.load(model_path)

    print("模型加载成功 开始仿真测试...")

    # 4. 运行交互循环
    obs, info = env.reset()
    done = False
    step = 0
    total_reward = 0.0

    while not done:
        # 让 AI 大脑根据当前看到的路况 (obs) 做出预测决策
        # deterministic=True 表示使用确定的最优策略，而不是训练时的随机探索
        action, _states = model.predict(obs, deterministic=True)

        # 将决策传达给环境，执行动作
        obs, reward, terminated, truncated, info = env.step(action)

        total_reward += reward
        step += 1
        done = terminated or truncated

    print(f"测试结束！")
    print(f"总共运行步数: {step}")
    print(f"该回合总奖励 (拥堵越少负数越小): {total_reward}")

    env.close()


if __name__ == "__main__":
    run_test()