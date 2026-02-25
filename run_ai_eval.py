import gymnasium as gym
from stable_baselines3 import PPO
from sumo_rl import SumoEnvironment
from envs.wrappers import ThreeJunctionCentralizedWrapper


def run_ai_evaluation():
    print("正在加载 AI 模型并生成统计报表...")

    # 1. 创建底层环境
    raw_env = SumoEnvironment(
        net_file='SUMOroutes.net.xml',
        route_file='traffic.rou.rou.xml',
        out_csv_name='logs/ai_test_output',
        use_gui=False,  # 关掉 GUI，让后台全速计算，几秒钟就能跑完拿数据！
        num_seconds=3600,
        # 👇 核心在这里：让 AI 跑的时候也顺便生成这两份报表！
        additional_sumo_cmd="--statistic-output logs/ai_stats.xml --tripinfo-output logs/ai_tripinfo.xml"
    )

    # 2. 套上包装类
    env = ThreeJunctionCentralizedWrapper(raw_env)

    # 3. 加载你刚刚训练好的满级 AI
    model = PPO.load("saved_models/ppo_3juc")

    # 4. 让 AI 接管并跑完这 3600 秒
    obs, info = env.reset()
    done = False

    print("AI 正在指挥交通，请稍候...")
    while not done:
        # deterministic=True 表示不瞎探索，直接用训练出的最强武功
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

    print("✅ AI 测试跑完啦！环境关闭中...")
    env.close()
    print("🎉 恭喜！去 logs 文件夹下查看 ai_stats.xml 吧！")


if __name__ == "__main__":
    run_ai_evaluation()