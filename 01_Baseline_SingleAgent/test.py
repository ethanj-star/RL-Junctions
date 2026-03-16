import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from sumo_rl import SumoEnvironment
# 请确保这里的导入路径与 train.py 中保持一致
from wrappers import ThreeJunctionCentralizedWrapper

# ====== 1. 核心路径动态获取 (与 train.py 保持同步) ======
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# 如果你的 test.py 和 train.py 在同一个目录，ROOT_DIR 逻辑保持一致
ROOT_DIR = os.path.dirname(CURRENT_DIR)

net_path = os.path.join(ROOT_DIR, 'SUMOroutes.net.xml')
route_path = os.path.join(ROOT_DIR, 'traffic.rou.rou.xml')

# ====== 2. 手动指定你要测试哪一次训练的模型 ======
# 比如你想测试第 1 次跑出来的模型，就写 1
RUN_IDX = 1
RUN_DIR = os.path.join(ROOT_DIR, 'saved_models', f'single_queue_run_{RUN_IDX}')
MODEL_PATH = os.path.join(RUN_DIR, 'ppo_model.zip')
VEC_NORM_PATH = os.path.join(RUN_DIR, 'vec_normalize.pkl')


def make_test_env():
    """创建一个单进程测试环境"""
    raw_env = SumoEnvironment(
        net_file=net_path,
        route_file=route_path,
        out_csv_name=os.path.join(ROOT_DIR, 'logs', f'test_run_{RUN_IDX}_output'),
        use_gui=True,  # 开启可视化界面看动画
        num_seconds=3600,
        reward_fn='queue'  # 【极其重要】必须和 train.py 中完全一致！
    )
    env = ThreeJunctionCentralizedWrapper(raw_env)
    return env


def run_test():
    print(f"正在加载第 {RUN_IDX} 次训练的环境和模型...")

    # ====== 3. 构建向量化环境并加载归一化统计数据 ======
    # 哪怕只测一个环境，也要套上 DummyVecEnv，因为 VecNormalize 只能包裹 VecEnv
    env = DummyVecEnv([make_test_env])

    # 加载训练时保存的归一化字典
    if os.path.exists(VEC_NORM_PATH):
        env = VecNormalize.load(VEC_NORM_PATH, env)
        # 【极其重要】测试时必须关闭网络参数的更新，以及奖励的归一化，但要保持观测值归一化！
        env.training = False
        env.norm_reward = False
        print("✅ 成功加载 VecNormalize 状态归一化数据！")
    else:
        print(f"❌ 找不到归一化文件: {VEC_NORM_PATH}，请检查路径！")
        return

    # ====== 4. 加载模型 ======
    if os.path.exists(MODEL_PATH):
        model = PPO.load(MODEL_PATH)
        print("✅ 模型加载成功！开始仿真测试...")
    else:
        print(f"❌ 找不到模型文件: {MODEL_PATH}，请检查路径！")
        return

    # ====== 5. 运行交互循环 ======
    # 注意：SB3 的 VecEnv API 和原生的 Gymnasium 略有不同
    # VecEnv 的 reset 只返回 obs (没有 info)
    obs = env.reset()

    # VecEnv 会把 done 变成一个数组，因为我们只有 1 个环境，所以看 dones[0]
    dones = [False]
    step = 0
    total_reward = 0.0

    while not dones[0]:
        # deterministic=True 表示使用确定的最优策略，不包含随机探索
        action, _states = model.predict(obs, deterministic=True)

        # VecEnv 的 step 返回 4 个值，且都是数组形式
        obs, rewards, dones, infos = env.step(action)

        # 累加奖励 (取数组的第0个元素)
        total_reward += rewards[0]
        step += 1

    print(f"\n🎉 测试结束！")
    print(f"总共运行步数: {step}")
    print(f"该回合总奖励 (拥堵越少负数越小): {total_reward:.2f}")

    env.close()


if __name__ == "__main__":
    run_test()