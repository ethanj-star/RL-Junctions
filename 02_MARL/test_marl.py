import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, VecEnvWrapper
from sumo_rl import parallel_env
import supersuit as ss
import random
import torch


def run_marl_test():
    # 加入这几行，保证每次测同一个模型，跑出的分数小数点都不差！!!!!每次更改seed
    seed = 868
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    print(f"正在加载第 {RUN_IDX} 次 MARL 训练的环境和模型...")


# 核心路径动态获取 (与 train_marl.py 保持同步)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CURRENT_DIR)

net_path = os.path.join(ROOT_DIR, 'SUMOroutes.net.xml')
route_path = os.path.join(ROOT_DIR, 'traffic.rou.rou.xml')

# ！！！！！每次更改 手动指定你要测试哪一次训练的模型
# 比如你想测试第 1 次跑出来的模型，就写 1
RUN_IDX = 1
RUN_DIR = os.path.join(ROOT_DIR, 'saved_models', f'marl_run_{RUN_IDX}')
LOG_DIR = os.path.join(ROOT_DIR, 'logs', f'marl_run_{RUN_IDX}')

# 默认加载训练结束时保存的最终模型
MODEL_PATH = os.path.join(RUN_DIR, 'ppo_marl_model.zip')
VEC_NORM_PATH = os.path.join(RUN_DIR, 'vec_normalize_marl.pkl')

# 【高级玩法】：如果你不想测最终模型，而是想测训练到一半的某个 Checkpoint (比如 300000 步)
# 请取消下面这行代码的注释，并修改对应的步数：
# MODEL_PATH = os.path.join(RUN_DIR, 'checkpoints', 'rl_model_300000_steps.zip')


# 魔法补丁来处理 API 版本冲突
class SB3CompatibilityWrapper(VecEnvWrapper):
    def __init__(self, venv):
        super().__init__(venv)

    def reset(self):
        obs = self.venv.reset()
        if isinstance(obs, tuple) and len(obs) == 2:
            return obs[0]
        return obs

    def step_async(self, actions):
        self.venv.step_async(actions)

    def step_wait(self):
        results = self.venv.step_wait()
        if len(results) == 5:
            obs, rews, terms, truncs, infos = results
            dones = np.logical_or(terms, truncs)
            return obs, rews, dones, infos
        return results


def run_marl_test():
    print(f"正在加载第 {RUN_IDX} 次 MARL 训练的环境和模型...")

    # 创建原生的 PettingZoo 多智能体并行环境 (开启 GUI 看动画)
    env = parallel_env(
        net_file=net_path,
        route_file=route_path,
        out_csv_name=os.path.join(LOG_DIR, 'test_marl_output'), # 测试数据也会自动存在对应的 run 文件夹下
        use_gui=True,  # 开启可视化界面
        num_seconds=3600,
        reward_fn='queue'  # 必须和训练时保持一致
    )

    # SuperSuit 魔法转换 (还原训练时的架构)
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, num_vec_envs=1, num_cpus=1, base_class='stable_baselines3')

    # 套上 5 返回值兼容补丁
    env = SB3CompatibilityWrapper(env)

    # 加载训练时保存的 VecNormalize 统计数据 (戴上眼镜)
    if not os.path.exists(VEC_NORM_PATH):
        print(f" 找不到归一化文件: {VEC_NORM_PATH}，AI 将无法理解环境！")
        return

    env = VecNormalize.load(VEC_NORM_PATH, env)
    # 告诉它这是考试不是训练，不要再更新均值和方差了！
    env.training = False
    # 测试时我们想看真实的原始奖励（比如 -500），而不是被缩放后的小数（比如 -0.2）
    env.norm_reward = False

    # 5. 加载你训练好的 MARL 模型
    if not os.path.exists(MODEL_PATH):
        print(f" 找不到模型文件: {MODEL_PATH}，请检查 RUN_IDX 编号或文件路径。")
        return

    model = PPO.load(MODEL_PATH)
    print(" 模型和归一化参数加载成功！开始仿真测试...")

    # 6. 运行交互循环
    obs = env.reset()
    step = 0
    total_reward = 0.0

    # 注意：因为是向量化环境，done 也是一个数组 (比如 [False, False, False])
    while True:
        # deterministic=True 表示不掷骰子，严格按照训练出来的最优策略执行
        action, _states = model.predict(obs, deterministic=True)

        obs, rewards, dones, infos = env.step(action)

        # 累加 3 个路口的真实奖励
        total_reward += np.sum(rewards)
        step += 1

        # 如果任何一个伪环境(智能体)结束，或者整体超时，就结束测试
        if np.any(dones):
            break

    print(f"\n 测试结束！")
    print(f"总共运行控制步数: {step}")
    print(f"3个路口总累计奖励 (排队越少负数越小): {total_reward:.2f}")

    env.close()

if __name__ == "__main__":
    run_marl_test()