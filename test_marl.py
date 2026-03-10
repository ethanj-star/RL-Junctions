import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, VecEnvWrapper
from sumo_rl import parallel_env
import supersuit as ss


# ====== 依然需要这个魔法补丁来处理 API 版本冲突 ======
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
    print("正在加载 MARL 环境和模型...")

    # 创建原生的 PettingZoo 多智能体并行环境 (开启 GUI 看动画)
    env = parallel_env(
        net_file='SUMOroutes.net.xml',  # 注意检查路径，如果你的 test.py 在子文件夹里，这里要加 ../
        route_file='traffic.rou.rou.xml',
        out_csv_name='logs/test_marl_output',
        use_gui=True,  # 开启可视化界面
        num_seconds=3600,
        reward_fn='pressure'  # 必须和训练时保持一致
    )

    # SuperSuit 魔法转换 (还原训练时的架构)
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, num_vec_envs=1, num_cpus=1, base_class='stable_baselines3')

    # 套上 5 返回值兼容补丁
    env = SB3CompatibilityWrapper(env)

    # 加载训练时保存的 VecNormalize 统计数据 (戴上眼镜)
    norm_path = "saved_models/vec_normalize_marl.pkl"
    if not os.path.exists(norm_path):
        print(f"找不到归一化文件: {norm_path}，AI 将无法理解环境！")
        return

    env = VecNormalize.load(norm_path, env)
    # 告诉它这是考试不是训练，不要再更新均值和方差了！
    env.training = False
    # 测试时我们想看真实的原始奖励（比如 -500），而不是被缩放后的小数（比如 -0.2）
    env.norm_reward = False

    # 5. 加载你训练好的 MARL 模型
    model_path = "防覆盖log/ppo_marl_3juc.zip"
    if not os.path.exists(model_path):
        print(f"找不到模型文件: {model_path}")
        return

    model = PPO.load(model_path)
    print("模型和归一化参数加载成功！开始仿真测试")

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

    print(f"\n测试结束！")
    print(f"总共运行控制步数: {step}")
    print(f"3个路口总累计奖励 (Pressure 越接近0越好): {total_reward:.2f}")

    env.close()


if __name__ == "__main__":
    run_marl_test()