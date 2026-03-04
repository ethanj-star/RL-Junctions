import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecMonitor, VecNormalize, VecEnvWrapper
from sumo_rl import parallel_env
import supersuit as ss


# ====== 魔法补丁：解决 5 个返回值与 4 个返回值的 API 世纪冲突 ======
class SB3CompatibilityWrapper(VecEnvWrapper):
    def __init__(self, venv):
        super().__init__(venv)

    def reset(self):
        # 如果新版环境返回 (obs, info) 两个值，我们只取 obs 喂给老实巴交的 SB3
        obs = self.venv.reset()
        if isinstance(obs, tuple) and len(obs) == 2:
            return obs[0]
        return obs

    def step_async(self, actions):
        self.venv.step_async(actions)

    def step_wait(self):
        results = self.venv.step_wait()
        # 如果是新版的 5 个返回值 (obs, reward, terminated, truncated, info)
        if len(results) == 5:
            obs, rews, terms, truncs, infos = results
            # 把 terminated 和 truncated 合并回 done
            dones = np.logical_or(terms, truncs)
            return obs, rews, dones, infos
        # 如果已经是 4 个返回值了，原样返回
        return results


if __name__ == '__main__':
    print("正在初始化多智能体 SUMO 环境...")

    # 1. 创建原生的 PettingZoo 多智能体并行环境
    env = parallel_env(
        net_file='SUMOroutes.net.xml',
        route_file='traffic.rou.rou.xml',
        out_csv_name='logs/marl_output',
        use_gui=False,
        num_seconds=3600,
        reward_fn='pressure'
    )

    # 2. SuperSuit 魔法转换
    env = ss.pettingzoo_env_to_vec_env_v1(env)

    # 3. 拼接成向量环境 (单核，但自带3路口并行)
    env = ss.concat_vec_envs_v1(
        env,
        num_vec_envs=1,
        num_cpus=1,
        base_class='stable_baselines3'
    )

    # 4. 【关键】套上我们的魔法补丁，把 5 个球变成 4 个球！
    env = SB3CompatibilityWrapper(env)

    # 5. 包装 Monitor
    env = VecMonitor(env)

    # 6. 包装 VecNormalize (防崩溃护甲)
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)

    # 7. 创建 PPO 模型
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-5,
        n_steps=1024,
        batch_size=256,
        n_epochs=5,
        clip_range=0.2,
        ent_coef=0.01,
        target_kl=0.05,
        verbose=1,
        device="cpu"
    )

    print("环境就绪！SB3 正在接收多智能体数据流。开始训练...")
    model.learn(total_timesteps=200000)

    # 8. 保存模型和归一化参数
    model.save("saved_models/ppo_marl_3juc")
    env.save("saved_models/vec_normalize_marl.pkl")

    env.close()
    print("多智能体训练完成，模型与日志均已保存！")