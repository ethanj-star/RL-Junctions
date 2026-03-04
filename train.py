import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, VecNormalize
from sumo_rl import SumoEnvironment
from envs.wrappers import ThreeJunctionCentralizedWrapper


def make_env(rank, seed=0):
    def _init():
        out_csv = f'logs/output_process_{rank}'
        raw_env = SumoEnvironment(
            net_file='SUMOroutes.net.xml',
            route_file='traffic.rou.rou.xml',
            out_csv_name=out_csv,
            use_gui=False,
            num_seconds=3600
        )
        env = ThreeJunctionCentralizedWrapper(raw_env)
        return env

    return _init


if __name__ == '__main__':
    num_cpu = 4
    print(f"🚀 正在后台启动 {num_cpu} 个并行的 SUMO 环境...")

    env = SubprocVecEnv([make_env(i) for i in range(num_cpu)])
    env = VecMonitor(env)

    # ================== 核心优化 1：数据归一化 ==================
    # 它可以自动把观测状态和奖励都缩放到均值为0，方差为1的小范围内。
    # 这样即使 SUMO 发生了死锁，爆出了极其巨大的负奖励，VecNormalize 也会把它温柔地压制住，防止梯度爆炸！
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)

    # ================== 核心优化 2：PPO 参数的“稳如老狗”配置 ==================
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-5,  # 【调优】进一步降低学习率（从1e-4降到3e-5），步子迈小点
        n_steps=1024,  # 【调优】减少每次更新前走的路，更频繁地小步修正
        batch_size=256,
        n_epochs=5,  # 【调优】对同一批数据只学习5遍（默认10），防止对偶发拥堵过度反应
        clip_range=0.2,
        ent_coef=0.005,  # 【调优】稍微降低探索率，不让它乱尝试危险动作
        target_kl=0.05,  # 【终极刹车】限制每次策略改变的最大距离 (KL Divergence)，如果步子太大自动停止当前更新！
        verbose=1,
        device="cpu"
    )

    print(f"🔥 环境就绪！开始多进程并行训练...")
    model.learn(total_timesteps=200000)

    # 保存模型
    model.save("saved_models/ppo_3juc_multi")
    # 【非常重要】如果用了 VecNormalize，必须把它的统计数据也存下来！
    env.save("saved_models/vec_normalize.pkl")

    env.close()
    print("✅ 训练完成，模型已保存！")