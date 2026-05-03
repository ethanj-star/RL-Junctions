import os
import random
import torch
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import VecMonitor, VecNormalize, VecEnvWrapper
from sumo_rl import parallel_env
import supersuit as ss
from typing import Callable

# 终极修复：直接导入模块，拒绝 os.system 的静默失败
from generate_Random_Traffic import generate_route_file
# 新增导入：用于重写观察空间实现通信
from sumo_rl.environment.observations import DefaultObservationFunction
from gymnasium import spaces


def linear_schedule_with_min(initial_value: float, min_value: float) -> Callable[[float], float]:
    """带保底机制的线性衰减学习率生成器"""

    def func(progress_remaining: float) -> float:
        return min_value + progress_remaining * (initial_value - min_value)

    return func


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CURRENT_DIR)
net_path = os.path.join(ROOT_DIR, 'SUMOroutes.net.xml')
route_path = os.path.join(ROOT_DIR, 'traffic.random.rou.xml')


def get_next_run_number(base_dir, prefix="marl_run_"):
    if not os.path.exists(base_dir):
        return 1
    existing_runs = []
    for folder in os.listdir(base_dir):
        if folder.startswith(prefix):
            try:
                num = int(folder.replace(prefix, ""))
                existing_runs.append(num)
            except ValueError:
                continue
    return max(existing_runs) + 1 if existing_runs else 1


class SB3CompatibilityWrapper(VecEnvWrapper):
    """补丁：解决 5 个返回值与 4 个返回值的 API 冲突"""

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


def custom_green_wave_reward(traffic_signal):
    """
    平衡修复版奖励：强化 PBRS 惩罚权重 + 削弱动能奖金 + 辅路熔断机制
    """
    my_id = traffic_signal.id
    current_step = getattr(traffic_signal.env, "sim_step", 0)

    current_phase = traffic_signal.sumo.trafficlight.getPhase(traffic_signal.id)
    is_main_green = (current_phase == 0)

    # 1. 时间戳维护 (保持不变，供观测器使用)
    if is_main_green:
        if not hasattr(traffic_signal, 'my_green_start'):
            traffic_signal.my_green_start = current_step
    else:
        if hasattr(traffic_signal, 'my_green_start'):
            delattr(traffic_signal, 'my_green_start')

    # 2. PBRS 计算 (强化惩罚力度)
    total_queue = traffic_signal.get_total_queued()
    # 【修复 1】：将归一化系数从 100.0 改为 10.0，让排队的惩罚痛感增加 10 倍！
    base_penalty = -total_queue / 10.0
    phi_current = -total_queue / 10.0

    delta_time = getattr(traffic_signal.env, "delta_time", 5)
    if getattr(traffic_signal, 'last_potential', None) is None or current_step <= delta_time:
        traffic_signal.last_potential = phi_current
        shaping_reward = 0.0
    else:
        gamma = 0.99
        shaping_reward = (gamma * phi_current) - traffic_signal.last_potential
        traffic_signal.last_potential = phi_current

    pbrs_score = base_penalty + shaping_reward

    # 3. 辅路熔断机制 (防“爆表”终极保险)
    side_street_queue = 0
    # 统计辅路 (带有 top 或 bottom 的车道) 的排队
    for lane in traffic_signal.lanes:
        if "top" in lane or "bottom" in lane:
            side_street_queue += traffic_signal.sumo.lane.getLastStepHaltingNumber(lane)

    # 【修复 3】：如果辅路积压超过 15 辆车，触发熔断！
    is_melt_down = side_street_queue > 15

    # 4. 动能流率奖励 (削弱诱惑)
    kinetic_bonus = 0.0

    # 只有主路绿灯，且没有触发辅路熔断时，才给绿波奖励
    if is_main_green and not is_melt_down:
        for lane in traffic_signal.lanes:
            if "top" not in lane and "bottom" not in lane:
                mean_speed = traffic_signal.sumo.lane.getLastStepMeanSpeed(lane)
                veh_num = traffic_signal.sumo.lane.getLastStepVehicleNumber(lane)

                if mean_speed > 7.0 and veh_num > 0:
                    # 【修复 2】：将系数从 0.2 降到 0.05，大幅削弱动能奖金的量级
                    kinetic_bonus += (mean_speed / 13.89) * veh_num * 0.05

    # 5. 最终结算
    # 如果触发熔断，额外给予一次性重罚，逼迫它立刻切灯！
    meltdown_penalty = -5.0 if is_melt_down else 0.0
    final_reward = pbrs_score + kinetic_bonus + meltdown_penalty
    return final_reward

# Map neighbors
NEIGHBOR_MAP = {
    'A0': [None, 'B0'],
    'B0': ['A0', 'C0'],
    'C0': ['B0', None]
}


class CommObservationFunction(DefaultObservationFunction):
    """
    全知晓雷达观测器：11 维基础 + 6 维深度前馈情报
    每个邻居提供 3 维情报：[主干道排队数(软归一), 是否主路绿灯, 绿灯已持续秒数(归一)]
    """

    def __init__(self, ts):
        super().__init__(ts)

    def observation_space(self):
        base_space = super().observation_space()
        base_dim = base_space.shape[0]

        # 新增 6 维：左邻居(3维) + 右邻居(3维)
        new_dim = base_dim + 6

        return spaces.Box(
            low=np.zeros(new_dim, dtype=np.float32),
            high=np.ones(new_dim, dtype=np.float32) * np.inf,
        )

    def __call__(self):
        base_obs = super().__call__()
        my_id = self.ts.id
        neighbors = NEIGHBOR_MAP.get(my_id, [None, None])

        extra_obs = []
        for neighbor_id in neighbors:
            if neighbor_id is None:
                # 边缘路口的缺失邻居：用 3 个 0.0 占位
                extra_obs.extend([0.0, 0.0, 0.0])
            else:
                neighbor_ts = self.ts.env.traffic_signals.get(neighbor_id)
                if not neighbor_ts:
                    extra_obs.extend([0.0, 0.0, 0.0])
                    continue

                # 情报 1：主干道排队数 (经过软归一化，上限设为 50 辆，防止输入爆炸)
                main_arterial_queue = 0
                for lane in neighbor_ts.lanes:
                    if "top" not in lane and "bottom" not in lane:
                        main_arterial_queue += neighbor_ts.sumo.lane.getLastStepHaltingNumber(lane)
                queue_norm = min(main_arterial_queue / 50.0, 1.0)

                # 情报 2：邻居是否处于主路绿灯相位 (Phase 0)
                current_phase = neighbor_ts.sumo.trafficlight.getPhase(neighbor_id)
                is_main_green = 1.0 if current_phase == 0 else 0.0

                # 情报 3：绿灯已亮秒数前馈 (对齐 max_green=60)
                green_duration_norm = 0.0
                if is_main_green == 1.0 and hasattr(neighbor_ts, 'my_green_start'):
                    current_step = getattr(self.ts.env, "sim_step", 0)
                    green_duration_norm = min((current_step - neighbor_ts.my_green_start) / 60.0, 1.0)

                extra_obs.extend([queue_norm, is_main_green, green_duration_norm])

        final_obs = np.concatenate([base_obs, extra_obs])
        return np.array(final_obs, dtype=np.float32)


if __name__ == '__main__':
    print("正在初始化多智能体 SUMO 环境...")

    seed = 8848
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    saved_models_base = os.path.join(ROOT_DIR, 'saved_models')
    logs_base = os.path.join(ROOT_DIR, 'logs')
    os.makedirs(saved_models_base, exist_ok=True)
    os.makedirs(logs_base, exist_ok=True)

    run_idx = get_next_run_number(saved_models_base, "marl_run_")
    print(f"\n 自动检测到历史记录，本次 MARL 分配为: [ 第 {run_idx} 次运行 ]")

    run_save_dir = os.path.join(saved_models_base, f'marl_run_{run_idx}')
    os.makedirs(run_save_dir, exist_ok=True)
    run_csv_dir = os.path.join(logs_base, f'marl_run_{run_idx}')
    os.makedirs(run_csv_dir, exist_ok=True)
    csv_base_path = os.path.join(run_csv_dir, 'marl_output')
    tensorboard_log_path = os.path.join(logs_base, 'ppo_marl_tb')

    print(" [系统启动] 正在生成初始的泊松随机交通流...")
    generate_route_file()

    env = parallel_env(
        net_file=net_path,
        route_file=route_path,
        out_csv_name=csv_base_path,
        use_gui=False,
        num_seconds=3600,
        reward_fn=custom_green_wave_reward,
        observation_class=CommObservationFunction,
        min_green=5,  # 【修复补回】：必须有这个下限，防止1秒切灯鬼畜
        max_green=60  # 绿灯防死锁安全锁
    )

    # 核心拦截器：猴子补丁
    original_reset = env.reset


    def custom_reset(*args, **kwargs):
        print("\n [完全体通信版] 监听到底层环境重置信号，正在为本局生成全新泊松车流...")
        generate_route_file()
        return original_reset(*args, **kwargs)


    env.reset = custom_reset

    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(
        env,
        num_vec_envs=1,
        num_cpus=1,
        base_class='stable_baselines3'
    )

    env = SB3CompatibilityWrapper(env)
    env = VecMonitor(env)
    env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.)

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=linear_schedule_with_min(3e-4, 3e-5),
        n_steps=2048,
        batch_size=256,
        n_epochs=10,
        clip_range=0.2,
        ent_coef=0.03,  # 增加熵系数，鼓励在拥堵时尝试新动作
        target_kl=0.05,
        verbose=1,
        device="cpu",  # 如果有GPU可改为 "cuda"
        tensorboard_log=tensorboard_log_path
    )

    print("\n" + "=" * 40)
    print(" 神经网络真实维度 (带定向主干道全知晓雷达补丁)：")
    print(f" 状态输入维度 (Observation): {model.policy.observation_space}")
    print(f" 动作输出维度 (Action): {model.policy.action_space}")
    print("=" * 40 + "\n")

    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path=os.path.join(run_save_dir, 'checkpoints'),
        name_prefix='rl_model'
    )

    print("环境就绪！SB3 正在接收附带上游灯色前馈的军用级情报。开始训练...")
    # 由于是 3 个智能体 (concat_vec_envs 会将 1 个 env 乘以 3)，总 step 会消耗得比平时快 3 倍
    model.learn(total_timesteps=300000, callback=checkpoint_callback, tb_log_name=f"run_{run_idx}")

    model.save(os.path.join(run_save_dir, "ppo_marl_model"))
    env.save(os.path.join(run_save_dir, "vec_normalize_marl.pkl"))

    env.close()
    print(f"\n 多智能体训练完成！")