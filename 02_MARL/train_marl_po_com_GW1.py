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

MAIN_GREEN_STATE = os.environ.get("JUC_MAIN_GREEN_STATE", "rrrrGGggrrrrGGgg")
SIDE_GREEN_STATE = os.environ.get("JUC_SIDE_GREEN_STATE", "GGggrrrrGGggrrrr")


def get_signal_state(sumo, signal_id):
    return sumo.trafficlight.getRedYellowGreenState(signal_id)


def is_main_green_state(state):
    return state == MAIN_GREEN_STATE


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
    终极融合奖励函数：无悖论 PBRS (基于势能的排队惩罚) + 绿波动能时空接力奖励。
    """
    # 获取当前信号灯ID、当前仿真步数和环境步长
    my_id = traffic_signal.id
    current_step = getattr(traffic_signal.env, "sim_step", 0)
    delta_time = getattr(traffic_signal.env, "delta_time", 5)
    # 判断是否为新的仿真回合
    is_new_episode = current_step <= delta_time

    # 跨回合的状态泄露清除 (防止上一个 Episode 的时间戳干扰新 Episode)
    # 如果是新回合，清理上回合残留的绿灯起始时间和相位记录
    if is_new_episode:
        if hasattr(traffic_signal, 'my_green_start'):
            delattr(traffic_signal, 'my_green_start')
        if hasattr(traffic_signal, 'last_signal_state'):
            delattr(traffic_signal, 'last_signal_state')

    # 通过真实灯色判断是否为主路绿灯
    current_state = get_signal_state(traffic_signal.sumo, traffic_signal.id)
    is_main_green = is_main_green_state(current_state)

    # 状态追踪：判断是否“刚刚”切为主路绿灯
    if not hasattr(traffic_signal, 'last_signal_state'):
        traffic_signal.last_signal_state = current_state
    # 如果当前是绿灯且上一帧不是绿灯，说明处于绿灯上升沿（刚变绿）
    just_turned_green = (is_main_green and traffic_signal.last_signal_state != MAIN_GREEN_STATE)
    traffic_signal.last_signal_state = current_state


    # 1. 独立时间戳维护模块

    # 记录主路绿灯的持续状态
    if is_main_green:
        # 首次进入绿灯时，记录绿灯开始的仿真步数
        if not hasattr(traffic_signal, 'my_green_start'):
            traffic_signal.my_green_start = current_step
    else:
        # 一旦不是绿灯，立刻删除绿灯时间戳，为下一次绿灯做准备
        if hasattr(traffic_signal, 'my_green_start'):
            delattr(traffic_signal, 'my_green_start')


    # 2. 纯粹 PBRS 计算模块 (基于绝对排队长度)
    # 获取当前路口所有车道的总排队车辆数
    total_queue = traffic_signal.get_total_queued()
    # 基础惩罚：归一化后的排队数
    # 基础惩罚为负数，排队越长惩罚越大
    base_penalty = -total_queue / 100.0

    # 势能定义：拥堵越严重，势能越低 (负数)
    phi_current = -total_queue / 100.0

    # PBRS (Potential-Based Reward Shaping) 计算逻辑
    if getattr(traffic_signal, 'last_potential', None) is None or is_new_episode:
        # 初始状态无 shaping reward，记录当前势能
        traffic_signal.last_potential = phi_current
        shaping_reward = 0.0
    else:
        # 根据势能差计算 shaping reward，帮助算法稳定收敛并防止刷分
        gamma = 0.99
        shaping_reward = (gamma * phi_current) - traffic_signal.last_potential
        traffic_signal.last_potential = phi_current


    # 3. 绿波联动奖金模块 (时空协同引擎)
    green_wave_bonus = 0.0

    # 专门针对 B0 路口的绿波逻辑（硬编码的路口拓扑）
    if my_id == 'B0':
        # 获取上游路口 A0 的实例
        a0_ts = traffic_signal.env.traffic_signals.get('A0')
        if a0_ts and hasattr(a0_ts, 'my_green_start'):
            # 计算 A0 路口当前的绿灯已持续时间
            a0_green_duration = current_step - a0_ts.my_green_start

            # 扩大时空窗口，包容 delta_time=5 的步长
            # 当 A0 的绿灯亮了 10~16 步时（车流预计到达 B0）
            if 10 <= a0_green_duration <= 16:
                # 严格要求“刚刚”变绿灯才能拿暴击奖励，防止死锁绿灯白嫖
                # 只有 B0 刚好此时变绿，才给予 2.0 的高额奖励
                if just_turned_green:
                    green_wave_bonus += 2.0  # 给予足量的奖励权重对抗排队惩罚


    # 4. 奖励结算
    # base_penalty: [-X, 0], shaping_reward: ~[-0.5, 0.5], green_wave_bonus: [0, 2.0]
    # 汇总所有模块的奖励与惩罚，返回最终 Step Reward
    final_reward = base_penalty + (1.0 * shaping_reward) + green_wave_bonus
    return final_reward
# Map neighbors
NEIGHBOR_MAP = {
    'A0': [None, 'B0'],
    'B0': ['A0', 'C0'],
    'C0': ['B0', None]
}


class CommObservationFunction(DefaultObservationFunction):
    def __init__(self, ts):
        super().__init__(ts)

    def observation_space(self):
        base_space = super().observation_space()
        base_dim = base_space.shape[0]
        new_dim = base_dim + 2

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
                extra_obs.append(0.0)
            else:
                neighbor_ts = self.ts.env.traffic_signals.get(neighbor_id)
                # 安全防御：如果没拿到 neighbor_ts，补 0
                if not neighbor_ts:
                    extra_obs.append(0.0)
                    continue

                main_arterial_queue = 0
                for lane in neighbor_ts.lanes:
                    # 【注意】这里依赖了 net.xml 的车道命名规范！
                    if "top" not in lane and "bottom" not in lane:
                        main_arterial_queue += neighbor_ts.sumo.lane.getLastStepHaltingNumber(lane)

                extra_obs.append(float(main_arterial_queue))

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
    print(" 神经网络真实维度 (带定向主干道通信补丁)：")
    print(f" 状态输入维度 (Observation): {model.policy.observation_space}")
    print(f" 动作输出维度 (Action): {model.policy.action_space}")
    print("=" * 40 + "\n")

    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path=os.path.join(run_save_dir, 'checkpoints'),
        name_prefix='rl_model'
    )

    print("环境就绪！SB3 正在接收剔除噪声后的主干道情报。开始训练...")
    # 由于是 3 个智能体 (concat_vec_envs 会将 1 个 env 乘以 3)，总 step 会消耗得比平时快 3 倍
    model.learn(total_timesteps=300000, callback=checkpoint_callback, tb_log_name=f"run_{run_idx}")

    model.save(os.path.join(run_save_dir, "ppo_marl_model"))
    env.save(os.path.join(run_save_dir, "vec_normalize_marl.pkl"))

    env.close()
    print(f"\n 多智能体训练完成！")
