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
    终极融合奖励函数 (自适应微调版):
    加权 PBRS + 动态时空窗口 + 绿波维持吞吐量奖励
    """
    my_id = traffic_signal.id
    current_step = getattr(traffic_signal.env, "sim_step", 0)
    delta_time = getattr(traffic_signal.env, "delta_time", 5)
    is_new_episode = current_step <= delta_time

    if is_new_episode:
        if hasattr(traffic_signal, 'my_green_start'):
            delattr(traffic_signal, 'my_green_start')
        if hasattr(traffic_signal, 'my_green_start_queue'):
            delattr(traffic_signal, 'my_green_start_queue')
        if hasattr(traffic_signal, 'last_phase'):
            delattr(traffic_signal, 'last_phase')

    current_phase = traffic_signal.sumo.trafficlight.getPhase(traffic_signal.id)
    is_main_green = (current_phase == 0)

    if not hasattr(traffic_signal, 'last_phase'):
        traffic_signal.last_phase = current_phase
    just_turned_green = (is_main_green and traffic_signal.last_phase != 0)
    traffic_signal.last_phase = current_phase

    # ==========================================
    # 1. 独立时间戳与“发车快照”维护模块
    # ==========================================
    if is_main_green:
        if not hasattr(traffic_signal, 'my_green_start'):
            traffic_signal.my_green_start = current_step

            # 【核心微调 2 前置】：拍快照！记录刚变绿那一瞬间主路的排队车数
            # 这代表了即将释放的上游车队规模
            main_q = 0
            for lane in traffic_signal.lanes:
                if "top" not in lane and "bottom" not in lane:
                    main_q += traffic_signal.sumo.lane.getLastStepHaltingNumber(lane)
            traffic_signal.my_green_start_queue = main_q
    else:
        if hasattr(traffic_signal, 'my_green_start'):
            delattr(traffic_signal, 'my_green_start')
        if hasattr(traffic_signal, 'my_green_start_queue'):
            delattr(traffic_signal, 'my_green_start_queue')

    # ==========================================
    # 2. 纯粹 PBRS 计算模块 (【核心微调 3】：主次权重倾斜)
    # ==========================================
    weighted_queue = 0.0
    for lane in traffic_signal.lanes:
        halting = traffic_signal.sumo.lane.getLastStepHaltingNumber(lane)
        if "top" not in lane and "bottom" not in lane:
            # 主干道排队痛感增加，权重 1.5
            weighted_queue += halting * 1.5
        else:
            # 支路保持正常痛感 1.0
            weighted_queue += halting * 1.0

    base_penalty = -weighted_queue / 100.0
    phi_current = -weighted_queue / 100.0

    if getattr(traffic_signal, 'last_potential', None) is None or is_new_episode:
        traffic_signal.last_potential = phi_current
        shaping_reward = 0.0
    else:
        gamma = 0.99
        shaping_reward = (gamma * phi_current) - traffic_signal.last_potential
        traffic_signal.last_potential = phi_current

    # ==========================================
    # 3. 绿波联动奖金模块 (【核心微调 1 & 2】：动态窗口与维持奖金)
    # ==========================================
    green_wave_bonus = 0.0

    # 动态寻找上游路口
    upstream_id = None
    if my_id == 'B0':
        upstream_id = 'A0'
    elif my_id == 'C0':
        upstream_id = 'B0'

    if upstream_id:
        upstream_ts = traffic_signal.env.traffic_signals.get(upstream_id)
        if upstream_ts and hasattr(upstream_ts, 'my_green_start'):
            upstream_green_duration = current_step - upstream_ts.my_green_start

            # 提取上游“发车快照”中的车队规模
            upstream_q = getattr(upstream_ts, 'my_green_start_queue', 0)

            # 【核心微调 2】：动态时空窗口
            # 基础窗口为 10-14 秒。上游积压的车每多 3 辆，尾车到达的时间就越晚，窗口向后延展 1 秒（最多延展 10 秒）
            window_start = 10
            window_end = 14 + min(int(upstream_q / 3.0), 10)

            if window_start <= upstream_green_duration <= window_end:
                if just_turned_green:
                    green_wave_bonus += 2.0  # 接力暴击奖金

    # 【核心微调 1】：绿波维持（吞吐量）奖金
    maintenance_bonus = 0.0
    if is_main_green:
        for lane in traffic_signal.lanes:
            if "top" not in lane and "bottom" not in lane:
                # 获取总车数和静止车数
                veh_num = traffic_signal.sumo.lane.getLastStepVehicleNumber(lane)
                halting_num = traffic_signal.sumo.lane.getLastStepHaltingNumber(lane)
                # 计算正在移动的车数 (吞吐量)
                moving_num = veh_num - halting_num

                if moving_num > 0:
                    # 每通过一辆移动的车辆给予 0.1 的持续正向反馈
                    maintenance_bonus += moving_num * 0.1

    # ==========================================
    # 3.5 专家先验诱导：AC 共振奖金 (AC Synchronization Bonus)
    # 鼓励外围路口 A0 和 C0 形成同步发车机制，向 B0 施加对称车流
    # ==========================================
    ac_sync_bonus = 0.0
    if my_id in ['A0', 'C0']:
        # 寻找远端的兄弟节点
        peer_id = 'C0' if my_id == 'A0' else 'A0'
        peer_ts = traffic_signal.env.traffic_signals.get(peer_id)

        if peer_ts:
            # 获取远端兄弟当前的真实相位
            peer_phase = peer_ts.sumo.trafficlight.getPhase(peer_id)
            # 如果我和兄弟同时处于主路绿灯，给予共振鼓励！
            if is_main_green and peer_phase == 0:
                # 奖励不宜过大，0.5 即可，作为一种“软引导”
                ac_sync_bonus += 0.5


    # 叠加所有正向奖金
    total_bonus = green_wave_bonus + maintenance_bonus + ac_sync_bonus

    # ==========================================
    # 4. 奖励结算
    # ==========================================
    final_reward = base_penalty + (1.0 * shaping_reward) + total_bonus
    return final_reward


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
                extra_obs.extend([0.0, 0.0, 0.0])
            else:
                neighbor_ts = self.ts.env.traffic_signals.get(neighbor_id)
                if not neighbor_ts:
                    extra_obs.extend([0.0, 0.0, 0.0])
                    continue

                main_arterial_queue = 0
                for lane in neighbor_ts.lanes:
                    if "top" not in lane and "bottom" not in lane:
                        main_arterial_queue += neighbor_ts.sumo.lane.getLastStepHaltingNumber(lane)
                queue_norm = min(main_arterial_queue / 50.0, 1.0)

                current_phase = neighbor_ts.sumo.trafficlight.getPhase(neighbor_id)
                is_main_green = 1.0 if current_phase == 0 else 0.0

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
        min_green=5,
        max_green=60
    )

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
        ent_coef=0.03,
        target_kl=0.05,
        verbose=1,
        device="cpu",
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

    print("环境就绪！SB3 正在接收附带上游灯色前馈信息。开始训练...")
    model.learn(total_timesteps=300000, callback=checkpoint_callback, tb_log_name=f"run_{run_idx}")

    model.save(os.path.join(run_save_dir, "ppo_marl_model"))
    env.save(os.path.join(run_save_dir, "vec_normalize_marl.pkl"))

    env.close()
    print(f"\n 多智能体训练完成！")