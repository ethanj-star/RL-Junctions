import os
import random
import torch
import numpy as np
import math  # 【新增】支持高斯平滑奖金的计算
import supersuit as ss
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, VecEnvWrapper
from sumo_rl import parallel_env
from gymnasium import spaces

# 导入底层依赖 (保持与训练时绝对一致)
from generate_Random_Traffic import generate_route_file
from sumo_rl.environment.observations import DefaultObservationFunction

# ==========================================
# 1. 核心路径动态获取与配置
# ==========================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# 兼容脚本放在 02_MARL 或根目录下的情况
ROOT_DIR = CURRENT_DIR if "02_MARL" not in CURRENT_DIR else os.path.dirname(CURRENT_DIR)

net_path = os.path.join(ROOT_DIR, 'SUMOroutes.net.xml')
route_path = os.path.join(ROOT_DIR, 'traffic.random.rou.xml')

MAIN_GREEN_STATE = os.environ.get("JUC_MAIN_GREEN_STATE", "rrrrGGggrrrrGGgg")
SIDE_GREEN_STATE = os.environ.get("JUC_SIDE_GREEN_STATE", "GGggrrrrGGggrrrr")


def is_main_green_state(ts):
    state = ts.sumo.trafficlight.getRedYellowGreenState(ts.id)
    return state == MAIN_GREEN_STATE

# ！！！！！每次更改：指定您要测试哪一次训练的模型 ！！！！！
RUN_IDX = 8  # 确保这里是您刚刚跑完绿波优化的那个编号
RUN_DIR = os.path.join(ROOT_DIR, 'saved_models', f'marl_run_{RUN_IDX}')
LOG_DIR = os.path.join(ROOT_DIR, 'logs', f'marl_run_{RUN_IDX}')

# 默认加载训练结束时保存的最终模型与归一化参数
MODEL_PATH = os.path.join(RUN_DIR, 'ppo_marl_model.zip')
VEC_NORM_PATH = os.path.join(RUN_DIR, 'vec_normalize_marl.pkl')

# ==========================================
# 2. 核心架构复刻 (必须与训练环境 1:1 还原)
# ==========================================

# 2.1 拓扑字典与自定义通信观测器
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
                neighbor_ts = self.ts.env.traffic_signals[neighbor_id]
                main_arterial_queue = 0
                for lane in neighbor_ts.lanes:
                    if "top" not in lane and "bottom" not in lane:
                        main_arterial_queue += neighbor_ts.sumo.lane.getLastStepHaltingNumber(lane)
                extra_obs.append(float(main_arterial_queue))

        final_obs = np.concatenate([base_obs, extra_obs])
        return np.array(final_obs, dtype=np.float32)


# 2.2 【替换】完全体绿波奖励函数 (必须与训练时使用的函数一模一样)
def custom_green_wave_reward(traffic_signal):
    """
    终极融合奖励函数：无悖论 PBRS (基于势能的奖励重塑) + 绿波动能时空接力奖励。
    该函数旨在让智能体在保证局部路口不拥堵的前提下，自主学会相邻路口间的相位差协同（绿波带）。
    """
    my_id = traffic_signal.id
    current_step = getattr(traffic_signal.env, "sim_step", 0)

    def is_main_green(ts):
        return is_main_green_state(ts)

    # ==========================================
    # 1. 独立时间戳维护模块 (解决多智能体异步竞态条件)
    # ==========================================
    if is_main_green(traffic_signal):
        if not hasattr(traffic_signal, 'my_green_start'):
            traffic_signal.my_green_start = current_step
    else:
        if hasattr(traffic_signal, 'my_green_start'):
            delattr(traffic_signal, 'my_green_start')

    # ==========================================
    # 2. 纯粹 PBRS 计算模块 (排队疏散的基础保底机制)
    # ==========================================
    total_queue = traffic_signal.get_total_queued()
    base_reward = -total_queue / 100.0
    phi_current = -total_queue / 100.0

    delta_time = getattr(traffic_signal.env, "delta_time", 5)
    is_new_episode = current_step <= delta_time

    if not hasattr(traffic_signal, 'last_potential') or is_new_episode:
        traffic_signal.last_potential = phi_current
        shaping_reward = 0.0
    else:
        gamma = 0.99
        shaping_reward = (gamma * phi_current) - traffic_signal.last_potential
        traffic_signal.last_potential = phi_current

    pbrs_score = base_reward + shaping_reward

    # ==========================================
    # 3. 绿波联动奖金模块 (时空协同引擎)
    # ==========================================
    green_wave_bonus = 0.0

    if my_id in ['A0', 'C0']:
        a0_ts = traffic_signal.env.traffic_signals.get('A0')
        c0_ts = traffic_signal.env.traffic_signals.get('C0')
        if a0_ts and c0_ts and is_main_green(a0_ts) and is_main_green(c0_ts):
            green_wave_bonus += 0.05

    elif my_id == 'B0':
        a0_ts = traffic_signal.env.traffic_signals.get('A0')
        if a0_ts and hasattr(a0_ts, 'my_green_start'):
            a0_green_duration = current_step - a0_ts.my_green_start

            if 10 <= a0_green_duration <= 15 and is_main_green(traffic_signal):
                green_wave_bonus += 0.5

    # ==========================================
    # 4. 奖励结算
    # ==========================================
    final_reward = pbrs_score + green_wave_bonus
    return final_reward


# 2.3 SB3 API 补丁
class SB3CompatibilityWrapper(VecEnvWrapper):
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


# ==========================================
# 3. 主测试运行函数
# ==========================================
def run_marl_test():
    seed = 8848
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    print(f"\n 正在加载第 {RUN_IDX} 次 MARL 训练的完全体环境和模型...")

    # 先生成一次兜底文件
    generate_route_file()

    # 创建带有 GUI、通信观测器、绿波奖励 和 防死锁约束的平行环境
    env = parallel_env(
        net_file=net_path,
        route_file=route_path,
        out_csv_name=os.path.join(LOG_DIR, 'test_marl_output'),
        use_gui=True,  # 开启可视化界面观赏绿波
        num_seconds=3600,
        reward_fn=custom_green_wave_reward,        # 【修改】使用与训练相同的绿波奖励函数
        observation_class=CommObservationFunction, # 必须使用与训练一致的观测空间
        min_green=10,                              # 【修改】必须与训练保持一致，防闪烁
        max_green=50                               # 【修改】必须与训练保持一致，防死锁
    )

    # 挂载底层 reset 猴子补丁 (确保测试依然面对全新随机车流)
    original_reset = env.reset

    def custom_reset(*args, **kwargs):
        print("\n [完全体通信版] 测试环境重置：正在生成全新泊松随机交通流考卷...")
        generate_route_file()
        return original_reset(*args, **kwargs)

    env.reset = custom_reset

    # 空间转换映射
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, num_vec_envs=1, num_cpus=1, base_class='stable_baselines3')
    env = SB3CompatibilityWrapper(env)

    # 4. 加载训练时的环境感知状态 (VecNormalize)
    if not os.path.exists(VEC_NORM_PATH):
        print(f" 找不到归一化文件: {VEC_NORM_PATH}，AI 将无法理解环境！")
        return

    env = VecNormalize.load(VEC_NORM_PATH, env)
    env.training = False  # 停止更新环境统计特征
    env.norm_reward = False  # 测试时看真实奖励的绝对值

    # 5. 加载 PPO 大脑
    if not os.path.exists(MODEL_PATH):
        print(f" 找不到模型文件: {MODEL_PATH}，请检查 RUN_IDX 编号或文件路径。")
        return

    model = PPO.load(MODEL_PATH)
    print(" 模型(带 13 维主干道通信感知) 和 归一化参数加载成功！开始仿真测试...")

    # 6. 运行仿真并渲染
    obs = env.reset()
    step = 0
    total_reward = 0.0

    while True:
        # deterministic=True：严格执行策略，不乱探索
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = env.step(action)

        total_reward += np.sum(rewards)
        step += 1

        if np.any(dones):
            break

    print(f"\n 测试结束！")
    print(f"总共运行控制步数: {step}")
    print(f"3个路口总累计 PBRS 奖励: {total_reward:.2f}")

    env.close()

if __name__ == "__main__":
    run_marl_test()
