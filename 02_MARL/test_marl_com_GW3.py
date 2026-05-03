import os
import random
import torch
import numpy as np
import math  # 支持高斯平滑等计算
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
ROOT_DIR = CURRENT_DIR if "02_MARL" not in CURRENT_DIR else os.path.dirname(CURRENT_DIR)

net_path = os.path.join(ROOT_DIR, 'SUMOroutes.net.xml')
route_path = os.path.join(ROOT_DIR, 'traffic.random.rou.xml')

# ！！！！！每次更改：指定您要测试哪一次训练的模型 ！！！！！
RUN_IDX = 14  # 确保这里是您刚刚跑完 17维雷达优化 的那个编号 (如15)
RUN_DIR = os.path.join(ROOT_DIR, 'saved_models', f'marl_run_{RUN_IDX}')
LOG_DIR = os.path.join(ROOT_DIR, 'logs', f'marl_run_{RUN_IDX}')

# 默认加载训练结束时保存的最终模型与归一化参数
MODEL_PATH = os.path.join(RUN_DIR, 'ppo_marl_model.zip')
VEC_NORM_PATH = os.path.join(RUN_DIR, 'vec_normalize_marl.pkl')

# ==========================================
# 2. 核心架构复刻 (必须与训练环境 1:1 还原)
# ==========================================

# 2.1 拓扑字典与自定义 17维 通信观测器
NEIGHBOR_MAP = {
    'A0': [None, 'B0'],
    'B0': ['A0', 'C0'],
    'C0': ['B0', None]
}


class CommObservationFunction(DefaultObservationFunction):
    """
    全知晓雷达观测器：11 维基础 + 6 维深度前馈情报 (总计 17 维)
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

                # 情报 1：主干道排队数 (经过软归一化，上限设为 50 辆)
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


# 2.2 【替换】完全体绿波奖励函数 (必须与训练时使用的函数一模一样，驱动时间戳)
def custom_green_wave_reward(traffic_signal):
    """
    平衡修复版奖励：强化 PBRS + 削弱动能奖金 + 辅路熔断机制
    (测试时主要依靠它来计算 my_green_start)
    """
    my_id = traffic_signal.id
    current_step = getattr(traffic_signal.env, "sim_step", 0)

    current_phase = traffic_signal.sumo.trafficlight.getPhase(traffic_signal.id)
    is_main_green = (current_phase == 0)

    # 1. 时间戳维护 (供观测器使用)
    if is_main_green:
        if not hasattr(traffic_signal, 'my_green_start'):
            traffic_signal.my_green_start = current_step
    else:
        if hasattr(traffic_signal, 'my_green_start'):
            delattr(traffic_signal, 'my_green_start')

    # 2. PBRS 计算
    total_queue = traffic_signal.get_total_queued()
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

    # 3. 辅路熔断机制
    side_street_queue = 0
    for lane in traffic_signal.lanes:
        if "top" in lane or "bottom" in lane:
            side_street_queue += traffic_signal.sumo.lane.getLastStepHaltingNumber(lane)

    is_melt_down = side_street_queue > 15

    # 4. 动能流率奖励
    kinetic_bonus = 0.0
    if is_main_green and not is_melt_down:
        for lane in traffic_signal.lanes:
            if "top" not in lane and "bottom" not in lane:
                mean_speed = traffic_signal.sumo.lane.getLastStepMeanSpeed(lane)
                veh_num = traffic_signal.sumo.lane.getLastStepVehicleNumber(lane)
                if mean_speed > 7.0 and veh_num > 0:
                    kinetic_bonus += (mean_speed / 13.89) * veh_num * 0.05

    # 5. 最终结算
    meltdown_penalty = -5.0 if is_melt_down else 0.0
    final_reward = pbrs_score + kinetic_bonus + meltdown_penalty
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
        reward_fn=custom_green_wave_reward,  # 使用最新奖励函数
        observation_class=CommObservationFunction,  # 使用 17 维观测空间
        min_green=5,  # 【同步】匹配训练环境的最小绿灯
        max_green=60  # 【同步】匹配训练环境的最大绿灯
    )

    # 挂载底层 reset 猴子补丁
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
    env.training = False
    env.norm_reward = False

    # 5. 加载 PPO 大脑
    if not os.path.exists(MODEL_PATH):
        print(f" 找不到模型文件: {MODEL_PATH}，请检查 RUN_IDX 编号或文件路径。")
        return

    model = PPO.load(MODEL_PATH)
    print(" 模型 (带 17 维主干道雷达感知) 和 归一化参数加载成功！开始仿真测试...")

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