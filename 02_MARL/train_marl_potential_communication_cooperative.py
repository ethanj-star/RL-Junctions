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

# 【终极修复：直接导入模块，拒绝 os.system 的静默失败】
from generate_Random_Traffic import generate_route_file

#  新增导入：用于重写观察空间实现通信 rewrite the observation space to implement communication
from sumo_rl.environment.observations import DefaultObservationFunction
from gymnasium import spaces


def linear_schedule_with_min(initial_value: float, min_value: float) -> Callable[[float], float]:
    """
    带保底机制的线性衰减学习率生成器。
    :param initial_value: 初始最大学习率 (例如 3e-4)
    :param min_value: 最低保底学习率 (例如 3e-5)
    :return: 返回一个根据剩余进度计算当前学习率的函数
    """

    def func(progress_remaining: float) -> float:
        """
        progress_remaining 的值会从 1.0 (训练开始) 线性下降到 0.0 (训练结束)
        """
        # 数学映射：当进度为 1 时，结果是 initial_value；当进度为 0 时，结果是 min_value
        return min_value + progress_remaining * (initial_value - min_value)

    return func


# 路径动态获取 (Dynamic path acquisition)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CURRENT_DIR)
net_path = os.path.join(ROOT_DIR, 'SUMOroutes.net.xml')
# 换交通流 对接 8 向泊松随机车流
route_path = os.path.join(ROOT_DIR, 'traffic.random.rou.xml')


# 自动编号：寻找下一个可用的 Run 编号 (Auto-numbering: Find the next available Run number)
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


# 补丁：解决 5 个返回值与 4 个返回值的 API 冲突 (恢复至最干净版本，防冲突)
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


# 新增：带空间合作机制的基于潜力奖励塑形 (Cooperative Potential-Based Reward)
def pbrs_reward(traffic_signal):
    # ==========================================
    # 🌟 核心革新：从“自私”走向“利他”
    # ==========================================
    # 1. 计算自身的排队惩罚
    own_queue = traffic_signal.get_total_queued()

    # 2. 计算邻居的排队惩罚 (空间合作机制)
    neighbor_queue = 0
    my_id = traffic_signal.id
    neighbors = NEIGHBOR_MAP.get(my_id, [None, None])

    # 合作系数 alpha (0.5 代表把邻居一半的痛苦当做自己的痛苦)
    # 这个值如果在 0.1~0.3，偏向利己；如果在 0.5~1.0，高度利他。
    alpha = 0.2

    for neighbor_id in neighbors:
        if neighbor_id is not None:
            # 拿到邻居路口的实例对象
            neighbor_ts = traffic_signal.env.traffic_signals[neighbor_id]
            # 累加邻居的排队长度
            neighbor_queue += neighbor_ts.get_total_queued()

    # 计算全新的合作型基础奖励！
    base_reward = - (own_queue + alpha * neighbor_queue)
    # ==========================================

    # 3. 获取信号灯状态计算势能
    # (注意：势能计算依然只看自己，因为自己的红绿灯只能直接决定自己路口的绿灯比例)
    current_light_state = traffic_signal.sumo.trafficlight.getRedYellowGreenState(traffic_signal.id)
    active_phase_queue = 0
    for i, lane in enumerate(traffic_signal.lanes):
        if current_light_state[i] in ('G', 'g', 'y', 'Y'):
            active_phase_queue += traffic_signal.sumo.lane.getLastStepHaltingNumber(lane)

    # 计算排队占比 (分母使用自己的排队数)
    phi_current = active_phase_queue / (own_queue + 1e-6)

    # 4. 提取上一步的势能 (处理跨回合清零)
    current_step = getattr(traffic_signal.env, "sim_step", 0)
    delta_time = getattr(traffic_signal.env, "delta_time", 5)
    is_new_episode = current_step <= delta_time

    if not hasattr(traffic_signal, 'last_potential') or is_new_episode:
        traffic_signal.last_potential = phi_current
        shaping_reward = 0.0
    else:
        gamma = 0.99
        shaping_reward = (gamma * phi_current) - traffic_signal.last_potential
        traffic_signal.last_potential = phi_current

    # 5. 组合最终奖励 (组合合作基础奖励与局部势能奖励)
    beta = 100.0
    final_reward = base_reward + (beta * shaping_reward)

    return final_reward / 100.0


#  核心增强：主干道定向通信机制
#   Map neighbors
NEIGHBOR_MAP = {
    'A0': [None, 'B0'],
    'B0': ['A0', 'C0'],
    'C0': ['B0', None]
}


class CommObservationFunction(DefaultObservationFunction):
    """
    带有通信机制的自定义观测器：11 维基础状态 + 2 维邻居主干道排队信息
    (Custom observer with communication mechanism: 11-dimensional base state + 2-dimensional neighboring main arterial queue info)
    """

    def __init__(self, ts):
        super().__init__(ts)
        # 【修改点】：删掉了在这里提前查询维度的代码，避免了 sumo-rl 的初始化顺序 Bug

    def observation_space(self):
        # 【修改点】：把空间维度的计算延迟到这里！
        # computation of spatial dimensions
        # 当外部调用这个函数时，TrafficSignal 初始化完成
        base_space = super().observation_space()
        base_dim = base_space.shape[0]

        # 新增 2 维：左邻居主干道排队，右邻居主干道排队
        # (Added 2 dimensions: Left neighbor main arterial queue, Right neighbor main arterial queue)
        new_dim = base_dim + 2

        return spaces.Box(
            low=np.zeros(new_dim, dtype=np.float32),
            high=np.ones(new_dim, dtype=np.float32) * np.inf,
        )

    def __call__(self):
        # 1. 提取自己的基础数据 (1. Extract own base data)
        base_obs = super().__call__()

        # 2. 查拓扑字典，获取邻居 ID (get neighbor IDs)
        my_id = self.ts.id
        neighbors = NEIGHBOR_MAP.get(my_id, [None, None])

        extra_obs = []
        for neighbor_id in neighbors:
            if neighbor_id is None:
                # 边缘路口：零填充占位 (Edge intersection: Zero-padding placeholder)
                extra_obs.append(0.0)
            else:
                neighbor_ts = self.ts.env.traffic_signals[neighbor_id]

                # 【神级优化】：只计算主干道 (排除带有 top/bottom 的辅路车道)
                # (Only calculate main arterial roads (excluding auxiliary lanes containing 'top'/'bottom'))
                main_arterial_queue = 0
                for lane in neighbor_ts.lanes:
                    if "top" not in lane and "bottom" not in lane:
                        main_arterial_queue += neighbor_ts.sumo.lane.getLastStepHaltingNumber(lane)

                # 塞入极其纯净的主干道拥堵情报 (Insert main arterial congestion info)
                extra_obs.append(float(main_arterial_queue))

        # 3. 拼接生成最终的通信增强状态数组 (generate the final communication-enhanced state array)
        final_obs = np.concatenate([base_obs, extra_obs])
        return np.array(final_obs, dtype=np.float32)


#  主程序 (Main Program)

if __name__ == '__main__':
    print("正在初始化多智能体 SUMO 环境...")  # (Initializing multi-agent SUMO environment...)

    seed = 8848
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    saved_models_base = os.path.join(ROOT_DIR, 'saved_models')
    logs_base = os.path.join(ROOT_DIR, 'logs')
    os.makedirs(saved_models_base, exist_ok=True)
    os.makedirs(logs_base, exist_ok=True)

    run_idx = get_next_run_number(saved_models_base, "marl_run_")
    print(
        f"\n 自动检测到历史记录，本次 MARL 分配为: [ 第 {run_idx} 次运行 ]")  # (\n Historical records auto-detected, this MARL is assigned as: [ Run {run_idx} ])

    run_save_dir = os.path.join(saved_models_base, f'marl_run_{run_idx}')
    os.makedirs(run_save_dir, exist_ok=True)
    run_csv_dir = os.path.join(logs_base, f'marl_run_{run_idx}')
    os.makedirs(run_csv_dir, exist_ok=True)
    csv_base_path = os.path.join(run_csv_dir, 'marl_output')
    tensorboard_log_path = os.path.join(logs_base, 'ppo_marl_tb')

    # 【前置保障】：先生成一次兜底文件，防止首次启动找不到文件报错
    print(" [系统启动] 正在生成初始的泊松随机交通流...")
    generate_route_file()

    # 1. 创建环境 (挂载了纯净的 PBRS 奖励 和 主干道定向通信)
    # Create environment
    env = parallel_env(
        net_file=net_path,
        route_file=route_path,
        out_csv_name=csv_base_path,
        use_gui=False,
        num_seconds=3600,
        reward_fn=pbrs_reward,
        observation_class=CommObservationFunction
    )

    # 核心拦截器：猴子补丁 (a Monkey Patch)
    # 替换底层的 reset 方法，保证交通流生成在 SUMO 读取之前完成！ dynamic change reset function，add new traffic flow .xml
    original_reset = env.reset
    def custom_reset(*args, **kwargs):
        print("\n [完全体通信版] 监听到底层环境重置信号，正在为本局生成全新泊松车流...")
        generate_route_file()
        return original_reset(*args, **kwargs)
    # 将原生 reset 替换为我们的拦截器
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

    # output realtime dimentions
    print("\n" + "=" * 40)
    print(
        " 神经网络真实维度 (带定向主干道通信补丁)：")  # (True neural network dimensions (with directed main arterial communication patch):)
    print(f" 状态输入维度 (Observation): {model.policy.observation_space}")
    print(f" 动作输出维度 (Action): {model.policy.action_space}")
    print("=" * 40 + "\n")

    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path=os.path.join(run_save_dir, 'checkpoints'),
        name_prefix='rl_model'
    )

    print("环境就绪！SB3 正在接收剔除噪声后的主干道情报。开始训练...")
    model.learn(total_timesteps=300000, callback=checkpoint_callback, tb_log_name=f"run_{run_idx}")

    model.save(os.path.join(run_save_dir, "ppo_marl_model"))
    env.save(os.path.join(run_save_dir, "vec_normalize_marl.pkl"))

    env.close()
    print(f"\n 多智能体训练完成！所有产出均已安全保存至专属目录:")
    print(f"模型与断点: {run_save_dir}")
    print(f"日志文件: {run_csv_dir}")