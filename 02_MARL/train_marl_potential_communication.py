import os
import random
import torch
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import VecMonitor, VecNormalize, VecEnvWrapper
from sumo_rl import parallel_env
import supersuit as ss


#  新增导入：用于重写观察空间实现通信 rewrite the observation space to implement communication

from sumo_rl.environment.observations import DefaultObservationFunction
from gymnasium import spaces
# 路径动态获取 (Dynamic path acquisition)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CURRENT_DIR)
net_path = os.path.join(ROOT_DIR, 'SUMOroutes.net.xml')
route_path = os.path.join(ROOT_DIR, 'traffic.rou.rou.xml')


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


# 补丁：解决 5 个返回值与 4 个返回值的 API 冲突 (Patch: Resolve API conflict between 5 return values and 4 return values)
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


# 新增：基于潜力的奖励塑形 (Potential-Based Reward)
def pbrs_reward(traffic_signal):
    # 1. 计算基础奖励 (Base Reward: sum up every queued cars)
    total_queue = traffic_signal.get_total_queued()
    base_reward = -total_queue

    # 2. 获取信号灯状态 (2. Get traffic light state)
    current_light_state = traffic_signal.sumo.trafficlight.getRedYellowGreenState(traffic_signal.id)
    active_phase_queue = 0
    for i, lane in enumerate(traffic_signal.lanes):
        if current_light_state[i] in ('G', 'g', 'y', 'Y'):
            active_phase_queue += traffic_signal.sumo.lane.getLastStepHaltingNumber(lane)

    # 计算排队占比 (Calculate queue ratio)
    phi_current = active_phase_queue / (total_queue + 1e-6)

    # 3. 提取上一步的势能 (3. Extract the potential from the previous step)
    if not hasattr(traffic_signal, 'last_potential'):
        traffic_signal.last_potential = 0.0

    # 4. 计算最终的塑形奖励 F (4. Calculate the final shaping reward F)
    gamma = 0.99
    shaping_reward = (gamma * phi_current) - traffic_signal.last_potential
    traffic_signal.last_potential = phi_current

    # 5. 组合最终奖励 (5. Combine the final reward)
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
    print("正在初始化多智能体 SUMO 环境...") # (Initializing multi-agent SUMO environment...)

    seed = 8848
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    saved_models_base = os.path.join(ROOT_DIR, 'saved_models')
    logs_base = os.path.join(ROOT_DIR, 'logs')
    os.makedirs(saved_models_base, exist_ok=True)
    os.makedirs(logs_base, exist_ok=True)

    run_idx = get_next_run_number(saved_models_base, "marl_run_")
    print(f"\n 自动检测到历史记录，本次 MARL 分配为: [ 第 {run_idx} 次运行 ]") # (\n Historical records auto-detected, this MARL is assigned as: [ Run {run_idx} ])

    run_save_dir = os.path.join(saved_models_base, f'marl_run_{run_idx}')
    os.makedirs(run_save_dir, exist_ok=True)
    run_csv_dir = os.path.join(logs_base, f'marl_run_{run_idx}')
    os.makedirs(run_csv_dir, exist_ok=True)
    csv_base_path = os.path.join(run_csv_dir, 'marl_output')
    tensorboard_log_path = os.path.join(logs_base, 'ppo_marl_tb')

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
        learning_rate=3e-5,
        n_steps=1024,
        batch_size=256,
        n_epochs=5,
        clip_range=0.2,
        ent_coef=0.01,
        target_kl=0.05,
        verbose=1,
        device="cpu",
        tensorboard_log=tensorboard_log_path
    )

# output realtime dimentions
    print("\n" + "=" * 40)
    print(" 神经网络真实维度 (带定向主干道通信补丁)：") # (True neural network dimensions (with directed main arterial communication patch):)
    print(f" 状态输入维度 (Observation): {model.policy.observation_space}")
    print(f" 动作输出维度 (Action): {model.policy.action_space}")
    print("=" * 40 + "\n")

    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path=os.path.join(run_save_dir, 'checkpoints'),
        name_prefix='rl_model'
    )

    print("环境就绪！SB3 正在接收剔除噪声后的主干道情报。开始训练...")
    model.learn(total_timesteps=500000, callback=checkpoint_callback, tb_log_name=f"run_{run_idx}")

    model.save(os.path.join(run_save_dir, "ppo_marl_model"))
    env.save(os.path.join(run_save_dir, "vec_normalize_marl.pkl"))

    env.close()
    print(f"\n 多智能体训练完成！所有产出均已安全保存至专属目录:")
    print(f"模型与断点: {run_save_dir}")
    print(f"日志文件: {run_csv_dir}")