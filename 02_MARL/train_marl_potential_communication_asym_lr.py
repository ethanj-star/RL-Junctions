import os
import random
import torch
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import VecMonitor, VecNormalize, VecEnvWrapper
from sumo_rl import parallel_env
import supersuit as ss
from typing import Callable, Dict, Any, Type, Union
# 重写观察空间实现通信 rewrite the observation space to implement communication
from sumo_rl.environment.observations import DefaultObservationFunction
from gymnasium import spaces
# 终极修复：直接导入模块，拒绝 os.system 的静默失败
from generate_Random_Traffic import generate_route_file
# 非对称学习率 PPO 类 Asymmetric Learning Rate
from typing import Callable, Dict, Any, Type, Union, List


# ==========================================
# 🌟 终极非对称学习率 PPO 类 (Asymmetric PPO)
# ==========================================
class AsymmetricPPO(PPO):
    """
    支持 Actor 和 Critic 独立学习率配比，且完美兼容动态衰减 (LR Schedule) 的 PPO 类
    """

    def __init__(self, *args, critic_lr_multiplier=1.5, **kwargs):
        # 必须先赋值，再调用父类初始化
        self.critic_lr_multiplier = critic_lr_multiplier
        super().__init__(*args, **kwargs)

    def _setup_model(self) -> None:
        super()._setup_model()

        # 将参数分为 Actor 组和 Critic 组
        actor_params = []
        critic_params = []

        for name, param in self.policy.named_parameters():
            if "value_net" in name or "mlp_extractor.value_net" in name:
                critic_params.append(param)
            else:
                actor_params.append(param)

        # 【核心修复 1】：获取真正的初始学习率数值
        # 因为传入的 learning_rate 是个函数，我们要传入 1.0 (进度 100%) 来获取初始值
        if callable(self.lr_schedule):
            initial_lr = self.lr_schedule(1.0)
        else:
            initial_lr = self.learning_rate

        optimizer_class = self.policy.optimizer_class
        optimizer_kwargs = self.policy.optimizer_kwargs.copy()

        # 重新构建优化器，分成两个独立的参数组
        self.policy.optimizer = optimizer_class([
            {"params": actor_params, "lr": initial_lr},
            {"params": critic_params, "lr": initial_lr * self.critic_lr_multiplier}
        ], **optimizer_kwargs)

    # 【核心修复 2】：重写底层的学习率更新函数
    # 彻底拦截 SB3 默认的“强制一刀切”更新行为！
    def _update_learning_rate(self, optimizers: Union[List[torch.optim.Optimizer], torch.optim.Optimizer]) -> None:
        # 计算当前进度下的基础学习率
        current_base_lr = self.lr_schedule(self._current_progress_remaining)

        # 记录基础学习率到 Tensorboard
        self.logger.record("train/learning_rate", current_base_lr)

        # 独立更新两个组的学习率
        # param_groups[0] 是 Actor 组，param_groups[1] 是 Critic 组
        self.policy.optimizer.param_groups[0]["lr"] = current_base_lr
        self.policy.optimizer.param_groups[1]["lr"] = current_base_lr * self.critic_lr_multiplier


# ==========================================

# 其他原有逻辑 (保持参数逻辑对齐)
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
    if not os.path.exists(base_dir): return 1
    existing_runs = [int(f.replace(prefix, "")) for f in os.listdir(base_dir) if f.startswith(prefix)]
    return max(existing_runs) + 1 if existing_runs else 1


# 补丁：解决 5 个返回值与 4 个返回值的 API 冲突 (恢复至最干净版本，防冲突)
# (Patch: Resolve API conflict between 5 return values and 4 return values)
class SB3CompatibilityWrapper(VecEnvWrapper):
    def __init__(self, venv):
        super().__init__(venv)

    def reset(self):
        # 这里的生成逻辑已移交到底层拦截器，避免 SUMO 读取冲突
        obs = self.venv.reset()
        return obs[0] if isinstance(obs, tuple) else obs

    def step_async(self, actions): self.venv.step_async(actions)

    def step_wait(self):
        results = self.venv.step_wait()
        if len(results) == 5:
            obs, rews, terms, truncs, infos = results
            return obs, rews, np.logical_or(terms, truncs), infos
        return results


# 基于潜力的奖励塑形 (Potential-Based Reward)
def pbrs_reward(traffic_signal):
    # 1. 计算基础奖励 (Base Reward: sum up every queued cars)
    total_queue = traffic_signal.get_total_queued()
    base_reward = -total_queue
    # 2. 获取信号灯状态 (2. Get traffic light state)
    current_light_state = traffic_signal.sumo.trafficlight.getRedYellowGreenState(traffic_signal.id)
    active_phase_queue = sum(traffic_signal.sumo.lane.getLastStepHaltingNumber(lane)
                             for i, lane in enumerate(traffic_signal.lanes)
                             if current_light_state[i] in ('G', 'g', 'y', 'Y'))
    # 计算排队占比 (Calculate queue ratio)
    phi_current = active_phase_queue / (total_queue + 1e-6)
    # 3. 提取上一步的势能 (处理跨回合清零)
    # 使用 getattr 防御性获取，并用 <= delta_time 完美捕获第一步
    current_step = getattr(traffic_signal.env, "sim_step", 0)
    delta_time = getattr(traffic_signal.env, "delta_time", 5)
    is_new_episode = current_step <= delta_time

    if not hasattr(traffic_signal, 'last_potential') or is_new_episode:
        # 如果是第一步，让记忆直接等于当前状态（不产生任何差值）
        traffic_signal.last_potential = phi_current
        shaping_reward = 0.0  # 第一步没有状态转移，强行把势能奖金归零
    else:
        # 计算最终的塑形奖励 F ( Calculate the final shaping reward F)
        shaping_reward = (0.99 * phi_current) - traffic_signal.last_potential
        traffic_signal.last_potential = phi_current

    # 5. 组合最终奖励 (5. Combine the final reward)
    return (base_reward + 100.0 * shaping_reward) / 100.0


NEIGHBOR_MAP = {'A0': [None, 'B0'], 'B0': ['A0', 'C0'], 'C0': ['B0', None]}


class CommObservationFunction(DefaultObservationFunction):
    def observation_space(self):
        new_dim = super().observation_space().shape[0] + 2
        return spaces.Box(low=0, high=np.inf, shape=(new_dim,), dtype=np.float32)

    def __call__(self):
        base_obs = super().__call__()
        neighbors = NEIGHBOR_MAP.get(self.ts.id, [None, None])
        extra_obs = []
        for n_id in neighbors:
            if n_id is None:
                extra_obs.append(0.0)
            else:
                n_ts = self.ts.env.traffic_signals[n_id]
                q = sum(n_ts.sumo.lane.getLastStepHaltingNumber(l) for l in n_ts.lanes if
                        "top" not in l and "bottom" not in l)
                extra_obs.append(float(q))
        return np.concatenate([base_obs, extra_obs])

# 主训练流程

if __name__ == '__main__':
    seed = 8848
    random.seed(seed);
    np.random.seed(seed);
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
    print("[系统启动] 正在生成初始的泊松随机交通流...")
    generate_route_file()

    # 🔴【核心修复：把 out_csv_name 参数加回去，否则 SUMO 不会输出数据！】🔴
    env = parallel_env(
        net_file=net_path,
        route_file=route_path,
        out_csv_name=csv_base_path,  # <==== 缺失的就是这句！
        use_gui=False,
        num_seconds=3600,
        reward_fn=pbrs_reward,
        observation_class=CommObservationFunction
    )
    # 核心拦截器：狸猫换太子 (Monkey Patch)
    # 替换底层的 reset 方法，保证交通流生成在 SUMO 读取之前完成！

    original_reset = env.reset

    def custom_reset(*args, **kwargs):
        print("\n [非对称完全体] 监听到底层环境重置信号，正在为本局生成全新泊松车流...")
        generate_route_file()
        return original_reset(*args, **kwargs)

    # 将原生 reset 替换为拦截器
    env.reset = custom_reset

    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, num_vec_envs=1, num_cpus=1, base_class='stable_baselines3')
    env = SB3CompatibilityWrapper(env)
    env = VecMonitor(env)
    env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.)

    # 使用非对称学习率 PPO
    # Actor LR 基准: 3e-4 (随时间衰减)
    # Critic LR 倍率: 1.5 (即初始 Critic LR = 4.5e-4，稳健型评委)
    model = AsymmetricPPO(
        "MlpPolicy", env,
        learning_rate=linear_schedule_with_min(3e-4, 3e-5),
        critic_lr_multiplier=1.5,  # <==== 采用 1.5 倍的黄金非对称配比
        n_steps=2048, batch_size=256, n_epochs=10, ent_coef=0.03, verbose=1,
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

    print(f"\n 开始训练：非对称学习率完全体 (Actor:3e-4, Critic:4.5e-4)")
    model.learn(total_timesteps=300000, callback=checkpoint_callback, tb_log_name=f"run_{run_idx}")

    model.save(os.path.join(run_save_dir, "ppo_marl_model"))
    env.save(os.path.join(run_save_dir, "vec_normalize_marl.pkl"))

    env.close()

    print(f"\n 多智能体训练完成！所有产出均已安全保存至专属目录")
    print(f"模型与断点: {run_save_dir}")
    print(f"日志文件: {run_csv_dir}")