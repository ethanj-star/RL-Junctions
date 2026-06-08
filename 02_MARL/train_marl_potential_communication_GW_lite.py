"""
轻量级双向绿波训练脚本。
Lightweight bidirectional green-wave training script.

本文件在原始 potential communication 模型的基础上，仅加入少量主路 ETA 到达压力奖励。
Its purpose is to keep the original compact observation design and add only a small ETA-based arterial reward.

实验用途：
1. 验证“少量绿波奖励”是否能在不破坏平均等待时间的情况下改善主路连续通行；
2. 保留 PBRS 排队优化作为主要目标，避免模型只追求时空图直线；
3. 使用完整红绿灯 state 字符串判断主路绿灯，避免误用 phase 编号。
"""

import os
import random
import sys
from contextlib import contextmanager
from typing import Callable

import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import VecEnvWrapper, VecMonitor, VecNormalize
from sumo_rl import parallel_env
from sumo_rl.environment.observations import DefaultObservationFunction
import supersuit as ss

"""
轻量绿波训练脚本：在原 potential communication 模型基础上加入少量主路绿波奖励。
(English: Lightweight green-wave training based on the potential communication model.)

本脚本尽量保持观测空间简洁，用主路 ETA 压力辅助 PPO 学习绿波。
(English: It keeps observations compact and adds ETA-based main-road pressure.)
"""

# 路径配置：自动定位项目根目录、路网文件和随机交通流文件。
# (Path setup: locate project root, SUMO net file, and route file.)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CURRENT_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from generate_Random_Traffic import generate_route_file


net_path = os.path.join(ROOT_DIR, "SUMOroutes.net.xml")
route_path = os.path.join(ROOT_DIR, "traffic.random.rou.xml")

# 训练参数：默认用于轻量绿波实验，可通过环境变量覆盖训练步数。
# (Training settings: defaults for the lightweight green-wave experiment.)
SEED = 8848
TOTAL_TIMESTEPS = int(os.environ.get("JUC_TOTAL_TIMESTEPS", "300000"))

# 信号灯状态：用完整 state 字符串判断主路/支路绿灯，避免依赖 phase 编号。
# (Signal states: use full state strings instead of phase IDs.)
MAIN_GREEN_STATE = os.environ.get("JUC_MAIN_GREEN_STATE", "rrrrGGggrrrrGGgg")
SIDE_GREEN_STATE = os.environ.get("JUC_SIDE_GREEN_STATE", "GGggrrrrGGggrrrr")

# 绿波奖励参数：绿波塑形保持较小，PBRS 仍是主要优化目标。
# (Reward parameters: keep green-wave shaping small; PBRS remains primary.)
GREEN_WAVE_WEIGHT = 0.03
ETA_HORIZON = 20.0
ETA_DECAY = 8.0
MIN_ETA_SPEED = 2.0
MAX_ARRIVAL_PRESSURE = 4.0

# 主路进口边：每个信号灯对应两个双向主路进口。
# (Main-road approaches: two arterial approaches for each signal.)
MAIN_APPROACH_EDGES = {
    "A0": ["left0A0", "B0A0"],
    "B0": ["A0B0", "C0B0"],
    "C0": ["B0C0", "right0C0"],
}

NEIGHBOR_MAP = {
    "A0": [None, "B0"],
    "B0": ["A0", "C0"],
    "C0": ["B0", None],
}


# 文件目录切换工具：随机交通生成函数需要在项目根目录下运行。
# (Directory helper: route generation expects the project root as cwd.)
@contextmanager
def pushd(path: str):
    old_cwd = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old_cwd)


# 随机交通生成：每个 episode 前重新生成交通流，提高泛化性。
# (Traffic generation: regenerate random flows before each episode.)
def generate_route_file_in_root():
    with pushd(ROOT_DIR):
        generate_route_file()


# 学习率调度：训练初期学习率较高，后期保留最低学习率微调。
# (Learning-rate schedule: decay with a non-zero floor.)
def linear_schedule_with_min(initial_value: float, min_value: float) -> Callable[[float], float]:
    def func(progress_remaining: float) -> float:
        return min_value + progress_remaining * (initial_value - min_value)

    return func


# Run 编号工具：自动创建新的 marl_run_xx 输出目录。
# (Run numbering: create the next marl_run_xx output folder.)
def get_next_run_number(base_dir, prefix="marl_run_"):
    if not os.path.exists(base_dir):
        return 1

    existing_runs = []
    for folder in os.listdir(base_dir):
        if folder.startswith(prefix):
            try:
                existing_runs.append(int(folder.replace(prefix, "")))
            except ValueError:
                continue

    return max(existing_runs) + 1 if existing_runs else 1


# 兼容包装器：统一 PettingZoo/Gymnasium/SB3 的 reset 和 step 返回格式。
# (Compatibility wrapper: normalize reset/step API formats.)
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


# 安全调用工具：TraCI 查询异常时返回默认值，避免训练中断。
# (Safe TraCI call: return a default value if a query fails.)
def safe_call(default, func, *args):
    try:
        return func(*args)
    except Exception:
        return default


# 信号状态查询：读取真实灯色字符串，供绿波判断使用。
# (Signal query: read actual red/yellow/green state strings.)
def signal_state(sumo, signal_id: str) -> str:
    return sumo.trafficlight.getRedYellowGreenState(signal_id)


# 主路绿灯判断：所有绿波逻辑统一使用灯色字符串。
# (Main-road check: all green-wave logic uses state strings.)
def is_main_green_state(state: str) -> bool:
    return state == MAIN_GREEN_STATE


# 黄灯判断：黄灯期间不把路口当作稳定主路绿灯处理。
# (Yellow check: exclude transition states from stable green logic.)
def is_yellow_state(state: str) -> bool:
    return "y" in state.lower()


# 基础 PBRS 奖励：沿用原模型的排队塑形奖励。
# (Base PBRS reward: queue-based potential shaping from the original model.)
def pbrs_reward(traffic_signal):
    # 1. 计算基础排队惩罚 (Base queue penalty)
    # get_total_queued() 是 SUMO-RL 对当前路口所有进口道排队车辆的汇总。
    # 队列越长，base_reward 越小，模型会倾向于减少整体排队。
    total_queue = traffic_signal.get_total_queued()
    base_reward = -total_queue

    # 2. 读取真实灯色字符串 (Read real signal-state string)
    # getRedYellowGreenState 返回类似 rrrrGGggrrrrGGgg 的完整灯色，而不是 phase 编号。
    # 后续根据每个 lane 对应位置的灯色，统计当前被放行相位上的排队车辆。
    current_light_state = traffic_signal.sumo.trafficlight.getRedYellowGreenState(traffic_signal.id)
    active_phase_queue = 0

    for i, lane in enumerate(traffic_signal.lanes):
        # i < len(current_light_state) 是防御性判断，避免 lane 数和 state 字符串长度不一致时报错。
        # G/g/y/Y 表示该车道对应信号允许或即将允许通行，因此被纳入 active_phase_queue。
        if i < len(current_light_state) and current_light_state[i] in ("G", "g", "y", "Y"):
            active_phase_queue += traffic_signal.sumo.lane.getLastStepHaltingNumber(lane)

    # 3. 计算势能值 (Potential value)
    # active_phase_queue / total_queue 描述“当前绿灯服务的队列占总队列的比例”。
    # 1e-6 用来避免 total_queue 为 0 时出现除零错误。
    phi_current = active_phase_queue / (total_queue + 1e-6)

    # 4. 判断是否为新回合 (Detect new episode)
    # 每个 episode 开始时要清空上一回合的 last_potential，否则 shaping 会跨回合污染。
    current_step = getattr(traffic_signal.env, "sim_step", 0)
    delta_time = getattr(traffic_signal.env, "delta_time", 5)
    is_new_episode = current_step <= delta_time

    # 5. 势能差奖励 (Potential-based shaping)
    # 第一步没有上一状态，所以只记录 phi_current，不给 shaping 奖励。
    # 后续 step 使用 gamma * 当前势能 - 上一步势能，鼓励“状态正在变好”的动作。
    if not hasattr(traffic_signal, "last_potential") or is_new_episode:
        traffic_signal.last_potential = phi_current
        shaping_reward = 0.0
    else:
        gamma = 0.99
        shaping_reward = (gamma * phi_current) - traffic_signal.last_potential
        traffic_signal.last_potential = phi_current

    # 6. 合成 PBRS 奖励 (Combine base reward and shaping reward)
    # beta 放大 shaping 的影响；最后 /100 是为了把奖励缩放到 PPO 更稳定的数值范围。
    beta = 100.0
    final_reward = base_reward + (beta * shaping_reward)
    return final_reward / 100.0


# 单条主路进口 ETA 压力：车辆越接近停止线，保护主路绿灯的压力越大。
# (ETA pressure: closer arterial vehicles create stronger green-wave pressure.)
def edge_arrival_pressure(sumo, edge_id: str) -> float:
    # 1. 将 edge id 转换成 lane id (edge -> lane)
    # 当前路网主路为单车道，车道名通常是 edge_id + "_0"。
    lane_id = f"{edge_id}_0"
    lane_length = float(safe_call(0.0, sumo.lane.getLength, lane_id))
    vehicle_ids = list(safe_call([], sumo.lane.getLastStepVehicleIDs, lane_id))

    pressure = 0.0
    for veh_id in vehicle_ids:
        # 2. 计算车辆到停止线的 ETA (Estimated Time of Arrival)
        # lane_pos 是车辆在当前车道上的位置，lane_length - lane_pos 近似为距停止线距离。
        # max(speed, MIN_ETA_SPEED) 防止低速或停车车辆导致 ETA 无限大。
        lane_pos = float(safe_call(0.0, sumo.vehicle.getLanePosition, veh_id))
        speed = float(safe_call(0.0, sumo.vehicle.getSpeed, veh_id))
        distance_to_stopline = max(lane_length - lane_pos, 0.0)
        eta = distance_to_stopline / max(speed, MIN_ETA_SPEED)

        if eta <= ETA_HORIZON:
            # 3. ETA 越小，压力越大 (Closer vehicles create stronger pressure)
            # exp(-eta / ETA_DECAY) 是指数衰减，车辆越接近路口，对保持主路绿灯的影响越强。
            pressure += np.exp(-eta / ETA_DECAY)

    # 4. 截断压力值 (Clip pressure)
    # 防止高流量场景下绿波奖励过大，压倒 PBRS 的排队优化目标。
    return float(min(pressure, MAX_ARRIVAL_PRESSURE))


# 轻量绿波奖励：主路来车且主路绿灯时奖励，否则在红灯时惩罚。
# (Lite reward: reward main green for approaching arterial vehicles.)
def lightweight_green_wave_reward(traffic_signal) -> float:
    # 1. 判断当前信号状态 (Check signal state)
    # 这里用完整 state 字符串判断主路绿灯；黄灯作为过渡状态，不强行奖励或惩罚。
    current_state = signal_state(traffic_signal.sumo, traffic_signal.id)
    is_main_green = is_main_green_state(current_state)
    is_yellow = is_yellow_state(current_state)

    # 2. 汇总当前路口两个主路进口方向的 ETA 压力。
    # A0/B0/C0 每个路口都有双向主路来车，因此要把两个方向的 pressure 相加。
    arrival_pressure = 0.0
    for edge_id in MAIN_APPROACH_EDGES.get(traffic_signal.id, []):
        arrival_pressure += edge_arrival_pressure(traffic_signal.sumo, edge_id)
    arrival_pressure = min(arrival_pressure, MAX_ARRIVAL_PRESSURE)

    # 3. 没有主路来车压力时，不额外影响 PBRS。
    # 这样可以避免模型无车时仍然偏向主路绿灯。
    if arrival_pressure <= 0.0:
        return 0.0

    # 4. 有主路来车且当前为主路绿灯，则给小额正奖励。
    if is_main_green:
        return GREEN_WAVE_WEIGHT * arrival_pressure

    # 5. 黄灯属于切换过程，不作为稳定红灯惩罚。
    if is_yellow:
        return 0.0

    # 6. 主路来车但不是主路绿灯，说明可能造成停车，给予轻微惩罚。
    return -0.5 * GREEN_WAVE_WEIGHT * arrival_pressure


# 组合奖励：基础 PBRS + 轻量绿波塑形。
# (Combined reward: PBRS plus lightweight green-wave shaping.)
def pbrs_green_wave_lite_reward(traffic_signal):
    return pbrs_reward(traffic_signal) + lightweight_green_wave_reward(traffic_signal)


# 通信观测：保留基础观测，并附加相邻路口主路排队信息。
# (Communication observation: base observation plus neighboring arterial queues.)
class CommObservationFunction(DefaultObservationFunction):
    """
    Keep the same compact communication structure as train_marl_potential_communication.py:
    base observation + two neighbor main-road queue values.
    """

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
        # 1. 先取得 SUMO-RL 默认观测。
        # 默认观测通常包含当前相位、min_green 标记、各车道密度和排队等基础状态。
        base_obs = super().__call__()
        my_id = self.ts.id
        neighbors = NEIGHBOR_MAP.get(my_id, [None, None])

        # 2. 追加两个邻居路口的主路排队信息。
        # 这是一种轻量通信：不显著增加观测维度，但能让当前路口知道上下游是否积压。
        extra_obs = []
        for neighbor_id in neighbors:
            if neighbor_id is None:
                # 边界路口只有一个邻居，缺失邻居用 0 填充，保证观测维度固定。
                extra_obs.append(0.0)
                continue

            neighbor_ts = self.ts.env.traffic_signals.get(neighbor_id)
            if neighbor_ts is None:
                extra_obs.append(0.0)
                continue

            main_arterial_queue = 0
            for lane in neighbor_ts.lanes:
                # top/bottom 是支路进口；其余车道视为主路相关车道。
                if "top" not in lane and "bottom" not in lane:
                    main_arterial_queue += neighbor_ts.sumo.lane.getLastStepHaltingNumber(lane)

            extra_obs.append(float(main_arterial_queue))

        # 3. 拼接基础观测和通信观测，并转为 float32，符合 Gymnasium observation_space。
        final_obs = np.concatenate([base_obs, extra_obs])
        return np.array(final_obs, dtype=np.float32)


# 主程序：构建环境、训练 PPO，并保存模型与 VecNormalize 统计。
# (Main entry: build env, train PPO, and save model/statistics.)
if __name__ == "__main__":
    print("Initializing MARL PBRS + lightweight green-wave training...")

    # 1. 固定随机种子 (Set random seeds)
    # random / numpy / torch 分别控制 Python、数值计算和神经网络初始化的随机性。
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # 2. 准备模型和日志目录 (Prepare output folders)
    # saved_models 保存 PPO 模型和 VecNormalize；logs 保存 SUMO-RL 每个 episode 的 CSV。
    saved_models_base = os.path.join(ROOT_DIR, "saved_models")
    logs_base = os.path.join(ROOT_DIR, "logs")
    os.makedirs(saved_models_base, exist_ok=True)
    os.makedirs(logs_base, exist_ok=True)

    run_idx = get_next_run_number(saved_models_base, "marl_run_")
    print(f"Detected run index: {run_idx}")

    run_save_dir = os.path.join(saved_models_base, f"marl_run_{run_idx}")
    os.makedirs(run_save_dir, exist_ok=True)
    run_csv_dir = os.path.join(logs_base, f"marl_run_{run_idx}")
    os.makedirs(run_csv_dir, exist_ok=True)

    csv_base_path = os.path.join(run_csv_dir, "marl_output")
    tensorboard_log_path = os.path.join(logs_base, "ppo_marl_tb")

    print("Generating initial random bidirectional traffic...")
    generate_route_file_in_root()

    # 3. 创建 SUMO-RL 多智能体环境 (Create SUMO-RL parallel environment)
    # reward_fn 指向组合奖励；observation_class 指向通信观测类。
    env = parallel_env(
        net_file=net_path,
        route_file=route_path,
        out_csv_name=csv_base_path,
        use_gui=False,
        num_seconds=3600,
        reward_fn=pbrs_green_wave_lite_reward,
        observation_class=CommObservationFunction,
    )

    original_reset = env.reset

    def custom_reset(*args, **kwargs):
        # 每个 episode 开始前重新生成随机交通流，避免模型只记住某一份 route 文件。
        print("Regenerating random bidirectional traffic for this episode...")
        generate_route_file_in_root()
        return original_reset(*args, **kwargs)

    env.reset = custom_reset

    # 4. PettingZoo -> SB3 环境转换 (Convert PettingZoo env to SB3 VecEnv)
    # sumo_rl.parallel_env 是多智能体接口；PPO 需要 SB3 的向量化环境格式。
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(
        env,
        num_vec_envs=1,
        num_cpus=1,
        base_class="stable_baselines3",
    )

    env = SB3CompatibilityWrapper(env)
    env = VecMonitor(env)
    # 5. 归一化观测 (VecNormalize)
    # norm_obs=True 让模型看到稳定尺度的观测；norm_reward=False 保留真实奖励数值，方便分析。
    env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.0)

    # 6. 创建 PPO 模型 (Create PPO model)
    # ent_coef 提高探索强度；target_kl 限制每次策略更新幅度，减少训练震荡。
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
        tensorboard_log=tensorboard_log_path,
    )

    print("=" * 60)
    print("Observation space:", model.policy.observation_space)
    print("Action space:", model.policy.action_space)
    print(f"Main-road green state: {MAIN_GREEN_STATE}")
    print(f"Green-wave weight: {GREEN_WAVE_WEIGHT}")
    print("=" * 60)

    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path=os.path.join(run_save_dir, "checkpoints"),
        name_prefix="rl_model",
    )

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=checkpoint_callback,
        tb_log_name=f"run_{run_idx}_gw_lite",
    )

    # 7. 保存通用文件名，兼容原有测试和绘图脚本。
    # Generic names keep compatibility with existing plotting/testing scripts.
    model.save(os.path.join(run_save_dir, "ppo_marl_model"))
    env.save(os.path.join(run_save_dir, "vec_normalize_marl.pkl"))

    # 8. 保存带实验名称的文件名，方便区分 lite/curriculum/recovery 等不同阶段。
    # Explicit names make this run easy to identify later.
    model.save(os.path.join(run_save_dir, "ppo_marl_model_gw_lite"))
    env.save(os.path.join(run_save_dir, "vec_normalize_marl_gw_lite.pkl"))

    env.close()
    print(f"Training finished. Outputs saved to: {run_save_dir}")
