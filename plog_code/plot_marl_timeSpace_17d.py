import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import traci
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecMonitor, VecNormalize, VecEnvWrapper
from sumo_rl import parallel_env
import supersuit as ss
from gymnasium import spaces
from sumo_rl.environment.observations import DefaultObservationFunction

# ==========================================
# 1. 核心路径与环境配置
# ==========================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# 强制向上一级，获取 3JucRL 根目录
ROOT_DIR = os.path.dirname(CURRENT_DIR)

net_path = os.path.join(ROOT_DIR, 'SUMOroutes.net.xml')
route_path = os.path.join(ROOT_DIR, 'traffic.random.rou.xml')

# ==========================================
# 动态指定测试模型编号
# ==========================================
# ！！！！！每次更改：指定您要测试哪一次训练的模型 ！！！！！
RUN_IDX = 29  # 匹配您带有前馈雷达的最新模型
# 动态拼接当前选定模型所在的文件夹路径
RUN_DIR = os.path.join(ROOT_DIR, 'saved_models', f'marl_run_{RUN_IDX}')
# 指定您的最佳模型路径
MODEL_PATH = os.path.join(RUN_DIR, 'ppo_marl_model.zip')
# VecNormalize 对应的存档（保持观测空间缩放一致）
VEC_NORM_PATH = os.path.join(RUN_DIR, 'vec_normalize_marl.pkl')


# ==========================================
# 2. 依赖类复用 (必须与训练时完全一致 100% 还原)
# ==========================================
NEIGHBOR_MAP = {'A0': [None, 'B0'], 'B0': ['A0', 'C0'], 'C0': ['B0', None]}

def custom_green_wave_reward(traffic_signal):
    """
    【必须保留】：虽然测试时不看分数，但需要依靠它来驱动 my_green_start 计时器！
    """
    my_id = traffic_signal.id
    current_step = getattr(traffic_signal.env, "sim_step", 0)
    delta_time = getattr(traffic_signal.env, "delta_time", 5)
    is_new_episode = current_step <= delta_time

    if is_new_episode:
        if hasattr(traffic_signal, 'my_green_start'):
            delattr(traffic_signal, 'my_green_start')
        if hasattr(traffic_signal, 'last_phase'):
            delattr(traffic_signal, 'last_phase')

    current_phase = traffic_signal.sumo.trafficlight.getPhase(traffic_signal.id)
    is_main_green = (current_phase == 0)

    if not hasattr(traffic_signal, 'last_phase'):
        traffic_signal.last_phase = current_phase
    just_turned_green = (is_main_green and traffic_signal.last_phase != 0)
    traffic_signal.last_phase = current_phase

    # 核心：时间戳维护
    if is_main_green:
        if not hasattr(traffic_signal, 'my_green_start'):
            traffic_signal.my_green_start = current_step
    else:
        if hasattr(traffic_signal, 'my_green_start'):
            delattr(traffic_signal, 'my_green_start')

    # 为了让代码能跑通，下面的简化返回值即可，测试时并不消耗它
    return 0.0


class CommObservationFunction(DefaultObservationFunction):
    """
    【升级版 6 维雷达】：必须与训练时的空间结构分毫不差
    """
    def __init__(self, ts):
        super().__init__(ts)

    def observation_space(self):
        base_space = super().observation_space()
        base_dim = base_space.shape[0]
        new_dim = base_dim + 6  # 3维度 * 2个邻居
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

                # 1. 软归一化排队数
                main_arterial_queue = 0
                for lane in neighbor_ts.lanes:
                    if "top" not in lane and "bottom" not in lane:
                        main_arterial_queue += neighbor_ts.sumo.lane.getLastStepHaltingNumber(lane)
                queue_norm = min(main_arterial_queue / 50.0, 1.0)

                # 2. 绿灯状态
                current_phase = neighbor_ts.sumo.trafficlight.getPhase(neighbor_id)
                is_main_green = 1.0 if current_phase == 2 else 0.0

                # 3. 绿灯已亮秒数
                green_duration_norm = 0.0
                if is_main_green == 1.0 and hasattr(neighbor_ts, 'my_green_start'):
                    current_step = getattr(self.ts.env, "sim_step", 0)
                    green_duration_norm = min((current_step - neighbor_ts.my_green_start) / 60.0, 1.0)

                extra_obs.extend([queue_norm, is_main_green, green_duration_norm])

        final_obs = np.concatenate([base_obs, extra_obs])
        return np.array(final_obs, dtype=np.float32)


class SB3CompatibilityWrapper(VecEnvWrapper):
    def reset(self):
        obs = self.venv.reset()
        return obs[0] if isinstance(obs, tuple) and len(obs) == 2 else obs

    def step_async(self, actions):
        self.venv.step_async(actions)

    def step_wait(self):
        results = self.venv.step_wait()
        if len(results) == 5:
            obs, rews, terms, truncs, infos = results
            return obs, rews, np.logical_or(terms, truncs), infos
        return results


# ==========================================
# 3. 运行测试并进行微观探针雷达收割
# ==========================================
def run_test_and_harvest_data():
    print(f" 正在加载绿波评估模型...\n {MODEL_PATH}")

    env = parallel_env(
        net_file=net_path,
        route_file=route_path,
        out_csv_name=None,
        use_gui=False,
        num_seconds=3600,
        reward_fn=custom_green_wave_reward, # 【核心修复】开启奖励函数引擎，驱动时间戳计算
        observation_class=CommObservationFunction,
        min_green=5, # 保持底层动作空间严格对齐
        max_green=60
    )

    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, num_vec_envs=1, num_cpus=1, base_class='stable_baselines3')
    env = SB3CompatibilityWrapper(env)

    if os.path.exists(VEC_NORM_PATH):
        env = VecNormalize.load(VEC_NORM_PATH, env)
        env.training = False
        env.norm_reward = False
    else:
        print(" 警告：未找到 VecNormalize 文件！")

    model = PPO.load(MODEL_PATH)
    obs = env.reset()

    trajectory_data = []
    signal_states = {'A0': [], 'B0': [], 'C0': []}

    print(" 探针已挂载，开始基于真实物理时间扫描...")

    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, dones, _ = env.step(action)

        current_sim_time = traci.simulation.getTime()

        for ts_id in ['A0', 'B0', 'C0']:
            current_phase = traci.trafficlight.getPhase(ts_id)
            is_green = 1 if current_phase == 0 else 0
            signal_states[ts_id].append({"time": current_sim_time, "green": is_green})

        veh_ids = traci.vehicle.getIDList()
        for v_id in veh_ids:
            if "WE_MAIN_straight" in v_id:
                x_pos = traci.vehicle.getPosition(v_id)[0]
                trajectory_data.append({
                    "time": current_sim_time,
                    "veh_id": v_id,
                    "position": x_pos
                })

        if np.any(dones):
            print(f" 达到仿真终点 ({current_sim_time} 秒)，探针安全撤离。")
            break

    env.close()
    return pd.DataFrame(trajectory_data), signal_states


# ==========================================
# 4. 渲染顶级时空图 (Time-Space Diagram)
# ==========================================
def plot_green_wave(df_traj, signal_states, time_start, time_end, file_suffix, title_desc):
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False

    fig, ax = plt.subplots(figsize=(16, 8))

    ax.set_xlim(time_start, time_end)
    ax.set_ylim(0, 650)

    # A. 渲染信号灯色带背景
    junctions = {'A0': 100, 'B0': 300, 'C0': 500}
    band_width = 15

    for ts_id, y_pos in junctions.items():
        df_sig = pd.DataFrame(signal_states[ts_id])
        df_sig = df_sig[(df_sig['time'] >= time_start) & (df_sig['time'] <= time_end)]

        current_color = None
        start_t = time_start
        for _, row in df_sig.iterrows():
            t = row['time']
            color = '#2ecc71' if row['green'] == 1 else '#e74c3c'

            if current_color is None:
                current_color = color
                start_t = t
            elif color != current_color:
                rect = patches.Rectangle((start_t, y_pos - band_width / 2), t - start_t, band_width,
                                         linewidth=0, facecolor=current_color, alpha=0.6)
                ax.add_patch(rect)
                current_color = color
                start_t = t

        rect = patches.Rectangle((start_t, y_pos - band_width / 2), time_end - start_t, band_width,
                                 linewidth=0, facecolor=current_color, alpha=0.6)
        ax.add_patch(rect)

        ax.axhline(y=y_pos, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        ax.text(time_start + 5, y_pos + 10, f"Intersection {ts_id}", color='black', fontweight='bold')

    # B. 渲染车辆轨迹线
    df_window = df_traj[(df_traj['time'] >= time_start) & (df_traj['time'] <= time_end)]
    grouped = df_window.groupby('veh_id')

    for veh_id, group in grouped:
        if len(group) > 5:
            ax.plot(group['time'], group['position'], color='black', alpha=0.4, linewidth=1.2)

    # C. 图表修饰
    ax.set_title(f"Time-Space Diagram ({title_desc}) - Run {RUN_IDX}", fontsize=18, fontweight='bold', pad=20)
    ax.set_xlabel("Simulation Time (seconds)", fontsize=14)
    ax.set_ylabel("Absolute Position / Eastbound (meters)", fontsize=14)
    ax.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    # 动态保存路径，绑定 Run 编号
    save_path = os.path.join(ROOT_DIR, f"marl_run_{RUN_IDX}_green_wave_tsd_{file_suffix}.png")
    plt.savefig(save_path, dpi=300)
    print(f" [{title_desc}] 时空图渲染成功，已保存至: {save_path}")

    plt.close()


if __name__ == '__main__':
    traj_data, sig_data = run_test_and_harvest_data()

    print("\n 开始批量生成时空图切片...")

    plot_green_wave(
        df_traj=traj_data,
        signal_states=sig_data,
        time_start=400,
        time_end=800,
        file_suffix="offpeak_400_800",
        title_desc="Off-Peak 400s-800s"
    )

    plot_green_wave(
        df_traj=traj_data,
        signal_states=sig_data,
        time_start=1600,
        time_end=2000,
        file_suffix="peak_1600_2000",
        title_desc="Peak 1600s-2000s"
    )

    print("\n 绘图流水线执行完毕！可以打开图片验收您的绿波成果了！")