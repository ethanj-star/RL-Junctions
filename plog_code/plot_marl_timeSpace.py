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
RUN_IDX = 13  # 如果想测第4次的结果，直接改成 4 即可
# 动态拼接当前选定模型所在的文件夹路径
RUN_DIR = os.path.join(ROOT_DIR, 'saved_models', f'marl_run_{RUN_IDX}')
# 指定您的最佳模型路径
MODEL_PATH = os.path.join(RUN_DIR, 'ppo_marl_model.zip')
# VecNormalize 对应的存档（保持观测空间缩放一致）
VEC_NORM_PATH = os.path.join(RUN_DIR, 'vec_normalize_marl.pkl')


# ==========================================
# 2. 依赖类复用 (必须与训练时完全一致，否则模型无法读取)
# ==========================================
NEIGHBOR_MAP = {'A0': [None, 'B0'], 'B0': ['A0', 'C0'], 'C0': ['B0', None]}


class CommObservationFunction(DefaultObservationFunction):
    def observation_space(self):
        base_space = super().observation_space()
        return spaces.Box(
            low=np.zeros(base_space.shape[0] + 2, dtype=np.float32),
            high=np.ones(base_space.shape[0] + 2, dtype=np.float32) * np.inf,
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
                main_arterial_queue = sum(
                    neighbor_ts.sumo.lane.getLastStepHaltingNumber(lane)
                    for lane in neighbor_ts.lanes if "top" not in lane and "bottom" not in lane
                )
                extra_obs.append(float(main_arterial_queue))
        return np.array(np.concatenate([base_obs, extra_obs]), dtype=np.float32)


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
# 3. 运行测试并进行微观探针雷达收割 (终极修复版)
# ==========================================
def run_test_and_harvest_data():
    print(f" 正在加载绿波评估模型...\n {MODEL_PATH}")

    env = parallel_env(
        net_file=net_path,
        route_file=route_path,
        out_csv_name=None,
        use_gui=False,
        num_seconds=3600,
        reward_fn=lambda ts: 0.0,
        observation_class=CommObservationFunction
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

    # 改为无限循环，通过 done 信号安全退出
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, dones, _ = env.step(action)

        # 【核心修复】：获取 SUMO 引擎内部真实的秒数 (例如 5.0s, 10.0s ...)
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
                    "time": current_sim_time,  # 记录真实的物理秒数
                    "veh_id": v_id,
                    "position": x_pos
                })

        # 【核心修复】：一旦 3600 秒结束 (dones=True)，立刻拔掉探针，防止环境自动重置！
        if np.any(dones):
            print(f" 达到仿真终点 ({current_sim_time} 秒)，探针安全撤离。")
            break

    env.close()
    return pd.DataFrame(trajectory_data), signal_states


# ==========================================
# 4. 渲染顶级时空图 (Time-Space Diagram) - 参数化升级版
# ==========================================
def plot_green_wave(df_traj, signal_states, time_start, time_end, file_suffix, title_desc):
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False

    fig, ax = plt.subplots(figsize=(16, 8))

    # 动态应用传入的时间切片范围
    ax.set_xlim(time_start, time_end)
    ax.set_ylim(0, 650)  # X坐标系大约从 -50 到 666

    # A. 渲染信号灯色带背景
    junctions = {'A0': 100, 'B0': 300, 'C0': 500}
    band_width = 15  # 色带在 Y 轴上的显示宽度

    for ts_id, y_pos in junctions.items():
        df_sig = pd.DataFrame(signal_states[ts_id])
        df_sig = df_sig[(df_sig['time'] >= time_start) & (df_sig['time'] <= time_end)]

        current_color = None
        start_t = time_start
        for _, row in df_sig.iterrows():
            t = row['time']
            color = '#2ecc71' if row['green'] == 1 else '#e74c3c'  # 绿 / 红

            if current_color is None:
                current_color = color
                start_t = t
            elif color != current_color:
                rect = patches.Rectangle((start_t, y_pos - band_width / 2), t - start_t, band_width,
                                         linewidth=0, facecolor=current_color, alpha=0.6)
                ax.add_patch(rect)
                current_color = color
                start_t = t

        # 补齐最后一段
        rect = patches.Rectangle((start_t, y_pos - band_width / 2), time_end - start_t, band_width,
                                 linewidth=0, facecolor=current_color, alpha=0.6)
        ax.add_patch(rect)

        # 绘制路口基准辅助线 (文字位置跟随 time_start 动态移动)
        ax.axhline(y=y_pos, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        ax.text(time_start + 5, y_pos + 10, f"Intersection {ts_id}", color='black', fontweight='bold')

    # B. 渲染车辆轨迹线
    df_window = df_traj[(df_traj['time'] >= time_start) & (df_traj['time'] <= time_end)]
    grouped = df_window.groupby('veh_id')

    for veh_id, group in grouped:
        if len(group) > 5:
            ax.plot(group['time'], group['position'], color='black', alpha=0.4, linewidth=1.2)

    # C. 图表修饰 (动态标题)
    ax.set_title(f"Time-Space Diagram ({title_desc}) - Run 3", fontsize=18, fontweight='bold', pad=20)
    ax.set_xlabel("Simulation Time (seconds)", fontsize=14)
    ax.set_ylabel("Absolute Position / Eastbound (meters)", fontsize=14)
    ax.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    # 动态保存路径
    save_path = os.path.join(ROOT_DIR, f"marl_run_3_green_wave_tsd_{file_suffix}.png")
    plt.savefig(save_path, dpi=300)
    print(f" [{title_desc}] 时空图渲染成功，已保存至: {save_path}")

    # 自动关闭当前画布，防止与下一张图重叠
    plt.close()


if __name__ == '__main__':
    # 1. 跑一次 3600秒 的测试，收割全部雷达数据
    traj_data, sig_data = run_test_and_harvest_data()

    print("\n 开始批量生成时空图切片...")

    # 2. 生成【平峰期】图 (400s - 800s)
    plot_green_wave(
        df_traj=traj_data,
        signal_states=sig_data,
        time_start=400,
        time_end=800,
        file_suffix="offpeak_400_800",
        title_desc="Off-Peak 400s-800s"
    )

    # 3. 生成【高峰期】图 (1600s - 2000s)
    plot_green_wave(
        df_traj=traj_data,
        signal_states=sig_data,
        time_start=1600,
        time_end=2000,
        file_suffix="peak_1600_2000",
        title_desc="Peak 1600s-2000s"
    )

    print("\n 绘图流水线执行完毕！")