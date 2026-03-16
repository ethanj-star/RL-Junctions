# 🚦 交通信号灯控制：中心化单智能体强化学习 (Baseline)

本项目旨在使用深度强化学习（PPO 算法）优化包含三个相邻路口的交通信号灯网络。采用**中心化单智能体（Centralized Single-Agent）**架构作为多智能体（MARL）研究的 Baseline。

本项目基于 `Stable-Baselines3` 和 `sumo-rl` 构建，包含完善的模型训练、环境封装、动态测试以及抗数据碎片的绘图模块。

---

## 📂 项目目录与路径结构

建议的项目根目录结构如下，其中 `saved_models` 和 `logs` 会在运行 `train.py` 时自动生成并按次数编号：

```text
3JucRL/ (项目根目录)
├── SUMOroutes.net.xml         # [核心] SUMO 路网拓扑结构文件
├── traffic.rou.rou.xml        # [核心] SUMO 车流生成需求文件
│
├── wrappers.py                # Gym 环境封装器，用于状态/动作拼接与奖励整形
├── train.py                   # 模型训练主脚本 (多进程、环境归一化、动态编号)
├── test.py                    # 模型评估脚本 (加载预训练权重并开启 GUI 动画)
├── plot_learning_curve.py     # 训练曲线绘制脚本 (自动解析碎片 CSV 并生成图片)
│
├── saved_models/              # [自动生成] 模型存档与参数目录
│   ├── single_queue_run_1/    # 第 1 次实验的专属权重文件夹
│   │   ├── ppo_model.zip      # 训练好的 PPO 神经网络权重
│   │   └── vec_normalize.pkl  # 状态/奖励归一化统计字典 (测试时极度重要！)
│   └── single_queue_run_2/    # 第 2 次实验...
│
└── logs/                      # [自动生成] 训练日志与可视化图表目录
    ├── ppo_single_tb/         # TensorBoard 监控事件大汇总目录
    ├── single_queue_run_1/    # 第 1 次实验的 CSV 碎片数据及最终图表
    │   ├── output_0_conn0_ep1.csv    # 进程0、回合1的原始路口监控数据
    │   ├── output_1_conn0_ep1.csv    # 进程1、回合1的原始路口监控数据
    │   ├── ...                       # (几百个自动生成的碎片文件)
    │   └── run_1_learning_curves.png # 由 plot_learning_curve 自动生成的精美结果图
    └── single_queue_run_2/    # 第 2 次实验...

🧠 核心代码架构解析
1. 桥梁层：wrappers.py (环境封装与适配)
这是连接原生 SUMO-RL 多智能体环境与 SB3 单智能体算法的核心桥梁。

状态聚合：将原始字典格式的观测值（3 个路口，每个路口 11 维）压扁成一个 33 维的一维数组（Box）。

动作重构：将原始独立的离散动作 Discrete(2) 转换为联合多离散动作空间 MultiDiscrete([2, 2, 2])，使得单个网络能同时输出三个路口的相位。

奖励整形 (Reward Shaping)：将三个路口的独立奖励（排队长度的负数）相加后求平均 (sum(reward_dict.values()) / len(self.agents))。防止总奖励数值过大导致模型梯度爆炸。

2. 训练层：train.py (多进程并行训练)
负责调度环境、初始化 PPO 模型并执行训练。

自动动态归档：自带寻找下一个可用编号的机制。每次运行自动分配 single_queue_run_X，防止手滑覆盖上一次跑了一整夜的模型。

多进程提速：使用 SubprocVecEnv 开启了 4 个并行的 SUMO 仿真环境收集数据。

状态归一化 (VecNormalize)：使用 VecNormalize(norm_obs=True, norm_reward=True, clip_obs=10.) 动态跟踪并归一化观测值和奖励，极大提升了 PPO 收敛的稳定性。

全局随机种子：强制固定了 Python、Numpy、PyTorch 和 SB3 的 Seed (xxx)，确保基线实验完全可复现。

3. 测试层：test.py (可视化仿真评估)
用于加载已训练好的模型，开启 SUMO GUI 进行直观验证。

冻结归一化更新 (核心关键！)：测试时不仅要加载 ppo_model.zip，还必须加载 vec_normalize.pkl。同时必须设置 env.training = False 和 env.norm_reward = False，防止测试环境的交通流破坏训练时建立的归一化标准。

确定性动作：推断时开启 deterministic=True，关闭探索噪声，输出模型认知的最优解。

4. 评估层：plot_learning_curve.py (自动寻址与绘图)
用于解析 SUMO-RL 复杂的多进程日志，并绘制论文级别的学习曲线。

动态寻路：只需修改脚本开头的 RUN_IDX = X，脚本会自动去抓取对应实验批次的数据。

硬核碎文件解析：由于多进程下会生成海量类似 output_0_conn0_ep1.csv 的碎片文件，代码直接使用正则表达式 re.search(r'ep(\d+)\.csv', basename) 精准提取 Episode 编号，彻底杜绝了按 step 切分可能导致的曲线错乱。

平滑与置信区间：应用窗口滑动平均 (rolling)，并利用标准差绘制了数据波动的阴影带（Variance），图片自动归档保存。

🚀 快速启动指南
第一步：开始训练
确保终端处于项目根目录：

Bash
python train.py
提示：终端会打印当前分配的实验编号。结束后，产出物会安全存放在 saved_models 和 logs 目录下。

第二步：观察测试效果 (GUI)
打开 test.py，将顶部的 RUN_IDX 修改为你刚刚训练完成的编号（例如 1），然后运行：

Bash
python test.py
提示：SUMO 界面弹出后，调整右上角的 Delay (延时) 确保肉眼能看清，然后点击 "Play" 播放仿真动画。

第三步：生成训练曲线图
打开 plot_learning_curve.py，确保 RUN_IDX 编号正确，然后运行：

Bash
python plot_learning_curve.py
提示：运行结束后，前往 logs/single_queue_run_X/ 目录下查看高清的 run_X_learning_curves.png 训练曲线图。

💡 终极踩坑备忘录 (For Developer)
Gymnasium 版本断层：如果你打算重写 Wrapper，请牢记最新的 Gymnasium 接口中 reset() 返回 (obs, info) 元组，而 step() 返回 5 个值 (obs, reward, terminated, truncated, info)。旧版代码极易在此处崩溃。

多进程与多离散动作：SB3 中并非所有算法都支持 MultiDiscrete，请坚持使用 PPO 或 A2C，标准 DQN 会报错。

评价指标一致性：train.py 与 test.py 初始化 SUMO 环境时，reward_fn='queue' 参数必须严格对齐，否则测试的基准会被污染，AI 表现会像“没学过”一样。

