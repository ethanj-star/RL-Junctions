# 🚦 交通信号灯控制：多智能体强化学习 (MARL)

本项目旨在使用深度强化学习（PPO 算法）优化包含三个相邻路口的交通信号灯网络。采用**多智能体（Multi-Agent Reinforcement Learning, MARL）**架构，每个路口由独立的智能体控制，通过与环境的并行交互来学习协同策略。

本项目基于 `PettingZoo`、`SuperSuit`、`Stable-Baselines3` 和 `sumo-rl` 构建。彻底打通了原生多智能体环境与 SB3 单智能体算法库之间的壁垒，并包含完善的动态测试及防污染绘图模块。

---

## 📂 项目目录与路径结构

系统会自动寻找根目录并进行动态路由，`saved_models` 和 `logs` 会在运行 `train_marl.py` 时自动生成并按次数编号：

```text
3JucRL/ (项目根目录)
├── SUMOroutes.net.xml         # [核心] SUMO 路网拓扑结构文件
├── traffic.rou.rou.xml        # [核心] SUMO 车流生成需求文件
│
├── train_marl.py              # MARL 训练主脚本 (PettingZoo并行、SuperSuit转换、API补丁)
├── test_marl.py               # MARL 评估脚本 (环境架构复原、加载预训练权重、开启GUI)
├── plot_learning_curve_marl.py# MARL 训练曲线绘制脚本 (测试日志过滤、自动解析碎片CSV)
│
├── saved_models/              # [自动生成] 模型存档与参数目录
│   ├── marl_run_1/            # 第 1 次 MARL 实验的专属文件夹
│   │   ├── checkpoints/       # 定时保存的模型断点 (如 rl_model_50000_steps.zip)
│   │   ├── ppo_marl_model.zip # 训练结束后的最终网络权重
│   │   └── vec_normalize_marl.pkl # 状态/奖励归一化统计字典 (测试时极度重要！)
│   └── marl_run_2/            # 第 2 次实验...
│
└── logs/                      # [自动生成] 训练日志与可视化图表目录
    ├── ppo_marl_tb/           # TensorBoard 监控事件大汇总目录
    ├── marl_run_1/            # 第 1 次实验的 CSV 碎片数据及最终图表
    │   ├── marl_output_0_conn0_ep1.csv # 智能体在回合1的原始监控数据
    │   ├── test_marl_output_...csv     # 测试阶段产生的日志 (绘图时会被自动过滤)
    │   └── marl_run_1_learning_curves.png # 由脚本自动生成的精美结果图
    └── marl_run_2/            # 第 2 次实验...

🧠 核心代码架构解析
1. 训练层：train_marl.py (环境翻译与多智能体并行)
负责搭建 PettingZoo 环境，将其伪装成 SB3 认识的形状，并执行 PPO 训练。

原生并行环境：使用 parallel_env 创建 MARL 环境，并指定 reward_fn='queue'。这是因为默认的等待时间差（diff-waiting-time）函数会导致不同智能体之间产生严重的奖励干扰。

SuperSuit 魔法转换：PettingZoo 的 API 并不被 SB3 直接兼容。我们使用了 ss.pettingzoo_env_to_vec_env_v1 和 ss.concat_vec_envs_v1 将多智能体环境强行“翻译”并拼接成了 SB3 原生支持的向量环境形状。

API 世纪冲突补丁：新版 Gymnasium 强制要求 step() 返回 5 个值（分离了 terminated 和 truncated），而 SB3 依然期待 4 个值。我们通过自定义的 SB3CompatibilityWrapper 拦截环境输出，将这两个布尔值合并回统一的 done 信号 (np.logical_or(terms, truncs))。

防崩溃护甲 (VecNormalize)：使用 VecNormalize(norm_obs=True, norm_reward=True, clip_obs=10.) 强行将离谱的路况数据和极端的奖励扣分压缩到均值为 0、方差为 1 的平稳区间。

2. 测试层：test_marl.py (架构复原与断点推断)
用于加载训练好的模型，并在 GUI 中直观观察多个智能体的协同表现。

重构相同的环境漏斗：测试时必须完全复刻训练时的环境包装顺序：parallel_env -> SuperSuit -> SB3CompatibilityWrapper -> VecNormalize。

冻结归一化更新 (核心关键！)：测试前必须加载 vec_normalize_marl.pkl，并设定 env.training = False 和 env.norm_reward = False。告诉 AI 这是考试，不要用测试数据修改自己建立的世界观。

高级断点玩法：代码预留了直接加载特定步数模型（如 rl_model_300000_steps.zip）的接口，方便开发者对比训练初期与后期的策略演变。

3. 评估层：plot_learning_curve_marl.py (防污染聚合绘图)
自动寻址并绘制论文级别的学习曲线，同时采用了与单智能体截然不同的主题色。

防污染过滤机制：在读取海量碎片化 CSV 日志时，加入了 [f for f in csv_files if "test" not in os.path.basename(f)] 机制，精准排除了 test_marl.py 运行时产生的无关数据。

硬核碎文件解析：通过正则表达式 re.search(r'ep(\d+)\.csv', basename) 直接从文件名提取回合数，杜绝了多进程日志交叉导致的步数混乱。

MARL 专属色彩：特意将等待时间曲线设定为红色 (#e74c3c)，排队长度设定为绿色 (#2ecc71)，以便将来与单智能体的图表（紫色/蓝色）进行直观对比。

🚀 快速启动指南
第一步：开始训练
确保终端处于项目根目录：

Bash
python train_marl.py
提示：终端会打印当前分配的 MARL 实验编号。训练中会定期在 checkpoints 文件夹保存断点，最终产出物存放在 saved_models 和 logs 目录下。

第二步：观察测试效果 (GUI)
打开 test_marl.py，将顶部的 RUN_IDX 修改为你刚刚训练完成的编号，然后运行：

Bash
python test_marl.py
提示：SUMO 界面弹出后，调整右上角的 Delay (延时)，点击 "Play" 即可观察多路口红绿灯的动态调度。

第三步：生成训练曲线图
打开 plot_learning_curve_marl.py，确保 RUN_IDX 编号正确，然后运行：

Bash
python plot_learning_curve_marl.py
提示：运行结束后，前往 logs/marl_run_X/ 目录下即可查看精美的 marl_run_X_learning_curves.png 训练曲线图。

💡 终极踩坑备忘录 (For Developer)
Gymnasium vs SB3 返回值冲突：只要底层使用了最新的 sumo-rl 或 PettingZoo，step() 就会返回 5 个值。千万不要去掉 SB3CompatibilityWrapper，否则模型 learn() 的第一步就会直接崩溃。

测试环境的陷阱：如果不使用 SuperSuit 把测试环境也包装成 VecEnv，你将无法加载 VecNormalize，AI 看到的数据将会是一团乱码，表现出极其抽风的行为。

评价指标一致性：train_marl.py 与 test_marl.py 初始化 SUMO 环境时，reward_fn='queue' 参数必须严格对齐，否则测试结果毫无意义。


