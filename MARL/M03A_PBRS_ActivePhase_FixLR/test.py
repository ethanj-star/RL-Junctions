import random

import os
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize

from train import LOGS_DIR, MODELS_DIR, RUN_PREFIX, SEED, make_vec_env

# RUN_IDX 用于选择要测试的模型编号，USE_GUI 控制是否打开 SUMO 可视化界面。
# RUN_IDX selects the trained model to test, and USE_GUI controls whether the SUMO GUI is opened.
RUN_IDX = int(os.environ.get("JUC_RUN_IDX", "1"))
USE_GUI = os.environ.get("JUC_USE_GUI", "0") == "1"


# 加载指定 run 的模型和 VecNormalize 状态，执行一个完整仿真回合。
# This loads the model and VecNormalize state for the selected run, then runs one full simulation episode.
def run_test() -> None:
    run_dir = MODELS_DIR / f"{RUN_PREFIX}{RUN_IDX}"
    model_path = run_dir / "ppo_marl_model.zip"
    vec_path = run_dir / "vec_normalize_marl.pkl"
    if not model_path.exists() or not vec_path.exists():
        print(f"No trained model found for RUN_IDX={RUN_IDX}: {run_dir}")
        return

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    test_log_dir = LOGS_DIR / f"test_{RUN_PREFIX}{RUN_IDX}"
    test_log_dir.mkdir(parents=True, exist_ok=True)
    env = make_vec_env(test_log_dir / "test_marl_output", use_gui=USE_GUI)
    env = VecNormalize.load(str(vec_path), env)
    env.training = False
    env.norm_reward = False
    model = PPO.load(str(model_path))

    obs = env.reset()
    total_reward = 0.0
    steps = 0
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, _ = env.step(action)
        total_reward += float(np.sum(rewards))
        steps += 1
        if np.any(dones):
            break

    env.close()
    print(f"Test finished for RUN_IDX={RUN_IDX}")
    print(f"Steps: {steps}")
    print(f"Total reward across agents: {total_reward:.2f}")


if __name__ == "__main__":
    run_test()
