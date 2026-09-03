from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from sumo_rl import SumoEnvironment
import os

from train import LOGS_DIR, MODELS_DIR, NET_FILE, NUM_SECONDS, ROUTE_FILE, RUN_PREFIX, SEED, ThreeJunctionCentralizedWrapper, generate_route_file, pbrs_reward

# RUN_IDX 用于选择要测试的模型编号，USE_GUI 控制是否打开 SUMO 可视化界面。
# RUN_IDX selects the trained model to test, and USE_GUI controls whether the SUMO GUI is opened.
RUN_IDX = int(os.environ.get("JUC_RUN_IDX", "1"))
USE_GUI = os.environ.get("JUC_USE_GUI", "0") == "1"


# 测试环境必须和训练环境使用相同网络、车流和 PBRS reward 设置。
# The test environment must use the same network, route file, and PBRS reward setting as training.
def make_test_env():
    generate_route_file(seed=SEED)
    test_log_dir = LOGS_DIR / f"test_{RUN_PREFIX}{RUN_IDX}"
    test_log_dir.mkdir(parents=True, exist_ok=True)
    raw_env = SumoEnvironment(
        net_file=str(NET_FILE),
        route_file=str(ROUTE_FILE),
        out_csv_name=str(test_log_dir / "output"),
        use_gui=USE_GUI,
        num_seconds=NUM_SECONDS,
        reward_fn=pbrs_reward,
        sumo_seed=SEED,
    )
    return ThreeJunctionCentralizedWrapper(raw_env)


# 加载指定 run 的模型和 VecNormalize 状态，执行一个完整仿真回合。
# This loads the model and VecNormalize state for the selected run, then runs one full simulation episode.
def run_test() -> None:
    run_dir = MODELS_DIR / f"{RUN_PREFIX}{RUN_IDX}"
    model_path = run_dir / "ppo_model.zip"
    vec_path = run_dir / "vec_normalize.pkl"
    if not model_path.exists() or not vec_path.exists():
        print(f"No trained model found for RUN_IDX={RUN_IDX}: {run_dir}")
        return

    env = DummyVecEnv([make_test_env])
    env = VecNormalize.load(str(vec_path), env)
    env.training = False
    env.norm_reward = False
    model = PPO.load(str(model_path))

    obs = env.reset()
    dones = [False]
    total_reward = 0.0
    steps = 0
    while not dones[0]:
        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, _ = env.step(action)
        total_reward += float(rewards[0])
        steps += 1

    env.close()
    print(f"Test finished for RUN_IDX={RUN_IDX}")
    print(f"Steps: {steps}")
    print(f"Total reward: {total_reward:.2f}")


if __name__ == "__main__":
    run_test()
