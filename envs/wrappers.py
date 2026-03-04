import gymnasium as gym
import numpy as np
from gymnasium import spaces


class ThreeJunctionCentralizedWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env
        self.agents = ['A0', 'B0', 'C0']

        # 1. 重定义动作空间：从 3 个 Discrete(2) 变成 1 个 MultiDiscrete([2, 2, 2])
        self.action_space = spaces.MultiDiscrete([2, 2, 2])

        # 2. 重定义状态空间：把 3 个 11 维的 Box 拼成 1 个 33 维的 Box
        single_obs_space = env.observation_space
        self.observation_space = spaces.Box(
            low=np.tile(single_obs_space.low, 3),
            high=np.tile(single_obs_space.high, 3),
            shape=(33,),
            dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        # 注意：这里的 self.env.reset() 可能只返回一个值，我们需要处理兼容性
        obs_dict = self.env.reset()

        # 把字典压扁成一维数组
        obs_array = np.concatenate([obs_dict[agent] for agent in self.agents])
        return obs_array, {}

    def step(self, action_array):
        # 1. 将 SB3 传来的数组动作 [0, 1, 0] 拆解成字典
        action_dict = {
            self.agents[0]: action_array[0],
            self.agents[1]: action_array[1],
            self.agents[2]: action_array[2]
        }

        # 2. 传给底层 SUMO 环境
        next_obs_dict, reward_dict, done_dict, info_dict = self.env.step(action_dict)

        # 3. 压扁状态
        next_obs_array = np.concatenate([next_obs_dict[agent] for agent in self.agents])

        # ---------- 修改前 ----------
        # 4. 设计你的专属奖励机制 (Reward Shaping)
        # 这里用最简单的：把三个路口的奖励加起来求和
        # total_reward = sum(reward_dict.values())

        # ---------- 修改后 ----------
        # 4. 设计你的专属奖励机制 (Reward Shaping)
        # 将三个路口的奖励求平均，防止数值过大导致梯度爆炸
        total_reward = sum(reward_dict.values()) / len(self.agents)

        # 5. 处理结束标志 (如果任何一个路口结束，或者整体超时，就结束)
        # 注意 sumo-rl 旧版 done 可能是 dict 或者 bool
        terminated = done_dict['__all__'] if isinstance(done_dict, dict) else done_dict
        truncated = False  # 可以自定义超时逻辑

        return next_obs_array, total_reward, terminated, truncated, info_dict