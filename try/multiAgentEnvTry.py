from pettingzoo.classic import tictactoe_v3

env = tictactoe_v3.env(render_mode="human")
env.reset()

for agent in env.agent_iter():
    observation, reward, termination, truncation, info = env.last()
    action = env.action_space(agent).sample() if not (termination or truncation) else None
    env.step(action)
    if all(env.terminations.values()): break