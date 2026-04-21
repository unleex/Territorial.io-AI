from math import e

from gymnasium import make
import numpy as np
from stable_baselines3 import PPO

from utility import find_latest_checkpoint, make_env
from custom_environment.custom_environment_v0 import CustomEnvironment

import numpy as np
from stable_baselines3 import PPO


def evaluate(num_games=20, render_mode: str | None = None):
    env = CustomEnvironment()
    model = PPO.load(find_latest_checkpoint("models", "ppo_v1")[0])

    episode_returns = []

    for ep in range(num_games):
        observations, _ = env.reset(seed=ep)
        done = False
        total_rewards = {agent: 0.0 for agent in env.possible_agents}
        while env.agents:
            actions = {}
            for agent in env.agents:
                obs = observations[agent]
                action, _ = model.predict(obs)
                actions[agent] = action

            observations, rewards, terminations, truncations, _ = env.step(actions)

            for agent in rewards:
                total_rewards[agent] += rewards[agent]

            if render_mode is not None:
                env.render()

            done = all(
                terminations.get(a, False) or truncations.get(a, False)
                for a in env.possible_agents
            )
            if done:
                break

        episode_returns.append(sum(total_rewards.values()))

    env.close()

    print(f"Mean return: {np.mean(episode_returns):.3f}")
    print(f"Std return:  {np.std(episode_returns):.3f}")
    print(f"Episodes:    {num_games}")
    return episode_returns


if __name__ == "__main__":
    # print("\n--- Starting Quantitative Evaluation ---")
    # evaluate(num_episodes=20, render_mode=None)
    print("\n--- Starting Visual Debugging ---")
    evaluate(num_games=10, render_mode="human")
