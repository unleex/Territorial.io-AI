import numpy as np
import os
from matplotlib.colors import to_rgb
import imageio.v2 as imageio
from environment import CustomEnvironment


def _board_to_rgb(board: np.ndarray, colors: list[str], n_players: int) -> np.ndarray:
    img = np.zeros((*board.shape, 3), dtype=np.float32)
    for i in range(n_players):
        img[board == i] = to_rgb(colors[i])
    return (img * 255).astype(np.uint8)


def evaluate(
    *,
    num_games: int = 20,
    log_folder: str = "logs",
    video_log_folder: str | None = None,
    model,
):
    """
    video_log_folder: if not none, then video will be saved
    """
    if video_log_folder is not None:
        os.makedirs(video_log_folder, exist_ok=True)
    os.makedirs(log_folder, exist_ok=True)
    env = CustomEnvironment()
    episode_returns = []
    writer = None

    for ep in range(num_games):
        if writer is not None:
            writer.close()
        if video_log_folder:
            video_name = f"game {ep}.mp4"
            writer = imageio.get_writer(f"{video_log_folder}/{video_name}", fps=8)

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

            if writer is not None:
                frame = _board_to_rgb(
                    np.array(env.game.board),
                    env.game.countryColors,
                    env.game.n_players,
                )
                writer.append_data(frame)

            # game.render()

            done = all(
                terminations.get(a, False) or truncations.get(a, False)
                for a in env.possible_agents
            )
            if done:
                break

        episode_returns.append(sum(total_rewards.values()))

    if writer is not None:
        writer.close()
    env.close()

    print(f"Mean return: {np.mean(episode_returns):.3f}")
    print(f"Std return:  {np.std(episode_returns):.3f}")
    print(f"Episodes:    {num_games}")
    return episode_returns


if __name__ == "__main__":
    print("\n--- Starting Visual Debugging ---")

    evaluate(
        model=None,  # TODO: load new RLlib model
        num_games=10,
        video_log_folder=None,
    )
