from datetime import datetime

import numpy as np
from environment import CustomEnvironment
from prepare_env import ENV_NAME
from ray.rllib.callbacks.callbacks import RLlibCallback
from pathlib import Path
import imageio.v2 as imageio
from matplotlib.colors import to_rgb
from ray.rllib.env.env_runner import EnvRunner
from ray.rllib.env.multi_agent_env import MultiAgentEnvWrapper

VIDEO_LOG_DIR = (Path("logs") / ENV_NAME / "PPO" / "videos").expanduser()
VIDEO_LOG_DIR.mkdir(exist_ok=True, parents=True)


class VideoCallback(RLlibCallback):
    @staticmethod
    def _board_to_rgb(board: np.ndarray, colors: list, n_players: int) -> np.ndarray:
        # Assuming to_rgb is defined globally or imported
        img = np.zeros((*board.shape, 3), dtype=np.float32)
        for i in range(n_players):
            img[board == i] = to_rgb(colors[i])
        return (img * 255).astype(np.uint8)

    def __init__(self, save_freq=10):
        super().__init__()
        self.logdir = VIDEO_LOG_DIR / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.logdir.mkdir(exist_ok=True, parents=True)
        self.save_freq = save_freq
        self.episode_counter = 0

    def on_episode_start(self, *, episode, env_index, **kwargs):
        # Decide once at the start if this specific episode should be recorded
        record = self.episode_counter % self.save_freq == 0
        episode.user_data["record"] = record
        episode.user_data["frames"] = []
        self.episode_counter += 1

    def on_episode_step(self, *, episode, env, env_index, **kwargs):
        if not episode.user_data.get("record"):
            return

        # 'env' in the callback is typically the BaseEnv/VectorEnv.
        # Use get_sub_environments() to get the list and index it.
        sub_envs = env.get_sub_environments()
        # Access the specific sub-env using the provided env_index
        actual_env = sub_envs[env_index].par_env

        frame = self._board_to_rgb(
            np.array(actual_env.game.board),
            actual_env.game.countryColors,
            actual_env.game.n_players,
        )
        episode.user_data["frames"].append(frame)

    def on_episode_end(self, *, episode, **kwargs):
        if not episode.user_data.get("record"):
            return

        frames = episode.user_data["frames"]
        if frames:
            fname = f"episode_{self.episode_counter}_{episode.id_}.mp4"
            imageio.mimsave(self.logdir / fname, frames, fps=8)
