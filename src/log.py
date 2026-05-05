from datetime import datetime

import numpy as np
from env.custom_environment import CustomEnvironment
from prepare_env import ENV_NAME
from ray.rllib.callbacks.callbacks import RLlibCallback
from pathlib import Path
import imageio.v2 as imageio
from matplotlib.colors import to_rgb
from ray.rllib.env.env_runner import EnvRunner
from ray.rllib.env.multi_agent_env import MultiAgentEnvWrapper
from typing import Optional, Sequence

VIDEO_LOG_DIR = (Path("~/ray_results_new") / ENV_NAME / "PPO" / "videos").expanduser()
VIDEO_LOG_DIR.mkdir(exist_ok=True, parents=True)


class VideoCallback(RLlibCallback):
    @staticmethod
    def _board_to_rgb(
        board: np.ndarray, colors: list[str], n_players: int
    ) -> np.ndarray:
        img = np.zeros((*board.shape, 3), dtype=np.float32)
        for i in range(n_players):
            img[board == i] = to_rgb(colors[i])
        return (img * 255).astype(np.uint8)

    def __init__(
        self, env_runner_indices: Optional[Sequence[int]] = None, save_freq=100
    ):
        self._env_runner_indices = env_runner_indices
        self.logdir = VIDEO_LOG_DIR / str(datetime.now())
        self.save_freq = save_freq
        self.logdir.mkdir()
        self.episode_idx = 0

    def on_episode_start(self, *, episode, **kwargs):
        episode.user_data["frames"] = []

    def on_episode_step(
        self,
        *,
        base_env: MultiAgentEnvWrapper,
        worker: EnvRunner,
        episode,
        **kwargs,
    ):
        self.episode_idx += 1
        if self.episode_idx % self.save_freq != 0:
            return
        base_env: CustomEnvironment = base_env._unwrapped_env.par_env
        frame = self._board_to_rgb(
            np.array(base_env.game.board),
            base_env.game.countryColors,
            base_env.game.n_players,
        )
        episode.user_data["frames"].append(frame)

    def on_episode_end(self, *, worker: EnvRunner, episode, **kwargs):
        if self.episode_idx % self.save_freq != 0:
            return
        frames = episode.user_data["frames"]
        if frames:
            imageio.mimsave(
                self.logdir / f"episode_{self.episode_idx}.mp4", frames, fps=8
            )
