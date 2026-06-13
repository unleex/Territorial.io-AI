from datetime import datetime
from pathlib import Path
import os

import imageio.v2 as imageio
import numpy as np
from matplotlib.colors import to_rgb

from environment import CustomEnvironment
from ray.rllib.callbacks.callbacks import RLlibCallback
from ray.rllib.env.env_runner import EnvRunner
from game.countryClass import country


RUN_NAME = "scaled_reward"
VIDEO_SAVE_FREQ = 5
VIDEO_LOG_DIR = (
    Path("logs").absolute() / "custom_env" / RUN_NAME / "videos"
).expanduser()
VIDEO_LOG_DIR.mkdir(exist_ok=True, parents=True)

if "video_logdir" not in os.environ:
    logdir = VIDEO_LOG_DIR / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    logdir.mkdir(exist_ok=True, parents=True)
    os.environ["video_logdir"] = str(logdir)


def territory_border_mask(board: np.ndarray, owner: int) -> np.ndarray:
    h, w = board.shape
    mask = np.zeros((h, w), dtype=bool)
    cells = np.argwhere(board == owner)

    for r, c in cells:
        if r == 0 or c == 0 or r == h - 1 or c == w - 1:
            mask[r, c] = True
            continue
        if (
            board[r - 1, c] != owner
            or board[r + 1, c] != owner
            or board[r, c - 1] != owner
            or board[r, c + 1] != owner
        ):
            mask[r, c] = True

    return mask


class VideoCallback(RLlibCallback):
    @staticmethod
    def _render_frame(
        board: np.ndarray,
        players: dict[int, country],
        colors: list,
        n_players: int,
        attacked_targets=None,
    ) -> np.ndarray:

        map_img = np.zeros((*board.shape, 3), dtype=np.float32)
        for i in range(n_players):
            map_img[board == i] = to_rgb(colors[i])
        map_img = (map_img * 255).astype(np.uint8)

        if attacked_targets:
            for target in attacked_targets:
                if target in players:
                    border = territory_border_mask(board, target)
                    map_img[border] = np.array([255, 0, 0], dtype=np.uint8)

        h, w = board.shape
        hist_w = max(n_players * 2, 40)
        hist_img = np.zeros((h, hist_w, 3), dtype=np.uint8)

        # Add a subtle dark gray background to the chart area so it's visible
        hist_img.fill(30)

        bar_width = hist_w // n_players
        biggest = max((p.money for p in players.values()), default=1)

        for i, player in players.items():
            scaled = player.money / biggest if biggest > 0 else 0.0
            bar_h = int(scaled * h)
            start_x = i * bar_width
            end_x = start_x + bar_width

            color_rgb = (np.array(to_rgb(colors[i])) * 255).astype(np.uint8)
            hist_img[0:bar_h, start_x:end_x] = color_rgb

        return np.hstack((map_img, np.flipud(hist_img)))

    def __init__(self):
        super().__init__()
        self.logdir: str = os.environ["video_logdir"]
        self.save_freq = VIDEO_SAVE_FREQ
        self.episode_counter = 0

    def on_episode_start(self, *, episode, worker: EnvRunner, **kwargs):
        record = self.episode_counter % self.save_freq == 0
        episode.user_data["record"] = record
        episode.user_data["frames"] = []
        self.episode_counter += 1

    def on_episode_step(self, *, episode, base_env, env_index, **kwargs):
        if not episode.user_data.get("record"):
            return

        sub_envs = base_env.get_sub_environments()
        actual_env: CustomEnvironment = sub_envs[env_index].par_env
        game = actual_env.game

        attacked_targets = []
        for agent_id in episode.get_agents():
            info = episode.last_info_for(agent_id=agent_id)
            if info and "attacks" in info:
                attacked_targets.extend([a["target"] for a in info["attacks"]])

        frame = self._render_frame(
            board=np.array(game.board),
            players=game.id_to_country,
            colors=game.countryColors,
            n_players=game.n_players,
            attacked_targets=attacked_targets,
        )

        episode.user_data["frames"].append(frame)

    def on_episode_end(self, *, episode, **kwargs):
        if not episode.user_data.get("record"):
            return

        frames = episode.user_data["frames"]

        if frames:
            fname = f"episode_{self.episode_counter}_{episode.episode_id}.mp4"
            imageio.mimsave(Path(self.logdir) / fname, frames, fps=8)
            frames.clear()
