from datetime import datetime
import warnings
from pathlib import Path
import os

import imageio.v2 as imageio
import numpy as np
from matplotlib.colors import to_rgb

from environment import CustomEnvironment
from ray.rllib.callbacks.callbacks import RLlibCallback
from ray.rllib.env.env_runner import EnvRunner
from game.countryClass import country
from strategy_config import RUN_NAME


VIDEO_SAVE_FREQ = 5
VIDEO_LOG_DIR = (
    Path("logs").absolute() / "custom_env" / RUN_NAME / "videos"
).expanduser()
VIDEO_LOG_DIR.mkdir(exist_ok=True, parents=True)

if "video_logdir" not in os.environ:
    logdir = VIDEO_LOG_DIR / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    logdir.mkdir(exist_ok=True, parents=True)
    os.environ["video_logdir"] = str(logdir)


def attacker_touching_target_mask(
    board: np.ndarray, attacker: int, target: int
) -> np.ndarray:
    h, w = board.shape
    mask = np.zeros((h, w), dtype=bool)
    if attacker == target:
        warnings.warn(f"Self attack found: {attacker} {target}")
        return mask

    a_cells = np.argwhere(board == attacker)
    for r, c in a_cells:
        if r > 0 and board[r - 1, c] == target:
            mask[r, c] = True
            continue
        if r < h - 1 and board[r + 1, c] == target:
            mask[r, c] = True
            continue
        if c > 0 and board[r, c - 1] == target:
            mask[r, c] = True
            continue
        if c < w - 1 and board[r, c + 1] == target:
            mask[r, c] = True
            continue

    return mask


class VideoCallback(RLlibCallback):
    @staticmethod
    def _render_frame(
        board: np.ndarray,
        players: dict[int, country],
        colors: list,
        n_players: int,
        attacks=None,
    ) -> np.ndarray:
        map_img = np.zeros((*board.shape, 3), dtype=np.float32)
        for i in range(n_players):
            map_img[board == i] = to_rgb(colors[i])
        map_img = (map_img * 255).astype(np.uint8)

        if attacks is not None:
            for attack in attacks:
                attacker = attack["attacker"]
                target = attack["target"]
                if np.any(board == attacker) and np.any(board == target):
                    mask = attacker_touching_target_mask(board, attacker, target)
                    map_img[mask] = np.array([255, 0, 0], dtype=np.uint8)

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
        episode.user_data["colors"] = None
        self.episode_counter += 1

    def on_episode_step(self, *, episode, base_env, env_index, **kwargs):
        if not episode.user_data.get("record"):
            return

        sub_envs = base_env.get_sub_environments()
        actual_env: CustomEnvironment = sub_envs[env_index].par_env
        game = actual_env.game

        if episode.user_data["colors"] is None:
            episode.user_data["colors"] = list(actual_env.country_colors)

        attacks = []
        for agent_id in episode.get_agents():
            info = episode.last_info_for(agent_id=agent_id)
            if info and "attacks" in info:
                attacks.extend(info["attacks"])

        frame = self._render_frame(
            board=np.array(game.board),
            players=game.id_to_country,
            colors=episode.user_data["colors"],
            n_players=game.n_players,
            attacks=attacks,
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
