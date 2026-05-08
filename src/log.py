from datetime import datetime
from environment import CustomEnvironment
import numpy as np
from prepare_env import ENV_NAME
from ray.rllib.callbacks.callbacks import RLlibCallback
from pathlib import Path
import imageio.v2 as imageio
from matplotlib.colors import to_rgb
from ray.rllib.env.env_runner import EnvRunner

VIDEO_LOG_DIR = (Path("logs") / ENV_NAME / "PPO" / "videos").expanduser()
VIDEO_LOG_DIR.mkdir(exist_ok=True, parents=True)


class VideoCallback(RLlibCallback):
    @staticmethod
    def _render_frame(
        board: np.ndarray,
        money: np.ndarray,
        colors: list,
        n_players: int,
        cols: int,
        rows: int,
    ) -> np.ndarray:
        # 1. Draw the Map (Left Side)
        map_img = np.zeros((*board.shape, 3), dtype=np.float32)
        for i in range(n_players):
            map_img[board == i] = to_rgb(colors[i])
        map_img = (map_img * 255).astype(np.uint8)

        # 2. Draw the Histogram (Right Side)
        # We define a fixed width for the chart to keep memory low.
        # Using 2 pixels per player as a minimum, or 40 pixels total.
        h, w = board.shape
        hist_w = max(n_players * 2, 40)
        hist_img = np.zeros((h, hist_w, 3), dtype=np.uint8)

        # Add a subtle dark gray background to the chart area so it's visible
        hist_img.fill(30)

        bar_width = hist_w // n_players

        for i in range(n_players):
            # Scale the money according to your formula
            # Assuming money can be negative, taking absolute value as requested
            scaled = abs(money[i]) / 1500.0 / cols / rows

            # Clamp the value between 0 and 1 so it doesn't draw outside the array
            scaled = max(0.0, min(1.0, scaled))

            # Calculate how many pixels high the bar should be
            bar_h = int(scaled * h)

            if bar_h > 0:
                start_x = i * bar_width
                # Leave a 1-pixel gap between bars if possible, otherwise use full width
                end_x = start_x + bar_width - (1 if bar_width > 2 else 0)

                color_rgb = (np.array(to_rgb(colors[i])) * 255).astype(np.uint8)

                # NumPy indexing: [Y_start:Y_end, X_start:X_end]
                # We draw from the bottom (h) upwards (h - bar_h)
                hist_img[h - bar_h : h, start_x:end_x] = color_rgb

        # 3. Concatenate the map and the histogram side-by-side
        return np.hstack((map_img, hist_img))

    def __init__(self):
        super().__init__()
        self.logdir = VIDEO_LOG_DIR / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.logdir.mkdir(exist_ok=True, parents=True)
        self.save_freq = 40
        self.episode_counter = 0

    def on_episode_start(self, *, episode, env_runner: EnvRunner, **kwargs):
        record = (
            self.episode_counter % self.save_freq == 0 and env_runner.worker_index == 1
        )
        episode.user_data["record"] = record
        episode.user_data["frames"] = []
        self.episode_counter += 1

    def on_episode_step(self, *, episode, base_env, env_index, **kwargs):
        if not episode.user_data.get("record"):
            return

        sub_envs = base_env.get_sub_environments()
        actual_env: CustomEnvironment = sub_envs[env_index].par_env
        game = actual_env.game

        # Pass the required game state variables into our optimized renderer
        frame = self._render_frame(
            board=np.array(game.board),
            money=np.array(
                [player.money for player in game.id_to_country.values()]
            ),  # Assumes game.money is accessible like this
            colors=game.countryColors,
            n_players=game.n_players,
            cols=game.n_grid_columns,
            rows=game.n_grid_rows,
        )

        episode.user_data["frames"].append(frame)

    def on_episode_end(self, *, episode, **kwargs):
        if not episode.user_data.get("record"):
            return

        frames = episode.user_data["frames"]

        if frames:
            fname = f"episode_{self.episode_counter}_{episode.episode_id}.mp4"
            imageio.mimsave(self.logdir / fname, frames, fps=8)
            # Crucial: Clear the list to free up the 188GB system RAM
            frames.clear()
