import numpy as np
from ray.rllib.algorithms.ppo import PPOConfig
from custom_environment.env.custom_environment import CustomEnvironment
from model import MODEL_NAME
from prepare_env import ENV_NAME
from ray.rllib.callbacks.callbacks import RLlibCallback
from pathlib import Path
import imageio.v2 as imageio
from matplotlib.colors import to_rgb
from ray.rllib.env.env_runner import EnvRunner
from ray.rllib.env.multi_agent_env import MultiAgentEnvWrapper
from typing import Optional, Sequence
from ray.tune import get_context

VIDEO_LOG_DIR = (Path("~/ray_results") / ENV_NAME / "PPO" / "videos").expanduser()
VIDEO_LOG_DIR.mkdir(exist_ok=True)


class VideoCallback(RLlibCallback):
    @staticmethod
    def _board_to_rgb(
        board: np.ndarray, colors: list[str], n_players: int
    ) -> np.ndarray:
        img = np.zeros((*board.shape, 3), dtype=np.float32)
        for i in range(n_players):
            img[board == i] = to_rgb(colors[i])
        return (img * 255).astype(np.uint8)

    def __init__(self, env_runner_indices: Optional[Sequence[int]] = None):
        self._env_runner_indices = env_runner_indices

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
        if worker.worker_index < worker.config.get("num_env_runners", 0):
            return
        base_env: CustomEnvironment = base_env._unwrapped_env.par_env
        frame = self._board_to_rgb(
            np.array(base_env.game.board),
            base_env.game.countryColors,
            base_env.game.n_players,
        )
        episode.user_data["frames"].append(frame)

    def on_episode_end(self, *, worker: EnvRunner, episode, **kwargs):
        if worker.worker_index < worker.config.get("num_env_runners", 0):
            return
        log_dir = VIDEO_LOG_DIR / f"trial{get_context().get_trial_id()}"
        log_dir.mkdir(exist_ok=True)
        frames = episode.user_data["frames"]
        if frames:
            imageio.mimsave(
                log_dir / f"episode_{episode.episode_id}.mp4", frames, fps=8
            )


config = (
    PPOConfig()
    .environment(
        env=ENV_NAME,
        clip_actions=True,
        disable_env_checking=False,
    )
    .env_runners(num_env_runners=4, rollout_fragment_length=128)
    .training(
        train_batch_size=512,
        lr=2e-5,
        gamma=0.99,
        lambda_=0.9,
        use_gae=True,
        clip_param=0.4,
        grad_clip=None,
        entropy_coeff=0.1,
        vf_loss_coeff=0.25,
        minibatch_size=64,
        num_epochs=10,
        model={"custom_model": MODEL_NAME},
    )
    .multi_agent(
        policies={"p0"},
        policy_mapping_fn=(lambda aid, *args, **kwargs: "p0"),
    )
    .debugging(log_level="ERROR")
    .framework(framework="torch")
    .api_stack(
        enable_rl_module_and_learner=False,
        enable_env_runner_and_connector_v2=False,
    )
    .evaluation(
        evaluation_interval=15,
        evaluation_duration=10,
    )
    .callbacks(VideoCallback)
)
