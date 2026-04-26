from prepare_env import make_env, find_latest_checkpoint, ENV_NAME
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import (
    BaseCallback,
)
from evaluate import evaluate
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from stable_baselines3.common.torch_layers import NatureCNN
import os
from pathlib import Path

import ray
import supersuit as ss
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.tune.registry import register_env
from torch import nn
from config import config


class PeriodicEvalCallback(BaseCallback):
    def __init__(
        self,
        eval_freq: int,
        model: MaskablePPO,
        video_log_folder: str = "logs/videos",
        num_games: int = 5,
    ):
        super().__init__()
        self.eval_freq = eval_freq
        self.video_log_folder = video_log_folder
        self.num_games = num_games
        self.model = model

    def _on_step(self) -> bool:
        if self.num_timesteps % self.eval_freq != 0:
            return True
        epoch = self.num_timesteps // self.eval_freq
        print(f"[INFO] Running evaluation at epoch {epoch}")
        evaluate(
            model=self.model,
            num_games=self.num_games,
            video_log_folder=f"{self.video_log_folder}/epoch {epoch}",
        )
        return True


from ray.rllib.algorithms.ppo import PPOConfig


def train():

    storage_uri = (Path("~/ray_results") / ENV_NAME).expanduser().resolve().as_uri()

    tune.run(
        "PPO",
        name="PPO",
        stop={"timesteps_total": 5000000 if not os.environ.get("CI") else 50000},
        checkpoint_freq=10,
        storage_path=storage_uri,
        config=config.to_dict(),
    )


if __name__ == "__main__":
    train()
