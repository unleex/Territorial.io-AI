from prepare_env import ENV_NAME
from pathlib import Path

from ray import tune
from config import config


def train():
    storage_uri = (Path("~/ray_results") / ENV_NAME).expanduser().resolve().as_uri()
    tune.run(
        "PPO",
        name="PPO",
        stop={"timesteps_total": 1_000_000},
        checkpoint_freq=20,
        storage_path=storage_uri,
        config=config.to_dict(),
        reuse_actors=True,
        # restore="~/ray_results/custom_env/PPO/PPO_custom_env_2487a_00000_0_2026-04-26_17-42-26/checkpoint_000019",
    )


if __name__ == "__main__":
    train()
