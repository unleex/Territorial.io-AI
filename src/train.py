from prepare_env import ENV_NAME
from pathlib import Path

import ray
from ray import tune
from config import config


def train():
    ray.init(num_cpus=8, num_gpus=1)
    storage_uri = (Path("~/ray_results") / ENV_NAME).expanduser().resolve().as_uri()
    tune.run(
        "PPO",
        name="PPO",
        stop={"timesteps_total": 1_000_000},
        checkpoint_freq=10,
        storage_path=storage_uri,
        config=config.to_dict(),
        reuse_actors=True,
        # restore="~/ray_results/custom_env/PPO/PPO_custom_env_2487a_00000_0_2026-04-26_17-42-26/checkpoint_000019",
    )


if __name__ == "__main__":
    train()
