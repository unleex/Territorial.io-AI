from prepare_env import ENV_NAME
from pathlib import Path

from ray import tune
from config import config, NUM_GPUS, NUM_CPUS
import ray


def train():
    ray.init(num_cpus=NUM_CPUS, num_gpus=NUM_GPUS)
    storage_uri = (Path("logs") / ENV_NAME).expanduser().resolve().as_uri()
    tune.run(
        "PPO",
        name="PPO",
        stop={"timesteps_total": 8_000_000},
        checkpoint_freq=10,
        storage_path=storage_uri,
        config=config.to_dict(),
        reuse_actors=True,
        restore="logs/custom_env/PPO/PPO_custom_env_2d67b_00000_0_2026-05-06_21-48-31/checkpoint_000009",
    )


if __name__ == "__main__":
    train()
