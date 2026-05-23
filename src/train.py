from prepare_env import ENV_NAME
from pathlib import Path

from ray import tune
from config import config, NUM_GPUS, NUM_CPUS
import ray
from log import RUN_NAME


def train():
    ray.init(num_cpus=NUM_CPUS, num_gpus=NUM_GPUS)
    storage_uri = (Path("logs") / ENV_NAME).expanduser().resolve().as_uri()
    tune.run(
        "PPO",
        name=RUN_NAME,
        stop={"timesteps_total": 8_000_000},
        checkpoint_freq=10,
        storage_path=storage_uri,
        config=config.to_dict(),
        reuse_actors=True,
        # resume=True,
        restore="logs/custom_env/Multiagency/PPO_custom_env_dead7_00000_0_2026-05-23_19-38-47/checkpoint_000001",
    )


if __name__ == "__main__":
    train()
