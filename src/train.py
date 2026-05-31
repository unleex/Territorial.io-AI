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
        storage_path=storage_uri,
        config=config.to_dict(),
        reuse_actors=True,
        checkpoint_at_end=True,
        checkpoint_freq=0,
        # resume=True,
        # restore="logs/custom_env/all_pretrained_agents/PPO_custom_env_038bc_00000_0_2026-05-31_23-40-52/checkpoint_000023",
    )


if __name__ == "__main__":
    train()
