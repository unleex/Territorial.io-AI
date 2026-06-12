from prepare_env import ENV_NAME
from pathlib import Path

from ray import tune
from config import config, NUM_GPUS, NUM_CPUS
import ray
from log import RUN_NAME
from ray.air.integrations.wandb import WandbLoggerCallback

from ray.rllib.algorithms.registry import POLICIES
from players.bot import BotPolicy

POLICIES["BotPolicy"] = BotPolicy


def train():
    ray.init(
        num_cpus=NUM_CPUS,
        num_gpus=NUM_GPUS,
        # _temp_dir="/home2/mrgaschenko/Territorial.io-AI/tmp",
    )
    storage_uri = (Path("logs") / ENV_NAME).expanduser().resolve().as_uri()
    tune.run(
        "PPO",
        name=RUN_NAME,
        stop={"timesteps_total": 10_000_000},
        storage_path=storage_uri,
        config=config.to_dict(),
        reuse_actors=True,
        checkpoint_at_end=True,
        checkpoint_freq=20,
        callbacks=[WandbLoggerCallback("Territorial.io")],
        # resume=True,
        # restore="/home2/mrgaschenko/Territorial.io-AI/logs/custom_env/time_aware/PPO_custom_env_45afa_00000_0_2026-06-12_11-04-02/checkpoint_000008_filtered",
    )


if __name__ == "__main__":
    train()
