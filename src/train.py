from prepare_env import ENV_NAME
from pathlib import Path

from ray import tune
from config import config, NUM_GPUS, NUM_CPUS
import ray
from log import RUN_NAME


def train():
    ray.init(
        num_cpus=NUM_CPUS,
        num_gpus=NUM_GPUS,
        _temp_dir="/home2/mrgaschenko/tmp",
    )
    storage_uri = (Path("logs") / ENV_NAME).expanduser().resolve().as_uri()
    tune.run(
        "PPO",
        name=RUN_NAME,
        stop={"timesteps_total": 24_000_000},
        storage_path=storage_uri,
        config=config.to_dict(),
        reuse_actors=True,
        checkpoint_at_end=True,
        checkpoint_freq=20,
        # resume=True,
        restore="/home2/mrgaschenko/Territorial.io-AI/logs/custom_env/all_pretrained_agents/PPO_custom_env_8efad_00000_0_2026-06-02_18-05-57/checkpoint_000008",
    )


if __name__ == "__main__":
    train()
