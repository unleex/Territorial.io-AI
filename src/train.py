from prepare_env import ENV_NAME
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import (
    BaseCallback,
)
from evaluate import evaluate
from pathlib import Path

import ray
from ray import tune
from config import config


def train():
<<<<<<< HEAD

    env = make_env(num_cpus=8, render=False)
    latest_checkpoint, steps_done = find_latest_checkpoint("models/", "ppo_v1")
    timesteps = 300_000
    checkpoint_freq = 50_000
    eval_freq = 25_000
    if latest_checkpoint:
        print(f"[INFO] Resuming from checkpoint: {latest_checkpoint}")
        print(f"[INFO] Steps already done: {steps_done} / {timesteps}")
        model = PPO.load(latest_checkpoint, env=env)
        timesteps -= steps_done
    else:
        print("[INFO] No checkpoint found, starting from scratch.")
        model = PPO(
            policy="MultiInputPolicy",
            env=env,
            verbose=1,
            learning_rate=3e-4,
            n_steps=1024,
            batch_size=128,
            n_epochs=10,
            gamma=0.99,
            tensorboard_log="logs/ppo_territorial_tensorboard/",
            policy_kwargs={"normalize_images": False},
            device="cpu",
        )

    if timesteps <= 0:
        print("[INFO] Training already complete!")
        return

    # 4. Setup Callbacks (Save every 50k steps)
    checkpoint_callback = CheckpointCallback(
        save_freq=checkpoint_freq, save_path="models", name_prefix="ppo_v1"
=======
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
        resources_per_trial={'gpu': 1},
        # restore="~/ray_results/custom_env/PPO/PPO_custom_env_2487a_00000_0_2026-04-26_17-42-26/checkpoint_000019",
>>>>>>> 05c4ae6598dd813427af78e4d31fec3a2ded95ad
    )


if __name__ == "__main__":
    train()
