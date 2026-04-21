import supersuit as ss
import os
import glob
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import VecTransposeImage
from pettingzoo.utils.conversions import aec_to_parallel
from custom_environment.custom_environment_v0 import CustomEnvironment

def make_env():
    """
    Factory: AEC → Parallel → VecEnv pipeline.
    SuperSuit's pettingzoo_env_to_vec_env_v1 requires a PARALLEL env.
    """
    env = CustomEnvironment()

    env = ss.black_death_v3(env) 
    env = ss.pettingzoo_env_to_vec_env_v1(env)

    env = ss.concat_vec_envs_v1(env, num_vec_envs=4, num_cpus=0, base_class="stable_baselines3")

    return env

def find_latest_checkpoint(checkpoint_dir, prefix):
    """Находит последний сохранённый чекпоинт по номеру шага."""
    pattern = os.path.join(checkpoint_dir, f"{prefix}_*_steps.zip")
    files = glob.glob(pattern)
    if not files:
        return None, 0
    # Извлекаем номер шага из имени файла и берём максимальный
    def extract_step(path):
        name = os.path.basename(path)           # ppo_v1_300000_steps.zip
        parts = name.replace(".zip", "").split("_")
        for part in reversed(parts):
            if part.isdigit():
                return int(part)
        return 0
    latest = max(files, key=extract_step)
    steps_done = extract_step(latest)
    return latest, steps_done


def train():
    # 1. Instantiate the PettingZoo environment
    env = make_env()
    # env = VecTransposeImage(env)  # (VecEnv wrapper to handle image observations)

    # 2. Wrap for compatibility (no need)
    # SB3 expects a single-agent Gymnasium env. SuperSuit handles the conversion.
    # env = ss.pettingzoo_env_to_vec_env_v1(env)

    # Concatenate for parallel training (e.g., run 8 games at once)
    # env = ss.concat_vec_envs_v1(
    #     env, num_vec_envs=8, num_cpus=0
    # )
    latest_checkpoint, steps_done = find_latest_checkpoint(
        "./models/", "ppo_v1"
    )

    # 3. Define the Model due to last checkpoint or from scratch
    if latest_checkpoint:
        print(f"[INFO] Resuming from checkpoint: {latest_checkpoint}")
        print(f"[INFO] Steps already done: {steps_done} / {1_000_000}")
        model = PPO.load(latest_checkpoint, env=env)
        remaining = 1_000_000 - steps_done
    else:
        print("[INFO] No checkpoint found, starting from scratch.")
        model = PPO(
            policy="CnnPolicy",
            env=env,
            verbose=1,
            learning_rate=3e-4,
            n_steps=1024,
            batch_size=32,
            n_epochs=10,
            gamma=0.99,
            tensorboard_log="./ppo_territorial_tensorboard/",
            policy_kwargs={"normalize_images": False},
        )
        remaining = 1_000_000

    
    if remaining <= 0:
        print("[INFO] Training already complete!")
        return

    # 4. Setup Callbacks (Save every 50k steps)
    checkpoint_callback = CheckpointCallback(
        save_freq=50000, save_path="./models/", name_prefix="ppo_v1"
    )

    # 5. The Train Loop
    model.learn(
        total_timesteps=1_000_000, callback=checkpoint_callback, progress_bar=True
    )

    # 6. Save Final
    model.save("ppo_territorial_final")


if __name__ == "__main__":
    train()
