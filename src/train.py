from utility import make_env, find_latest_checkpoint
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
    CheckpointCallback,
)
from evaluate import evaluate


class PeriodicEvalCallback(BaseCallback):
    def __init__(
        self,
        eval_freq: int,
        video_log_folder: str = "logs/videos",
        num_games: int = 5,
    ):
        super().__init__()
        self.eval_freq = eval_freq
        self.video_log_folder = video_log_folder
        self.num_games = num_games

    def _on_step(self) -> bool:
        if self.num_timesteps % self.eval_freq != 0:
            return True
        epoch = self.num_timesteps // self.eval_freq
        print(f"[INFO] Running evaluation at epoch {epoch}")
        evaluate(
            num_games=self.num_games,
            video_log_folder=f"{self.video_log_folder}/epoch {epoch}",
        )
        return True


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
    latest_checkpoint, steps_done = find_latest_checkpoint("models/", "ppo_v1")
    timesteps = 1_000_000
    callback_freq = 50_000
    # 3. Define the Model due to last checkpoint or from scratch
    if latest_checkpoint:
        print(f"[INFO] Resuming from checkpoint: {latest_checkpoint}")
        print(f"[INFO] Steps already done: {steps_done} / {timesteps}")
        model = PPO.load(latest_checkpoint, env=env)
        timesteps -= steps_done
    else:
        print("[INFO] No checkpoint found, starting from scratch.")
        model = PPO(
            policy="CnnPolicy",
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
        save_freq=callback_freq, save_path="models", name_prefix="ppo_v1"
    )
    eval_callback = PeriodicEvalCallback(
        eval_freq=callback_freq,
        video_log_folder="logs/videos",
        num_games=5,
    )
    callbacks = CallbackList([checkpoint_callback, eval_callback])

    # 5. The Train Loop
    model.learn(total_timesteps=timesteps, callback=callbacks, progress_bar=True)

    # 6. Save Final
    model.save("ppo_territorial_final")


if __name__ == "__main__":
    train()
