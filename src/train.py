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
        model: PPO,
        video_log_folder: str = "logs/videos",
        num_games: int = 5,
    ):
        super().__init__()
        self.eval_freq = eval_freq
        self.video_log_folder = video_log_folder
        self.num_games = num_games
        self.model = model

    def _on_step(self) -> bool:
        if self.num_timesteps % self.eval_freq != 0:
            return True
        epoch = self.num_timesteps // self.eval_freq
        print(f"[INFO] Running evaluation at epoch {epoch}")
        evaluate(
            model=self.model,
            num_games=self.num_games,
            video_log_folder=f"{self.video_log_folder}/epoch {epoch}",
        )
        return True


def train():

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
    )
    eval_callback = PeriodicEvalCallback(
        eval_freq=eval_freq,
        model=model,
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
