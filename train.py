import supersuit as ss
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from custom_environment.custom_environment_v0 import CustomEnvironment


def train():
    # 1. Instantiate the PettingZoo environment
    env = CustomEnvironment()

    # 2. Wrap for compatibility
    # SB3 expects a single-agent Gymnasium env. SuperSuit handles the conversion.
    env = ss.pettingzoo_env_to_vec_env_v1(env)

    # Concatenate for parallel training (e.g., run 8 games at once)
    env = ss.concat_vec_envs_v1(
        env, num_vec_envs=8, num_cpus=4, base_class="stable_baselines3"
    )

    # 3. Define the Model
    # Using CnnPolicy because your observation is (C, H, W)
    model = PPO(
        policy="CnnPolicy",
        env=env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,  # Rollout length
        batch_size=64,  # Mini-batch size
        n_epochs=10,  # Optimization epochs per update
        gamma=0.99,  # Discount factor
        tensorboard_log="./ppo_territorial_tensorboard/",
    )

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
