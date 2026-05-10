import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.offline.json_writer import JsonWriter
from config import config
from model import MODEL_NAME
from prepare_env import ENV_NAME

# --- CONFIGURATION ---
N_EPISODES = 1  # Number of episodes to "memorize"
DATA_PATH = "overfit_data"
TRAIN_ITERATIONS = 50


def collect_data(env_name, n_episodes, output_path):
    """Phase 1: Record episodes to a JSON file."""
    print(f"--- Phase 1: Recording {n_episodes} episodes to {output_path} ---")
    writer = JsonWriter(output_path)

    # We use a standard PPO config just to get a sample provider
    sampler_config = (
        PPOConfig()
        .environment(env_name)
        .env_runners(num_env_runners=0)
        .training(
            model={"custom_model": MODEL_NAME}, minibatch_size=10, train_batch_size=10
        )
        .api_stack(
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False,
        )
        .multi_agent(
            policies={"default_policy"},
            policy_mapping_fn=(lambda aid, *args, **kwargs: "default_policy"),
        )
    )
    algo = sampler_config.build()

    for _ in range(n_episodes):
        # This collects one complete episode
        batch = algo.env_runner.sample()
        writer.write(batch)

    algo.stop()
    print("Recording complete.\n")


def run_overfit_test(input_path, config: PPOConfig):
    """Phase 2: Train the model on the fixed recorded data."""
    print(f"--- Phase 2: Overfitting on {input_path} ---")
    config = (
        config.offline_data(
            input_=input_path,
            input_read_sample_batches=True,
        )
        .env_runners(num_env_runners=0)
        .evaluation(evaluation_interval=None)
        .callbacks(None)
        .multi_agent(
            policies={"default_policy"},
            policy_mapping_fn=(lambda aid, *args, **kwargs: "default_policy"),
        )
    )
    algo = config.build()

    print("\nStarting Training Loop...")
    print(f"{'Iter':<5} | {'Total':<9} | {'Policy':<9} | {'Value':<9} | {'Entropy':<9}")
    print("-" * 50)

    for i in range(500):  # Increased iterations so it has time to converge
        results = algo.train()

        # Navigate to the policy's learner data
        stats = results["info"]["learner"]["default_policy"]["learner_stats"]

        # Extract individual components (using .get to avoid crashes if a key is missing)
        total_loss = stats.get("total_loss", float("nan"))
        policy_loss = stats.get("policy_loss", float("nan"))
        vf_loss = stats.get("vf_loss", float("nan"))
        entropy = stats.get("entropy", float("nan"))

        # Print the formatted row
        print(
            f"{i:<5} | {total_loss:<9.4f} | {policy_loss:<9.4f} | {vf_loss:<9.4f} | {entropy:<9.4f}"
        )

        # --- THE NEW SUCCESS CRITERION ---
        # 1. Value loss must be near zero (Critic has memorized the rewards)
        # 2. Wait at least 10 iterations to ensure it's not a fluke initialization
        if vf_loss < 0.0005 and i > 10:
            print("\nSUCCESS: Value Function has perfectly memorized the trajectory!")
            print("Your CNN architecture and data pipeline are working correctly.")
            break

    algo.stop()


if __name__ == "__main__":
    ray.init()

    # 1. Generate the 'exam paper'
    collect_data(ENV_NAME, N_EPISODES, DATA_PATH)
    run_overfit_test(DATA_PATH, config)
