from strategy_config import N_PLAYERS
from ray.rllib.algorithms.ppo import PPOConfig
from prepare_env import ENV_NAME
from players.model import MODEL_NAME
from callback import MergedCallback
from league_play_callback import policies


config = (
    PPOConfig()
    .environment(
        env=ENV_NAME,
        clip_actions=True,
        disable_env_checking=False,
    )
    .multi_agent(
        policies=policies,
        policy_mapping_fn=(
            lambda aid, *args, **kwargs: "p0"
        ),  # will be changed in league callback
        count_steps_by="agent_steps",
        policies_to_train=["p0"],
    )
    .debugging(log_level="WARNING")
    .framework(framework="torch")
    .api_stack(
        enable_rl_module_and_learner=False,
        enable_env_runner_and_connector_v2=False,
    )
    .callbacks(MergedCallback)
    # .checkpointing(checkpoint_trainable_policies_only=True)
).experimental(_disable_preprocessor_api=True)
training_params = dict(
    train_batch_size=None,
    minibatch_size=None,
    lr=2e-5,
    gamma=0.999,
    lambda_=0.9,
    use_gae=True,
    clip_param=0.2,
    grad_clip=0.5,
    entropy_coeff=0.01,
    vf_loss_coeff=0.25,
    num_epochs=10,
    model={"custom_model": MODEL_NAME},
)
config.simple_optimizer = False
env_runners_params = dict(batch_mode="truncate_episodes")
