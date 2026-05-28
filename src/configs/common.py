from ray.rllib.algorithms.ppo import PPOConfig
from prepare_env import ENV_NAME
from model import MODEL_NAME
from callback import MergedCallback

config = (
    PPOConfig()
    .environment(
        env=ENV_NAME,
        clip_actions=True,
        disable_env_checking=False,
    )
    .multi_agent(
        policies={"p0"},
        policy_mapping_fn=(lambda aid, *args, **kwargs: "p0"),
        count_steps_by="agent_steps",
        policies_to_train=["p0"],
        policy_map_capacity=10,
        policy_states_are_swappable=True,
    )
    .debugging(log_level="WARN")
    .framework(framework="torch")
    .api_stack(
        enable_rl_module_and_learner=False,
        enable_env_runner_and_connector_v2=False,
    )
    .callbacks(MergedCallback)
)
training_params = dict(
    train_batch_size=None,
    minibatch_size=None,
    lr=2e-5,
    gamma=0.99,
    lambda_=0.9,
    use_gae=True,
    clip_param=0.2,
    grad_clip=0.5,
    entropy_coeff=0.01,
    vf_loss_coeff=0.25,
    num_epochs=15,
    model={"custom_model": MODEL_NAME},
)
