from ray.rllib.algorithms.ppo import PPOConfig
from model import MODEL_NAME
from prepare_env import ENV_NAME
from pathlib import Path
from log import VideoCallback

config = (
    PPOConfig()
    .environment(
        env=ENV_NAME,
        clip_actions=True,
        disable_env_checking=False,
    )
    .training(
        train_batch_size=20_000,
        lr=2e-5,
        gamma=0.99,
        lambda_=0.9,
        use_gae=True,
        clip_param=0.4,
        grad_clip=0.5,
        entropy_coeff=0.1,
        vf_loss_coeff=0.25,
        minibatch_size=4096,
        num_epochs=10,
        model={"custom_model": MODEL_NAME},
    )
    .multi_agent(
        policies={"p0"},
        policy_mapping_fn=(lambda aid, *args, **kwargs: "p0"),
    )
    .debugging(log_level="DEBUG")
    .framework(framework="torch")
    .api_stack(
        enable_rl_module_and_learner=False,
        enable_env_runner_and_connector_v2=False,
    )
    .callbacks(VideoCallback)
    .resources(num_gpus=1)
    .env_runners(num_env_runners=8,num_envs_per_env_runner=4)
)
