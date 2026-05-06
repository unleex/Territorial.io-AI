from ray.rllib.algorithms.ppo import PPOConfig
from model import MODEL_NAME
from prepare_env import ENV_NAME
from log import VideoCallback

config = (
    PPOConfig()
    .environment(
        env=ENV_NAME,
        clip_actions=True,
        disable_env_checking=False,
    )
    .training(
        train_batch_size=1500,
        lr=2e-5,
        gamma=0.99,
        lambda_=0.9,
        use_gae=True,
        clip_param=0.4,
        grad_clip=0.5,
        entropy_coeff=0.1,
        vf_loss_coeff=0.25,
        minibatch_size=256,
        num_epochs=10,
        model={"custom_model": MODEL_NAME},
    )
    .multi_agent(
        policies={"p0"},
        policy_mapping_fn=(lambda aid, *args, **kwargs: "p0"),
    )
    .debugging(log_level="INFO")
    .framework(framework="torch")
    .api_stack(
        enable_rl_module_and_learner=False,
        enable_env_runner_and_connector_v2=False,
    )
    .callbacks(VideoCallback)
    .env_runners(num_env_runners=6)
)
NUM_CPUS = 8
NUM_GPUS = 0
