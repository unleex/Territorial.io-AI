from model import MODEL_NAME
from prepare_env import ENV_NAME
from log import VideoCallback
from configs.common import config

CONFIG_NAME = "H200"
config = (
    config.environment(
        env=ENV_NAME,
        clip_actions=True,
        disable_env_checking=False,
    )
    .training(
        train_batch_size=32_768 // 2,
        minibatch_size=8192 // 2,
        lr=5e-5,
        gamma=0.99,
        lambda_=0.9,
        use_gae=True,
        clip_param=0.4,
        grad_clip=0.5,
        entropy_coeff=0.1,
        vf_loss_coeff=0.25,
        num_epochs=15,
        model={"custom_model": MODEL_NAME},
    )
    .multi_agent(
        policies={"p0"},
        policy_mapping_fn=(lambda aid, *args, **kwargs: "p0"),
        count_steps_by="agent_steps",
    )
    .debugging(log_level="WARN")
    .framework(framework="torch")
    .api_stack(
        enable_rl_module_and_learner=False,
        enable_env_runner_and_connector_v2=False,
    )
    .callbacks(VideoCallback)
    .resources(num_gpus=1)
    .env_runners(
        num_env_runners=18,
        num_envs_per_env_runner=14,
        num_cpus_per_env_runner=1,
    )
    .learners(
        num_gpus_per_learner=1,
        num_learners=1,
    )
)
NUM_CPUS = 20
NUM_GPUS = 1
