from configs.common import config, training_params, env_runners_params

training_params["train_batch_size"] = 4096
training_params["minibatch_size"] = 1024
CONFIG_NAME = "A100"
config = (
    config.training(**training_params)
    .resources(num_gpus=1)
    .env_runners(
        num_env_runners=18,
        num_envs_per_env_runner=2,
        num_cpus_per_env_runner=1,
        **env_runners_params,
    )
    .learners(
        num_gpus_per_learner=1,
        num_learners=1,
    )
)
NUM_CPUS = 20
NUM_GPUS = 1
