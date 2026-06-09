from configs.common import config, training_params, env_runners_params


CONFIG_NAME = "MAC"
training_params["train_batch_size"] = 1
training_params["minibatch_size"] = 1
config = config.training(**training_params).env_runners(
    num_env_runners=4, **env_runners_params, num_cpus_per_env_runner=2
)
NUM_CPUS = 8
NUM_GPUS = 0
