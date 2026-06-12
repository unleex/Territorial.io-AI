from configs.common import config, training_params, env_runners_params


training_params["train_batch_size"] = 5000
training_params["minibatch_size"] = 16
CONFIG_NAME = "TEST"
config = config.training(**training_params).env_runners(
    **env_runners_params, num_env_runners=0
)
NUM_CPUS = 19
NUM_GPUS = 0
