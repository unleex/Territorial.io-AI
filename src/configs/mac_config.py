from players.model import MODEL_NAME
from configs.cozmmon import config, training_params, env_runners_params


CONFIG_NAME = "MAC"
training_params["train_batch_size"] = 64
training_params["minibatch_size"] = 16
config = config.training(**training_params).env_runners(
    num_env_runners=6, **env_runners_params
)
NUM_CPUS = 12
NUM_GPUS = 0
