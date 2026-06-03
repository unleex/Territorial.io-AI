from configs.common import config, training_params

training_params["train_batch_size"] = 32
training_params["minibatch_size"] = 8
CONFIG_NAME = "TEST"
config = config.training(**training_params)
NUM_CPUS = 12
NUM_GPUS = 0
