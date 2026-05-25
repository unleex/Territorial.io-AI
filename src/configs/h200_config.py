from model import MODEL_NAME
from configs.common import config, training_params

CONFIG_NAME = "H200"
training_params["train_batch_size"] = 32_768 // 2
training_params["minibatch_size"] = 8192 // 2
config = (
    config.training(**training_params)
    .resources(num_gpus=1)
    .env_runners(
        num_env_runners=18,
        num_envs_per_env_runner=4,
        num_cpus_per_env_runner=1,
    )
    .learners(
        num_gpus_per_learner=1,
        num_learners=1,
    )
)
NUM_CPUS = 20
NUM_GPUS = 1
