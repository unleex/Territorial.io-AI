from model import MODEL_NAME
from configs.common import config, training_params, env_runners_params


CONFIG_NAME = "MAC"
config = config.training(
    train_batch_size=256,
    lr=2e-5,
    gamma=0.999,
    lambda_=0.9,
    use_gae=True,
    clip_param=0.4,
    grad_clip=0.5,
    entropy_coeff=0.1,
    vf_loss_coeff=0.25,
    minibatch_size=64,
    num_epochs=10,
    model={"custom_model": MODEL_NAME},
).env_runners(num_env_runners=6, **env_runners_params)
NUM_CPUS = 8
NUM_GPUS = 0
