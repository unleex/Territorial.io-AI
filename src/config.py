# from ray.rllib.examples.rl_modules.classes.action_masking_rlm import (
#     ActionMaskingTorchRLModule,
# )
# from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.rllib.core.rl_module.default_model_config import DefaultModelConfig
from custom_environment.env.custom_environment import CustomEnvironment
from prepare_env import make_env, find_latest_checkpoint, ENV_NAME
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import (
    BaseCallback,
)
from evaluate import evaluate
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from stable_baselines3.common.torch_layers import NatureCNN
import os
from pathlib import Path

import ray
import supersuit as ss
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.tune.registry import register_env
from torch import nn

config = (
    PPOConfig()
    .environment(
        env=ENV_NAME,
        clip_actions=True,
        disable_env_checking=False,
    )
    .env_runners(num_env_runners=4, rollout_fragment_length=128)
    .training(
        train_batch_size=512,
        lr=2e-5,
        gamma=0.99,
        lambda_=0.9,
        use_gae=True,
        clip_param=0.4,
        grad_clip=None,
        entropy_coeff=0.1,
        vf_loss_coeff=0.25,
        minibatch_size=64,
        num_epochs=10,
        model={"custom_model": MODEL_NAME},
    )
    .multi_agent(
        policies={"p0"},
        policy_mapping_fn=(lambda aid, *args, **kwargs: "p0"),
    )
    .debugging(log_level="ERROR")
    .framework(framework="torch")
    .resources(
        num_cpus_for_main_process=1
    )  # num_gpus=int(os.environ.get("RLLIB_NUM_GPUS", "0")))
    .rl_module(
        model_config=DefaultModelConfig(
            # Use explicit filters because [9, 80, 80] has no default auto-CNN config.
            conv_filters=[
                [9, 3, 3],  # 1st CNN layer: num_filters, kernel, stride
                [32, 3, 3],  # 2nd CNN layer
                [64, 3, 3],  # 3rd CNN layer
                [128, 3, 3],  # 4th CNN layer
            ],
        ),
        # Masking version (keep for later):
        # rl_module_spec=RLModuleSpec(module_class=ActionMaskingTorchRLModule),
    )
)
