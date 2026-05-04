from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.tune.registry import register_env

from custom_environment.custom_environment_v0 import CustomEnvironment
from supersuit import frame_stack_v1


def _make_rllib_env(config):
    base = CustomEnvironment()
    base = frame_stack_v1(base)
    wrapped = ParallelPettingZooEnv(base)
    wrapped._agent_ids = set(getattr(base, "possible_agents", []))
    return wrapped


ENV_NAME = "custom_env"
register_env(ENV_NAME, _make_rllib_env)
