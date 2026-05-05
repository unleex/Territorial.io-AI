from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.tune.registry import register_env

from env.custom_environment import CustomEnvironment


def _make_rllib_env(config):
    base = CustomEnvironment(rendering=False)
    wrapped = ParallelPettingZooEnv(base)
    wrapped._agent_ids = set(getattr(base, "possible_agents", []))
    return wrapped


ENV_NAME = "custom_env"
register_env(ENV_NAME, _make_rllib_env)
