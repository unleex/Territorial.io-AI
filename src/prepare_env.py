from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.tune.registry import register_env

from environment import CustomEnvironment


def make_env(config):
    n_bots = 2
    base = CustomEnvironment(rendering=False, n_agents=8 - n_bots)
    wrapped = ParallelPettingZooEnv(base)
    wrapped._agent_ids = set(getattr(base, "possible_agents"))
    return wrapped


ENV_NAME = "custom_env"
register_env(ENV_NAME, make_env)
