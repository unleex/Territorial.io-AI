from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.tune.registry import register_env

from environment import CustomEnvironment


def make_env(config):
    n_bots = 4
    base = CustomEnvironment(
        rendering=False, n_agents=8 - n_bots, landscape_path="maps/random.npy"
    )
    wrapped = ParallelPettingZooEnv(base)
    wrapped._agent_ids = set(getattr(base, "possible_agents"))
    return wrapped


ENV_NAME = "custom_env"
register_env(ENV_NAME, make_env)
