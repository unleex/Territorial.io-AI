from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.tune.registry import register_env

from environment import CustomEnvironment
from strategy_config import N_PLAYERS, MAP_SIZE


def make_env(_=None):
    rows, cols = MAP_SIZE
    base = CustomEnvironment(
        rendering=False,
        grid_rows=rows,
        grid_columns=cols,
        n_players=N_PLAYERS,
    )
    wrapped = ParallelPettingZooEnv(base)
    wrapped._agent_ids = set(getattr(base, "possible_agents"))
    return wrapped


ENV_NAME = "custom_env"
register_env(ENV_NAME, make_env)
