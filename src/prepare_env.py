import pettingzoo
from sympy import true

from custom_environment.custom_environment_v0 import CustomEnvironment
import glob
import supersuit as ss
import os
from supersuit.vector import MakeCPUAsyncConstructor
from supersuit.vector.vector_constructors import vec_env_args
from sb3_contrib.common.wrappers import ActionMasker
from pettingzoo.utils import BaseWrapper
from gymnasium import Env, make
from pettingzoo.utils.conversions import to_parallel


def fixed_concat_vec_envs_v1(vec_env, num_vec_envs, num_cpus=0, base_class="gymnasium"):
    num_cpus = min(num_cpus, num_vec_envs)
    if num_cpus <= 1:
        vec_env = MakeCPUAsyncConstructor(num_cpus)(
            *vec_env_args(vec_env, num_vec_envs)
        )
    else:
        vec_env = MakeCPUAsyncConstructor(num_cpus)(
            *vec_env_args(vec_env, num_vec_envs),
            vec_env.observation_space,
            vec_env.action_space,
        )

    if base_class == "gymnasium":
        return vec_env
    elif base_class == "stable_baselines":
        from supersuit.vector.sb_vector_wrapper import SBVecEnvWrapper

        return SBVecEnvWrapper(vec_env)
    elif base_class == "stable_baselines3":
        from supersuit.vector.sb3_vector_wrapper import SB3VecEnvWrapper

        return SB3VecEnvWrapper(vec_env)
    else:
        raise ValueError(
            "supersuit_vec_env only supports 'gymnasium', 'stable_baselines', and 'stable_baselines3' for its base_class"
        )


class SB3ActionMaskWrapper(BaseWrapper, Env):
    """Wrapper to allow PettingZoo environments to be used with SB3 illegal action masking."""

    def reset(self, seed=None, options=None):
        """Gymnasium-like reset function which assigns obs/action spaces to be the same for each agent.

        This is required as SB3 is designed for single-agent RL and doesn't expect obs/action spaces to be functions
        """
        super().reset(seed, options)

        # Strip the action mask out from the observation space
        self.observation_space = super().observation_space(self.possible_agents[0])
        self.action_space = super().action_space(self.possible_agents[0])

        # Return initial observation, info (PettingZoo AEC envs do not by default)
        return self.observe(self.possible_agents[0]), {}

    def step(self, action):
        """Gymnasium-like step function, returning observation, reward, termination, truncation, info.

        The observation is for the next agent (used to determine the next action), while the remaining
        items are for the agent that just acted (used to understand what just happened).
        """
        current_agent = self.agent_selection

        super().step(action)

        next_agent = self.agent_selection
        return (
            self.observe(next_agent),
            self._cumulative_rewards[current_agent],
            self.terminations[current_agent],
            self.truncations[current_agent],
            self.infos[current_agent],
        )

    def observe(self, agent):
        """Return only raw observation, removing action mask."""
        return super().observe(agent)["observation"]

    def action_mask(self, agent):
        """Separate function used in order to access the action mask."""
        return super().observe(agent)["action_mask"]


import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed


def make_env_multiprocessed(env_fn, env_id: str, rank: int, seed: int = 0):
    """
    Utility function for multiprocessed env.

    :param env_id: the environment ID
    :param num_env: the number of environments you wish to have in subprocesses
    :param seed: the initial seed for RNG
    :param rank: index of the subprocess
    """

    def _init():
        gym.register(env_id, env_fn)
        env = gym.make(env_id, render_mode="human")
        env.reset(seed=seed + rank)
        return env

    set_random_seed(seed)
    return _init


# TODO: frame_stack_v1 isn't default???
def make_env(num_cpus: int, render=True):
    """
    Factory: AEC → Parallel → VecEnv pipeline.
    SuperSuit's pettingzoo_env_to_vec_env_v1 requires a PARALLEL env.
    """
    env = CustomEnvironment(rendering=render)
    env = ss.black_death_v3(env)

    # def mask_fn(env):
    #     return env.action_mask()

    # env = SB3ActionMaskWrapper(env)
    # env.reset(seed=42)  # Must call reset() in order to re-define the spaces
    # env = ActionMasker(env, mask_fn)
    env = ss.pettingzoo_env_to_vec_env_v1(env)

    def env_fn(**kwargs):
        return env

    # env_id = "CustomEnvironmentv1"  # It is best practice to have a space name and version number.
    # env = SubprocVecEnv(
    #     [make_env_multiprocessed(env_fn, env_id, i) for i in range(num_cpus)]
    # )
    # env.metadata["is_parallelizable"] = True
    # env = to_parallel(env)

    env = fixed_concat_vec_envs_v1(
        env, num_vec_envs=num_cpus, num_cpus=num_cpus, base_class="stable_baselines3"
    )

    return env


import supersuit as ss
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.tune.registry import register_env


def _make_rllib_env(config):
    base = CustomEnvironment()
    wrapped = ParallelPettingZooEnv(base)
    wrapped._agent_ids = set(getattr(base, "possible_agents", []))
    return wrapped


ENV_NAME = "custom_env"
register_env(ENV_NAME, _make_rllib_env)


def find_latest_checkpoint(checkpoint_dir, prefix):
    """Находит последний сохранённый чекпоинт по номеру шага."""
    pattern = os.path.join(checkpoint_dir, f"{prefix}_*_steps.zip")
    files = glob.glob(pattern)
    if not files:
        return None, 0

    # Извлекаем номер шага из имени файла и берём максимальный
    def extract_step(path):
        name = os.path.basename(path)  # ppo_v1_300000_steps.zip
        parts = name.replace(".zip", "").split("_")
        for part in reversed(parts):
            if part.isdigit():
                return int(part)
        return 0

    latest = max(files, key=extract_step)
    steps_done = extract_step(latest)
    return latest, steps_done


# test make_env
if __name__ == "__main__":
    make_env(1)
