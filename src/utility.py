from custom_environment.custom_environment_v0 import CustomEnvironment
import glob
import supersuit as ss
import os
import numpy as np
from stable_baselines3.common.vec_env import VecEnvWrapper
from gymnasium import spaces
from supersuit.vector import MakeCPUAsyncConstructor
from supersuit.vector.vector_constructors import vec_env_args


class SplitObsWrapper(VecEnvWrapper):
    """
    Splits the flat Box observation produced by CustomEnvironment into a Dict
    observation that MultiInputPolicy expects:
 
        obs["board"]  shape (N, n_board_ch, H, W)  — spatial one-hot channels
        obs["stats"]  shape (N, n_stats_ch)        — per-player scalar stats
 
    CustomEnvironment encodes scalar stats as "constant" channels: every pixel
    in such a channel holds the same value, so the scalar is recovered by
    reading obs[:, channel, 0, 0].
 
    This wrapper is inserted AFTER the full supersuit pipeline (black_death,
    pettingzoo_env_to_vec_env, concat_vec_envs) because black_death_v3
    requires a plain Box observation space and cannot accept Dict.
    """
 
    def __init__(self, venv, n_board_channels: int, n_stats_channels: int):
        self.n_board_ch = n_board_channels
        self.n_stats_ch = n_stats_channels
 
        _, H, W = venv.observation_space.shape
 
        dict_obs_space = spaces.Dict({
            "board": spaces.Box(
                low=0, high=1,
                shape=(n_board_channels, H, W),
                dtype=np.float32,
            ),
            "stats": spaces.Box(
                low=0, high=1,
                shape=(n_stats_channels,),
                dtype=np.float32,
            ),
        })
        super().__init__(venv, observation_space=dict_obs_space)
 
    def _split(self, obs: np.ndarray) -> dict:
        # obs shape: (N, n_board_ch + n_stats_ch, H, W)
        board = obs[:, :self.n_board_ch]
        # Each stats channel is constant across H×W, so read top-left pixel
        stats = obs[:, self.n_board_ch:, 0, 0]
        return {"board": board, "stats": stats}
 
    def reset(self):
        obs = self.venv.reset()
        return self._split(obs)
 
    def step_wait(self):
        obs, reward, done, info = self.venv.step_wait()
        return self._split(obs), reward, done, info

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


def make_env(num_cpus: int, render=True):
    """
    Factory: AEC → Parallel → VecEnv pipeline.
    SuperSuit's pettingzoo_env_to_vec_env_v1 requires a PARALLEL env.
    """
    env = CustomEnvironment(rendering=render)
    n_board_ch = env.n_board_channels
    n_stats_ch = env.n_stats_channels
 
    env = ss.black_death_v3(env)
    env = ss.pettingzoo_env_to_vec_env_v1(env)
 
    env = fixed_concat_vec_envs_v1(
        env, num_vec_envs=num_cpus, num_cpus=num_cpus, base_class="stable_baselines3"
    )
 
    env = SplitObsWrapper(env, n_board_channels=n_board_ch, n_stats_channels=n_stats_ch)

    return env


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
