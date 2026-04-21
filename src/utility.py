from custom_environment.custom_environment_v0 import CustomEnvironment
import glob
import supersuit as ss
import os

from supersuit.vector import MakeCPUAsyncConstructor
from supersuit.vector.vector_constructors import vec_env_args


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


def make_env():
    """
    Factory: AEC → Parallel → VecEnv pipeline.
    SuperSuit's pettingzoo_env_to_vec_env_v1 requires a PARALLEL env.
    """
    env = CustomEnvironment()

    env = ss.black_death_v3(env)
    env = ss.pettingzoo_env_to_vec_env_v1(env)

    env = fixed_concat_vec_envs_v1(
        env, num_vec_envs=8, num_cpus=8, base_class="stable_baselines3"
    )

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
