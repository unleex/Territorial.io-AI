from custom_environment.custom_environment_v0 import CustomEnvironment
import glob
import supersuit as ss
import os


def make_env():
    """
    Factory: AEC → Parallel → VecEnv pipeline.
    SuperSuit's pettingzoo_env_to_vec_env_v1 requires a PARALLEL env.
    """
    env = CustomEnvironment()

    env = ss.black_death_v3(env)
    env = ss.pettingzoo_env_to_vec_env_v1(env)

    env = ss.concat_vec_envs_v1(
        env, num_vec_envs=4, num_cpus=0, base_class="stable_baselines3"
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
