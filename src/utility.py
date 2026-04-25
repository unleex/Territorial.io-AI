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


# XXX: replace the corresponding method of MarkovVectorEnv with thing below
"""
def concat_obs(self, obs_dict):
    obs_list = []
    for i, agent in enumerate(self.par_env.possible_agents):
        if agent not in obs_dict:
            raise AssertionError(
                "environment has agent death. Not allowed for pettingzoo_env_to_vec_env_v1 unless black_death is True"
            )
        if isinstance(obs_dict[agent], dict) and "action_mask" in obs_dict[agent]:
            try:
                obs_list.append(obs_dict[agent]["observation"])
            except Exception as e:
                print("THIS IS INSIDE MY WORKAROUND FOR ACTION MASKING!!!")
                raise e
            continue
        obs_list.append(obs_dict[agent])

    return concatenate(
        self.observation_space,
        obs_list,
        create_empty_array(self.observation_space, self.num_envs),
    )
"""


# TODO: frame_stack_v1 isn't default???
def make_env(num_cpus: int, render=True):
    """
    Factory: AEC → Parallel → VecEnv pipeline.
    SuperSuit's pettingzoo_env_to_vec_env_v1 requires a PARALLEL env.
    """
    env = CustomEnvironment(rendering=render)

    env = ss.black_death_v3(env)
    env = ss.pettingzoo_env_to_vec_env_v1(env)

    env = fixed_concat_vec_envs_v1(
        env, num_vec_envs=num_cpus, num_cpus=num_cpus, base_class="stable_baselines3"
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
