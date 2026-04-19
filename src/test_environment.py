from pettingzoo.test import parallel_api_test
from custom_environment.custom_environment_v0 import CustomEnvironment

if __name__ == "__main__":
    env = CustomEnvironment()
    parallel_api_test(env, num_cycles=1_000_000)

    # env = CustomActionMaskedEnvironment()
    # parallel_api_test(env, num_cycles=1_000_000)
