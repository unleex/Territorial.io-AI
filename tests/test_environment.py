from pettingzoo.test import parallel_api_test
from environment import CustomEnvironment

# XXX: in 25th line of parallel_api_test.py, replace the whole if-statement's body with
# return game.action_space(agent).sample(mask=agent_obs["action_mask"])
# that prevents action mask bug
if __name__ == "__main__":
    env = CustomEnvironment()
    parallel_api_test(env, num_cycles=1_000_000)

    # env = CustomActionMaskedEnvironment()
    # parallel_api_test(env, num_cycles=1_000_000)
