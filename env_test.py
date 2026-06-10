import numpy as np
from prepare_env import make_env
from time import perf_counter

N = 100
env = make_env(render=True)
env.reset()
times = []
for _ in range(N):
    start = perf_counter()
    env.step(env.action_space.sample())
    taken = perf_counter() - start
    print(taken)
    env.render()
    times.append(taken)

print("Mean:", np.mean(times))
