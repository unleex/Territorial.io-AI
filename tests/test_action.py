from environment import CustomEnvironment
from game.gameFuncs import findNeighbours
import time

env = CustomEnvironment()
env.step({0: [0, 10]})
assert env.game.id_to_country[0].size > 0
print("Neutral territory is attacked. Passed")
env = CustomEnvironment()
neigh = findNeighbours(env.game, 0)
while -1 in neigh:
    env.step({0: [0, 10]})
    neigh = findNeighbours(env.game, 0)
neighbor = list(neigh.items())[0][0]
old_neigh_size = env.game.id_to_country[neighbor].size
while env.agents:
    env.step({0: [env.permute_id(neighbor), 10]})
    env.render()
    time.sleep(0.05)
assert env.game.id_to_country[neighbor].size < old_neigh_size
