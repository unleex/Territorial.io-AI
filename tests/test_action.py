from environment import CustomEnvironment
from game.gameFuncs import findNeighbours

env = CustomEnvironment(rendering=True)
env.step({0: [0, 10]})
assert env.game.id_to_country[0].size > 0
print("Neutral territory is attacked. Passed")
env.reset()
neigh = findNeighbours(env.game, 0)
while not neigh or len(neigh) == 1 and -1 in neigh:
    env.step({0: [env.permute_id(-1), 10]})
    neigh = findNeighbours(env.game, 0)
neighbors = list(neigh.keys())
if -1 in neighbors:
    neighbors.remove(-1)
neighbor = int(neighbors[0])
old_neigh_size = env.game.id_to_country[neighbor].size
env.step({0: [env.permute_id(neighbor), 10]})
assert env.game.id_to_country[neighbor].size < old_neigh_size
print("Enemy territory is attacked. Passed")
