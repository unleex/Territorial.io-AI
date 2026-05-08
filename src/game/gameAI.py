import random
from typing import TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from game.game import Game
    from game.countryClass import country


# target = id of country that the function is finding neighbours for
def findNeighbours(game: "Game", target):
    """
    Returns a list of unique player IDs that share a border with the given player_id.
    """
    grid = game.board
    # 1. Mask of current player's territory
    mask = grid == target

    # 2. Find all cells adjacent to the mask (UP, DOWN, LEFT, RIGHT)
    # We create a combined mask of all neighbors
    adjacent_mask = np.zeros_like(mask)
    adjacent_mask[:-1, :] |= mask[1:, :]  # From below
    adjacent_mask[1:, :] |= mask[:-1, :]  # From above
    adjacent_mask[:, :-1] |= mask[:, 1:]  # From right
    adjacent_mask[:, 1:] |= mask[:, :-1]  # From left

    # 3. Filter: We only care about neighbors that are NOT the player themselves
    neighbor_pixels_mask = adjacent_mask & ~mask

    # 4. Extract the actual values (IDs) from the grid at those locations
    neighbor_ids = grid[neighbor_pixels_mask]
    ids, counts = np.unique(neighbor_ids, return_counts=True)
    return dict(zip(ids, counts))


def runAi(game: "Game", agent: "country"):
    # Find weakest neighbour
    d = findNeighbours(game, agent.id)
    # If there is still empty space and agent has some threshold of money
    if -1 in d:
        commit = int(5.0 * d[-1]) + 1
        if (
            agent.money > random.randint(agent.size * 10, agent.size * 30)
            and agent.money > commit
        ):
            agent.attackInit(game, -1, commit)  # Exact amount for one layer
        return

    # Find smallest neighbour
    smallest = None
    for i in d:
        if (
            smallest is None
            or game.id_to_country[i].money < game.id_to_country[smallest].money
        ):
            smallest = i
    ...
    # If the smallest neighbour is signifcantly smaller
    density = game.id_to_country[smallest].money / game.id_to_country[smallest].size
    if (
        game.id_to_country[smallest].money < agent.money * agent.aggro
        and int(d[smallest] * density) + 1 <= agent.money
    ):
        agent.attackInit(game, smallest, int(d[smallest] * density) + 1)
        return

    # if attack threshold not met
    if agent.money / agent.size / 1000 < agent.threshold:
        return

    # If the difference between the to countries isn't too big, or growth has stopped
    if (
        game.id_to_country[smallest].money / (agent.money * agent.aggro) <= agent.tooBig
        or agent.money >= agent.size * 1000
    ):
        agent.attackInit(game, smallest, int(agent.money * agent.aggro))
