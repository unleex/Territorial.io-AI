import random
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from custom_environment.env.game import Game
    from custom_environment.env.countryClass import country


# target = id of country that the function is finding neighbours for
def findNeighbours(game: "Game", target):
    d = dict()
    for i in range(len(game.board)):
        for j in range(len(game.board[0])):
            if (
                game.board[i][j] != target
                and not i - 1 < 0
                and game.board[i - 1][j] == target
                or not i + 1 >= len(game.board)
                and game.board[i + 1][j] == target
                or not j - 1 < 0
                and game.board[i][j - 1] == target
                or not j + 1 >= len(game.board[0])
                and game.board[i][j + 1] == target
            ):
                d[game.board[i][j]] = d.get(game.board[i][j], 0) + 1
    return d


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
            smallest == None
            or game.id_to_country[i].money < game.id_to_country[smallest].money
        ):
            smallest = i

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
