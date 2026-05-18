import numpy as np
import math
import decimal
import random
from game.gameAI import *
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from game.game import Game


def roundHalfUp(d):  # helper-fn
    # Round to nearest with ties going away from zero.
    rounding = decimal.ROUND_HALF_UP
    # See other rounding options here:
    # https://docs.python.org/3/library/decimal.html#rounding-modes
    return int(decimal.Decimal(d).to_integral_value(rounding=rounding))


class country:
    def __init__(
        self,
        id,
        color,
        name,
        money=100,
        size=1,
        attackProportion=0.3,
        attacks: dict = "Had to change cause of aliasing",
    ):
        self.id = id  # corresponding int on board
        self.color = color  # fill color on board
        self.name = name
        self.money = money
        self.size = size
        self.attackProportion = attackProportion  # Must be between 0 and 1 inclusive
        self.growthRate = 0.0
        self.attacks = attacks  # (money, original money)

        # For the bots
        self.threshold = random.randint(20, 40) / 100  # What threshold bots attack at
        self.tooBig = random.randint(
            3, 10
        )  # Bots won't attack if the difference is too big
        self.aggro = random.randint(10, 30) / 100  # How much bots attack with

        # For drawing name
        self.ratio = 2.8 / max(
            len(self.name), len(str(self.money))
        )  # ratio of height/width
        self.maxWidth = 0
        self.row = -1
        self.col = -1

    # Logistic equation: f(x) = L/(1+e**-k(x-a))
    # x = ln(L/f(x)-1)/-k + a
    def updateMoney(self):
        # find current position on curve
        L = self.size * 1000
        y = max(self.money, 1.000001)
        k = 0.05
        a = 150
        x = 0
        if y < L:
            x = math.log(L / y - 1) / -k + a
            x += 1
            self.growthRate = L / (1 + math.exp(-k * (x - a))) / max(self.money, 1)
            self.money = roundHalfUp(L / (1 + math.exp(-k * (x - a))))
        else:
            self.growthRate = 1.0
        self.money += self.size
        self.money = min(self.size * 1500, self.money)

    # Returns true if cell is in country being attacked and is neighbor of
    # attacking country (unused function)
    def isNeighbour(self, game: "Game", id, i, j):
        if (
            game.board[i][j] == id
            and not i - 1 < 0
            and game.board[i - 1][j] == self.id
            or not i + 1 >= len(game.board)
            and game.board[i + 1][j] == self.id
            or not j - 1 < 0
            and game.board[i][j - 1] == self.id
            or not j + 1 >= len(game.board[0])
            and game.board[i][j + 1] == self.id
        ):
            return True
        return False

    # initializing queue for dfs
    def attackInit(self, game, id, committed):
        if committed == 0:
            return
        if id not in findNeighbours(game, self.id):
            return
        self.money -= committed
        # If the country is already being attacked, add committed troops to current attack
        if id not in self.attacks:
            self.attacks[id] = (committed, committed)
        else:
            temp = self.attacks[id]
            self.attacks[id] = (temp[0] + committed, temp[1] + committed)

    def incrementAttack(self, game: "Game", id):
        if id != -1:
            target_country = game.id_to_country.get(id)
            if not target_country or target_country.size == 0:
                self.attacks[id] = None
                return

        # 2. Get the Frontier (Pixels of 'target_id' touching 'self.id')
        attacker_mask = game.board == self.id
        target_mask = game.board == id

        # Shift attacker mask to find adjacent cells
        adj_to_attacker = np.zeros_like(attacker_mask)
        adj_to_attacker[:-1, :] |= attacker_mask[1:, :]
        adj_to_attacker[1:, :] |= attacker_mask[:-1, :]
        adj_to_attacker[:, :-1] |= attacker_mask[:, 1:]
        adj_to_attacker[:, 1:] |= attacker_mask[:, :-1]

        # The conquerable pixels: adjacent to me AND belonging to target
        conquer_mask = adj_to_attacker & target_mask
        num_neighbours = np.count_nonzero(conquer_mask)

        if num_neighbours == 0:
            self.attacks[id] = None
            return

        money_committed = self.attacks[id][0]
        if id != -1:
            density = target_country.money / target_country.size
        else:
            density = 5.0  # Neutral land density

        total_cost = num_neighbours * density

        # If committed troops are insufficient for conquering
        if total_cost > money_committed:
            # Failure: Penalty but no land gain
            if id != -1:
                target_country.money -= money_committed
            else:
                self.money += money_committed
            self.attacks[id] = None
            return
        game.board[conquer_mask] = self.id
        self.size += conquer_mask.sum()
        if id != -1:
            game.id_to_country[id].size -= num_neighbours
            game.id_to_country[id].money -= roundHalfUp(num_neighbours * density)
        self.attacks[id] = (
            money_committed - roundHalfUp(num_neighbours * density),
            self.attacks[id][1],
        )

    def incrementAttacks(self, game):
        toRemove = []
        for key in self.attacks:
            self.incrementAttack(game, key)
            if self.attacks[key] is None:
                toRemove.append(key)

        for i in toRemove:
            del self.attacks[i]
