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
        attacks="Had to change cause of aliasing",
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
    # attacking country
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
        if id != -1 and (
            id not in game.id_to_country or game.id_to_country[id].size == 0
        ):
            self.attacks[id] = None
            return
        neighbours = 0
        density = 5.0
        money = self.attacks[id][0]
        if id != -1:
            density = game.id_to_country[id].money / game.id_to_country[id].size
        for i in range(len(game.board)):
            for j in range(len(game.board[0])):
                # If current tile == id and borders the attacking country
                if game.board[i][j] == id and self.isNeighbour(game, id, i, j):
                    neighbours += 1
                    game.board[i][j] = -2
        # If committed troops are insufficient for conquering
        if neighbours * density > money:
            # Reset game board to original state
            for i in range(len(game.board)):
                for j in range(len(game.board[0])):
                    if game.board[i][j] == -2:
                        game.board[i][j] = id
            if id != -1:
                game.id_to_country[id].money -= money
            else:
                self.money += money
            self.attacks[id] = None
            return
        # If there are no more cells that can be conquered
        if neighbours == 0:
            self.attacks[id] = None
            return
        for i in range(len(game.board)):
            for j in range(len(game.board[0])):
                # If current tile == id and borders the attacking country
                if game.board[i][j] == -2:
                    game.board[i][j] = self.id
        self.size += neighbours
        if id != -1:
            game.id_to_country[id].size -= neighbours
            game.id_to_country[id].money -= roundHalfUp(neighbours * density)
        self.attacks[id] = (
            money - roundHalfUp(neighbours * density),
            self.attacks[id][1],
        )

    def incrementAttacks(self, game):
        toRemove = []
        for key in self.attacks:
            self.incrementAttack(game, key)
            if self.attacks[key] == None:
                toRemove.append(key)

        for i in toRemove:
            del self.attacks[i]
