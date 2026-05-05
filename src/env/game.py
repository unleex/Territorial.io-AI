import random
from env.countryClass import country
from env.gameAI import runAi


# TODO: is cycle end even handled??
class Game:
    def __init__(self, n_players=8, grid_rows=80, grid_columns=80):

        self.countryColors = [
            "#ffffffff",
            "#ffff00",
            "#00ff00",
            "#00ffff",
            "#ff0000",
            "#A0A0A0",
            "#ff00ff",
            "#fc9105",
        ]
        self.ticks = 0
        self.gameOver = False
        self.n_players = n_players
        self.id_to_country: dict[int, country] = {}
        self.board: list[list[int]] = []
        self.n_grid_rows = grid_rows
        self.n_grid_columns = grid_columns
        for i in range(self.n_grid_rows):
            self.board.append([-1] * self.n_grid_columns)

        for i in range(self.n_players):
            row = random.randint(0, self.n_grid_rows - 1)
            col = random.randint(0, self.n_grid_columns - 1)
            # If the tile is already occupied, keep rolling random
            while self.board[row][col] != -1:
                row = random.randint(0, self.n_grid_rows - 1)
                col = random.randint(0, self.n_grid_columns - 1)
            self.id_to_country[i] = country(
                i, self.countryColors[i], str(i), attacks=dict()
            )
            self.board[row][col] = i

    def tick(self):
        self.ticks += 1
        L = []
        for key in self.id_to_country:
            L.append(key)
        for key in L:
            if self.id_to_country[key].size <= 0:
                if key == 0:
                    self.gameOver = True
                    break
                del self.id_to_country[key]
                continue
            self.id_to_country[key].updateMoney()
            if key != 0:
                runAi(self, self.id_to_country[key])
            self.id_to_country[key].incrementAttacks(self)
