import numpy as np
import random
from game.countryClass import country
from game.gameAI import runAi


class Game:
    def __init__(self, n_players=8, n_agents=8, grid_rows=80, grid_columns=80):
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
        self.agents = list(range(n_agents))
        self.ticks = 0
        self.gameOver = False
        self.n_players = n_players
        self.n_agents = n_agents
        self.id_to_country: dict[int, country] = {}
        board = []
        self.n_grid_rows = grid_rows
        self.n_grid_columns = grid_columns
        for i in range(self.n_grid_rows):
            board.append([-1] * self.n_grid_columns)
        self.board = np.array(board, dtype=np.int8)
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
            if key not in self.agents:
                runAi(self, self.id_to_country[key])
            self.id_to_country[key].incrementAttacks(self)
