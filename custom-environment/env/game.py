import random
from countryClass import country


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
        self.n_players = n_players
        self.id_to_country: dict[int, country] = {}
        self.board = []
        self.n_grid_rows = grid_rows
        self.n_grid_columns = grid_columns
        for i in range(self.n_grid_rows):
            self.board.append([-1] * self.n_grid_columns)

        for i in range(self.n_players):
            row = random.randint(0, self.n_grid_rows - 1)
            col = random.randint(0, self.n_grid_columns - 1)
            # If the tile is already occupied, keep rolling random
            while self.board[row][col] == 0:
                row = random.randint(0, self.n_grid_rows - 1)
                col = random.randint(0, self.n_grid_columns - 1)
            temp = dict()
            self.id_to_country[i] = country(
                i, self.countryColors[i], str(i), attacks=temp
            )
            self.board[row][col] = i
