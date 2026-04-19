import numpy as np
from pettingzoo import ParallelEnv
from game import Game
import actions as action_types


# TODO: action space is tuple of Box and Discrete
class CustomEnvironment(ParallelEnv):
    metadata = {
        "name": "custom_environment_v0",
    }

    def __init__(self):
        self.game = Game()

    def reset(self, seed=None, options=None):
        self.game = Game()

    def step(self, actions: dict[int, int]):
        for player_id, (action, commited) in actions:
            if action > 0:  # attacked smb
                self.game.id_to_country[player_id].attackInit(None, action, commited)
            # else action == 0 => wait

    def render(self):
        return np.array(self.game.board)

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]
