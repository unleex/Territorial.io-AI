from multiprocessing.spawn import prepare

from render import GameRenderer
import numpy as np
import numpy.typing as npt
from typing import Dict, Any, Tuple, List, Optional
from pettingzoo.utils.env import AgentID, ObsDict, ActionDict
from pettingzoo import ParallelEnv
from custom_environment.env.game import Game
from gymnasium import spaces

mock_info = {0: {}}
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
import numpy as np


# TODO multiple agents. for simplicity, now let's fit to single agent fitting to algorithmic baseline
class CustomEnvironment(ParallelEnv):
    metadata = {
        "name": "custom_environment_v0",
    }

    def _prepare(self):
        self.game = Game()
        self.id_permutation = np.full(self.game.n_players, -1)
        self.id_permutation[self.agent_id] = 0
        others = [i for i in range(self.game.n_players) if i != self.agent_id]
        np.random.shuffle(others)
        for i, other_player_id in enumerate(others, start=1):
            self.id_permutation[other_player_id] = i
        self.reverse_id_permutation = np.empty(self.game.n_players, dtype=int)
        for original_id, permuted_id in enumerate(self.id_permutation):
            self.reverse_id_permutation[permuted_id] = original_id

    def __init__(self):
        """
        ticks_delta: int (default = 1) how many game ticks to run between agent's decisions'
        """
        super().__init__()
        self.game: Game
        self.ticks_delta = 1  # FIXME: too low. find optimal
        self.id_permutation: np.ndarray
        self.reverse_id_permutation: np.ndarray
        self.agent_id = 0
        self.render_mode = None
        self._prepare()
        self.renderer = GameRenderer(self.game.countryColors)
        self.possible_agents = [0]
        self.agents = self.possible_agents[:]
        self.terminations = {0: False}
        self.truncations = {0: False}
        self.reward = 0.0
        # map of one-hot vectors (each player + neutral)
        obs_shape = (
            self.game.n_players + 1,
            self.game.n_grid_rows,
            self.game.n_grid_columns,
        )

        self.observation_spaces = {
            0: spaces.Box(
                low=0,
                high=1,
                shape=obs_shape,
                dtype=np.float32,
            )
        }
        self.action_spaces = {
            0: spaces.MultiDiscrete([self.game.n_players + 1, 11])
        }  # who to attack (or stall) + amount of troops (0%, 10%, 20%, ...)

    def observe(self, _=None):
        board = np.array(self.game.board)
        permuted_board = np.full(board.shape, -1)

        for original_id, new_id in enumerate(self.id_permutation):
            permuted_board[board == original_id] = new_id
        # shift to range [0, n_players]
        shifted = (permuted_board + 1).astype(int)

        num_channels = self.game.n_players + 1
        one_hot = np.eye(num_channels, dtype=np.float32)[shifted]
        # TODO add per-player stats like balance?
        # TODO definitely add cycle data
        # Transpose to (Channels, Height, Width) for PyTorch/CNN compatibility
        return one_hot.transpose(2, 0, 1)

    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ):
        self.agents = self.possible_agents[:]
        self._prepare()
        self.renderer.reset()
        return {0: self.observe()}, mock_info

    def step(self, action: dict[int, Any]):
        if not self.agents:
            return {}, {}, {}, {}, {}
        for _ in range(self.ticks_delta):
            self.game.tick()
        # TODO mask out self-attack
        old_player_size = self.game.id_to_country[self.agent_id].size
        target_channel = action[0][0]
        commited_bin = action[0][1]
        commited = commited_bin / 100.0  # Convert 0..10 to 0.0..1.0

        if target_channel == 0:  # neutral land
            target = -1
        else:
            target = self.reverse_id_permutation[target_channel - 1]

        if target >= 0:  # attacked smb
            self.game.id_to_country[self.agent_id].attackInit(
                self.game, target, commited
            )
        # else target == -1 => wait
        obs = {0: self.observe(self.agents[0])}
        reward = {0: self.game.id_to_country[self.agent_id].size - old_player_size}
        self.terminations[0] = (
            self.game.id_to_country[self.agent_id].size == 0
            or len(self.game.id_to_country) == 1
        )
        if self.terminations[0]:
            self.agents = []
        return obs, reward, self.terminations, self.truncations, mock_info

    def render(self, targeted_player=-1, commited=0):
        self.renderer.update(
            np.array(self.game.board), self.game.n_players, targeted_player, commited
        )

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]
