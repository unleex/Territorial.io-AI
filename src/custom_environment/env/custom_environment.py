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


# TODO multiple agents. For simplicity, now let's fit to single agent fitting to algorithmic baseline
class CustomEnvironment(ParallelEnv):
    metadata = {
        "name": "custom_environment_v0",
    }

    def _prepare(self):
        self.game = Game()
        neutral_original_id = -1
        neutral_perm_index = 0
        agent_perm_index = 1

        # Maps original ids in [-1, n_players - 1] to [0, n_players].
        # Array index for original_id is (original_id + 1).
        self.id_permutation = np.empty(self.game.n_players + 1, dtype=int)
        self.id_permutation[neutral_original_id + 1] = neutral_perm_index
        self.id_permutation[self.agent_id + 1] = agent_perm_index

        others = [
            player_id + 1
            for player_id in range(self.game.n_players)
            if player_id != self.agent_id
        ]
        np.random.shuffle(others)
        for perm_index, other_player_array_idx in enumerate(others, start=2):
            self.id_permutation[other_player_array_idx] = perm_index

        self.reverse_id_permutation = np.empty(self.game.n_players + 1, dtype=int)
        for original_id, permuted_id in enumerate(self.id_permutation):
            self.reverse_id_permutation[permuted_id] = original_id

    def permute_id(self, original_id: int) -> int:
        return int(self.id_permutation[original_id + 1])

    def unpermute_id(self, permuted_id: int) -> int:
        return int(self.reverse_id_permutation[permuted_id] - 1)

    def __init__(self, rendering=True):
        """
        ticks_delta: int (default = 1) how many game ticks to run between agent's decisions'
        """
        super().__init__()
        self.game: Game
        self.ticks_delta = 5
        self.id_permutation: np.ndarray
        self.reverse_id_permutation: np.ndarray
        self.agent_id = 0
        self.render_mode = None
        self._prepare()
        self.rendering = rendering
        if self.rendering:
            self.renderer = GameRenderer(self.game.countryColors)
        self.possible_agents = [0]
        self.agents = self.possible_agents[:]
        self.terminations = {0: False}
        self.truncations = {0: False}
        self.reward = 0.0
        # map of one-hot vectors (each player + neutral)
        self.n_board_channels = self.game.n_players + 1
        self.n_stats_channels = self.game.n_players * 2
        obs_shape = (
            self.n_board_channels + self.n_stats_channels,
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
        for original_id in range(-1, self.game.n_players):
            permuted_board[board == original_id] = self.permute_id(original_id)

        num_channels = self.game.n_players + 1
        one_hot = np.eye(num_channels, dtype=np.float32)[permuted_board]
        # TODO add per-player stats like balance?
        # TODO definitely add cycle data

        stats = np.zeros(self.n_stats_channels, dtype=np.float32)
        for perm_idx in range(1, self.game.n_players + 1):
            original_id = self.unpermute_id(perm_idx)
            stat_idx = perm_idx - 1
            if original_id in self.game.id_to_country:
                c = self.game.id_to_country[original_id]
                max_money = max(c.size * 1500, 1)
                stats[stat_idx] = float(np.clip(c.money / max_money, 0.0, 1.0))
                stats[self.game.n_players + stat_idx] = c.size / self.game.n_grid_rows / self.game.n_grid_columns 
            # else: player is dead → stays 0.0
            else:
                stats[stat_idx] = 0.0
                stats[self.game.n_players + stat_idx] = 0.0
 
        # Broadcast (n_stats,) → (H, W, n_stats) constant channels
        stats_channels = np.broadcast_to(
            stats[np.newaxis, np.newaxis, :],
            (self.game.n_grid_rows, self.game.n_grid_columns, self.n_stats_channels),
        ).astype(np.float32)
 
        full_obs = np.concatenate([one_hot, stats_channels], axis=2)  # (H, W, C)
        return full_obs.transpose(2, 0, 1)  # (C, H, W)

    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ):
        self.agents = self.possible_agents[:]
        self._prepare()
        if self.rendering:
            self.renderer.reset()
        return {0: self.observe()}, mock_info

    def step(self, action: dict[int, Any]):
        if not self.agents:
            return {}, {}, {}, {}, {}
        for _ in range(self.ticks_delta):
            self.game.tick()
        # TODO mask out self-attack
        old_player_size = self.game.id_to_country[self.agent_id].size
        # 0 is neutral, others are agents
        target = action[0][0]
        commited_bin = action[0][1]
        commited = (
            self.game.id_to_country[self.agent_id].money * commited_bin / 10.0
        )  # Convert 0..10 to 0.0..1.0
        target = self.unpermute_id(target)
        self.game.id_to_country[self.agent_id].attackInit(self.game, target, commited)
        # else target == -1 => wait
        obs = {0: self.observe(self.agents[0])}
        reward = {0: (self.game.id_to_country[self.agent_id].size - old_player_size) / (self.game.n_grid_rows * self.game.n_grid_columns)}
        self.terminations[0] = (
            self.game.id_to_country[self.agent_id].size == 0
            or len(self.game.id_to_country) == 1
        )
        if self.terminations[0]:
            if self.game.id_to_country[self.agent_id].size > 0:
                reward[0] += 10
            else:
                reward[0] -= 10 * ((self.game.n_grid_rows * self.game.n_grid_columns) - self.game.id_to_country[0].size) / (self.game.n_grid_rows * self.game.n_grid_columns)  # Defeat penalty normalized

        if self.terminations[0]:
            self.agents = []
        return obs, reward, self.terminations, self.truncations, mock_info

    def render(self, targeted_player=-1, commited=0, **kwargs):
        self.renderer.update(
            np.array(self.game.board),
            self.game.n_players,
            targeted_player,
            commited,
            **kwargs,
        )

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]
