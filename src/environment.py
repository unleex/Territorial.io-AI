from collections import deque
from copy import deepcopy

from render import GameRenderer
import numpy as np
from typing import Dict, Any, Optional
from pettingzoo import ParallelEnv
from game.game import Game
from gymnasium import spaces
from game.gameFuncs import findNeighbours

mock_info = {0: {}}


# TODO multiple agents. For simplicity, now let's fit single agent to algorithmic baseline
class CustomEnvironment(ParallelEnv):
    metadata = {
        "name": "custom_environment_v0",
    }

    def permute_id(self, original_id: int) -> int:
        return int(self.id_permutation[original_id + 1])

    def unpermute_id(self, permuted_id: int) -> int:
        return int(self.reverse_id_permutation[permuted_id] - 1)

    def __init__(self, rendering=True):
        """
        ticks_delta: int (default = 1) how many game ticks to run between agent's decisions
        """
        super().__init__()
        self.game: Game
        self.ticks_delta = 5
        self.id_permutation: np.ndarray
        self.reverse_id_permutation: np.ndarray
        self.agent_id = 0
        self.render_mode = None
        self.rendering = rendering
        self.possible_agents = [0]
        self.agents = self.possible_agents[:]
        self.terminations = {0: False}
        self.truncations = {0: False}
        self.reward = 0.0
        self.obs_stack_size = 4
        self.map_obs_deque: deque[np.ndarray] = deque(maxlen=self.obs_stack_size)
        self._prepare()
        # map of one-hot vectors (each player + neutral)
        obs_shape = (
            (self.game.n_players + 1) * self.obs_stack_size,
            self.game.n_grid_rows,
            self.game.n_grid_columns,
        )
        self.n_stats = self.game.n_players * 2
        stats_shape = (self.n_stats,)

        self.n_commit_bins = 11

        self.action_spaces = {
            0: spaces.MultiDiscrete([self.game.n_players + 1, self.n_commit_bins])
        }  # who to attack (or stall) + amount of troops (0%, 10%, 20%, ...)
        self.observation_spaces = {
            0: spaces.Dict(
                {
                    "observations": spaces.Box(
                        low=-1,
                        high=1,
                        shape=obs_shape,
                        dtype=np.float32,
                    ),
                    "stats": spaces.Box(
                        low=0,
                        high=1,
                        shape=stats_shape,
                        dtype=np.float32,
                    ),
                    "action_mask": spaces.Box(
                        low=0.0,
                        high=1.0,
                        shape=(self.game.n_players + 1 + self.n_commit_bins,),
                        dtype=np.float32,
                    ),
                }
            )
        }
        self.saved_stats = np.zeros(shape=stats_shape, dtype=np.float32)

    def _prepare(self):
        self.game = Game()
        if self.rendering:
            self.renderer = GameRenderer(self.game.countryColors)
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
        unstacked_obs_shape = (
            (self.game.n_players + 1),  # no self.obs_stack_size, we need raw one here
            self.game.n_grid_rows,
            self.game.n_grid_columns,
        )
        for _ in range(self.obs_stack_size):
            self.map_obs_deque.append(
                np.zeros(shape=unstacked_obs_shape, dtype=np.float32)
            )

    def _get_observation_frame(self, _=None):
        board = np.array(self.game.board)
        permuted_board = np.full(board.shape, -1)
        for original_id in range(-1, self.game.n_players):
            permuted_board[board == original_id] = self.permute_id(original_id)

        num_channels = self.game.n_players + 1
        one_hot = np.eye(num_channels, dtype=np.float32)[permuted_board]
        stats = np.zeros(self.n_stats, dtype=np.float32)
        for perm_idx in range(1, self.game.n_players + 1):
            original_id = self.unpermute_id(perm_idx)
            stat_idx = perm_idx - 1

            if original_id in self.game.id_to_country:
                c = self.game.id_to_country[original_id]
                stats[stat_idx] = (
                    c.money / 1500 / self.game.n_grid_rows / self.game.n_grid_columns
                )
                stats[self.game.n_players + stat_idx] = (
                    c.size / self.game.n_grid_rows / self.game.n_grid_columns
                )
            # else: player is dead → stays 0.0
        # Transpose to (Channels, Height, Width) for PyTorch/CNN compatibility
        return {
            "observations": one_hot.transpose(2, 0, 1),
            "stats": stats,
            "action_mask": self.get_action_mask(),
        }

    def _get_deltas(self):
        deltas = deepcopy(self.map_obs_deque)
        for delta_idx in range(2, self.obs_stack_size + 1):
            deltas[-delta_idx] -= self.map_obs_deque[-1]
        return deltas

    def observe(self, _=None):
        return {
            "observations": np.concatenate(self._get_deltas()),
            "stats": self.saved_stats,
            "action_mask": self.get_action_mask(),
        }

    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ):
        self.agents = self.possible_agents[:]
        self._prepare()
        if self.rendering:
            self.renderer.reset()
        self._update_frames(None)
        return {0: self.observe()}, mock_info

    def get_action_mask(self, agent=None):
        target_mask = np.zeros(self.game.n_players + 1, dtype=np.float32)
        neighbors = findNeighbours(self.game, 0)
        for neigh in neighbors:
            target_mask[self.permute_id(neigh)] = 1.0
        commit_mask = np.ones(self.n_commit_bins, dtype=np.float32)
        return np.concatenate([target_mask, commit_mask])

    def _update_frames(self, _=None):
        obs = self._get_observation_frame(_)
        self.map_obs_deque.append(obs["observations"])
        self.saved_stats = obs["stats"]

    def step(self, action: dict[int, Any]):
        if not self.agents:
            return {}, {}, {}, {}, {}
        # 0 is neutral, others are agents
        target = action[0][0]
        commited_bin = action[0][1]
        commited = (
            self.game.id_to_country[self.agent_id].money * commited_bin / 10.0
        )  # Convert 0..10 to 0.0..1.0
        target = self.unpermute_id(target)

        old_player_size = self.game.id_to_country[self.agent_id].size
        self.game.id_to_country[self.agent_id].attackInit(self.game, target, commited)
        for _ in range(self.ticks_delta):
            self.game.tick()
        self._update_frames()
        obs = {0: self.observe(self.agents[0])}

        reward = {
            0: (self.game.id_to_country[self.agent_id].size - old_player_size)
            / (self.game.n_grid_rows * self.game.n_grid_columns)
        }
        self.terminations[0] = (
            self.game.id_to_country[self.agent_id].size == 0
            or len(self.game.id_to_country) == 1
        )
        if self.terminations[0]:
            if self.game.id_to_country[self.agent_id].size > 0:
                reward[0] += 10
            else:
                reward[0] -= 10

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
