from collections import deque
from copy import deepcopy

from render import GameRenderer
import numpy as np
from typing import Dict, Any, Optional
from pettingzoo import ParallelEnv
from game.game import Game
from gymnasium import spaces
from game.gameFuncs import findNeighbours
from strategy_config import POLICY_COLORS
# mock_info = {0: {}}


# TODO multiple agents: create a pool of agents and bootstrap
# them each time for more diversity!


class CustomEnvironment(ParallelEnv):
    metadata = {
        "name": "custom_environment_v0",
    }

    def permute_id(self, original_id: int, agent: int) -> int:
        return int(self.id_permutation[agent][original_id + 1])

    def unpermute_id(self, permuted_id: int, agent: int) -> int:
        return int(self.reverse_id_permutation[agent][permuted_id] - 1)

    def __init__(self, rendering, n_players, grid_rows, grid_columns):
        """
        ticks_delta: int (default = 1) how many game ticks to run between agent's decisions
        """
        super().__init__()
        self.n_players = n_players
        # TODO remove this "legacy"
        self.n_agents = n_players
        self.id_permutation: Dict[int, np.ndarray] = {}
        self.reverse_id_permutation: Dict[int, np.ndarray] = {}
        self.map_obs_deque: Dict[int, deque[np.ndarray]] = {}

        self.grid_columns = grid_columns
        self.grid_rows = grid_rows
        self.max_steps = 3000
        self.reward_convexity = 1.5
        self.ticks_delta = 1
        self.render_mode = None
        self.rendering = rendering
        self.obs_stack_size = 1
        self.n_commit_bins = 11
        self.agents = []
        self._prepare()

        self.terminations = {agent: False for agent in self.possible_agents}
        self.truncations = {agent: False for agent in self.possible_agents}

        self.saved_stats: Dict[int, np.ndarray] = {}

        # map of one-hot vectors (each player + neutral)
        self.n_board_channels = self.game.n_players + 1
        self.n_stats_channels = self.game.n_players * 2
        obs_shape = (
            self.n_board_channels * self.obs_stack_size,
            self.game.n_grid_rows,
            self.game.n_grid_columns,
        )
        self.n_stats = self.game.n_players * 2
        stats_shape = (self.n_stats,)

        self.action_spaces = {
            agent: spaces.MultiDiscrete([self.game.n_players + 1, self.n_commit_bins])
            for agent in self.possible_agents
        }  # who to attack (or stall) + amount of troops (0%, 10%, 20%, ...)
        self.observation_spaces = {
            agent: spaces.Dict(
                {
                    "observations": spaces.Box(
                        low=-1,
                        high=1,
                        shape=obs_shape,
                        dtype=np.int8,
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
            for agent in self.possible_agents
        }

    def update_game_colors(self, new_colors: list[str]):
        assert len(new_colors) == self.n_players
        self.game.countryColors = new_colors

    def _build_permutations(self, agent):
        # Maps original ids in [-1, n_players - 1] to [0, n_players].
        # Array index for original_id is (original_id + 1).
        id_perm = np.zeros(self.game.n_players + 1, dtype=int)
        id_perm[agent + 1] = 1
        others = [
            other_agent + 1
            for other_agent in range(self.game.n_players)
            if other_agent != agent
        ]
        np.random.shuffle(others)

        for perm_id, id in enumerate(others, start=2):
            id_perm[id] = perm_id

        reverse_perm = np.zeros(self.game.n_players + 1, dtype=int)
        for id, perm_id in enumerate(id_perm):
            reverse_perm[perm_id] = id

        return id_perm, reverse_perm

    def _prepare(self):
        self.game = Game(
            n_players=self.n_players,
            n_agents=self.n_agents,
            grid_rows=self.grid_rows,
            grid_columns=self.grid_columns,
            country_colors=list(POLICY_COLORS.values())[: self.n_players],
        )
        self.possible_agents = self.game.agents
        if self.rendering:
            self.renderer = GameRenderer(self.game.countryColors)

        unstacked_obs_shape = (
            (self.game.n_players + 1),  # no self.obs_stack_size, we need raw one here
            self.game.n_grid_rows,
            self.game.n_grid_columns,
        )

        for agent in self.possible_agents:
            perm, reverse_perm = self._build_permutations(agent)
            self.id_permutation[agent] = perm
            self.reverse_id_permutation[agent] = reverse_perm

            self.map_obs_deque[agent] = deque(maxlen=self.obs_stack_size)

            for _ in range(self.obs_stack_size):
                self.map_obs_deque[agent].append(
                    np.zeros(unstacked_obs_shape, dtype=np.int8)
                )

    def _get_observation_frame(self, agent):
        board = np.array(self.game.board)
        permuted_board = np.full(board.shape, -1)
        for original_id in range(-1, self.game.n_players):
            permuted_board[board == original_id] = self.permute_id(original_id, agent)

        num_channels = self.game.n_players + 1
        one_hot = np.eye(num_channels, dtype=np.int8)[permuted_board]
        stats = np.zeros(self.n_stats, dtype=np.float32)
        for perm_idx in range(1, self.game.n_players + 1):
            original_id = self.unpermute_id(perm_idx, agent)
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
        # avoid roundoff errors that cause negatives
        stats = np.clip(stats, 0.0, 1.0)
        return {
            "observations": one_hot.transpose(2, 0, 1),
            "stats": stats,
            "action_mask": self.get_action_mask(agent),
        }

    def _get_deltas(self, agent=None):
        deltas = deepcopy(self.map_obs_deque[agent])
        for delta_idx in range(2, self.obs_stack_size + 1):
            deltas[-delta_idx] -= self.map_obs_deque[agent][-1]
        return deltas

    def observe(self, agent=None):
        return {
            "observations": np.concatenate(self._get_deltas(agent)),
            "stats": self.saved_stats[agent],
            "action_mask": self.get_action_mask(agent),
        }

    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ):
        self.current_step = 0
        self._prepare()
        self.agents = self.possible_agents[:]
        self.terminations = {agent: False for agent in self.possible_agents}
        self.truncations = {agent: False for agent in self.possible_agents}

        if self.rendering:
            self.renderer.reset()
        for agent in self.possible_agents:
            self._update_frames(agent)
        return {agent: self.observe(agent) for agent in self.possible_agents}, {
            agent: {} for agent in self.possible_agents
        }

    def get_action_mask(self, agent):
        target_mask = np.zeros(self.game.n_players + 1, dtype=np.float32)

        if agent in self.game.id_to_country:
            neighbors = findNeighbours(self.game, agent)
            for neigh in neighbors:
                target_mask[self.permute_id(neigh, agent)] = 1.0
        commit_mask = np.ones(self.n_commit_bins, dtype=np.float32)
        return np.concatenate([target_mask, commit_mask])

    def _update_frames(self, agent):
        obs = self._get_observation_frame(agent)
        self.map_obs_deque[agent].append(obs["observations"])
        self.saved_stats[agent] = obs["stats"]

    def step(self, action: Dict[int, Any]):
        old_player_size = {
            agent: (
                self.game.id_to_country[agent].size
                if agent in self.game.id_to_country
                else 0
            )
            for agent in self.possible_agents
        }

        for agent in self.agents:
            if agent not in self.game.id_to_country:
                continue
            target = action[agent][0]
            commited_bin = action[agent][1]
            commited = (
                self.game.id_to_country[agent].money * commited_bin / 10.0
            )  # Convert 0..10 to 0.0..1.0
            target = self.unpermute_id(target, agent)
            self.game.id_to_country[agent].attackInit(self.game, target, commited)

        for _ in range(self.ticks_delta):
            self.game.tick()

        for agent in self.agents:
            self._update_frames(agent)

        obs, rewards, terminations, truncations, infos = {}, {}, {}, {}, {}
        self.current_step += 1
        is_timeout = self.current_step >= self.max_steps
        for agent in list(self.agents):
            is_alive = (
                agent in self.game.id_to_country
                and self.game.id_to_country[agent].size > 0
            )
            won = is_alive and len(self.game.id_to_country) == 1

            new_size = self.game.id_to_country[agent].size if is_alive else 0
            rewards[agent] = (new_size - old_player_size[agent]) / (
                self.game.n_grid_rows * self.game.n_grid_columns
            )
            terminations[agent] = not is_alive or won
            truncations[agent] = is_timeout
            infos[agent] = {"agent_id": agent}
            obs[agent] = self.observe(agent)
            if terminations[agent] or truncations[agent]:
                if truncations[agent]:
                    # The game timed out. Rank everyone currently alive by size.
                    alive_agents = [
                        a for a in self.agents if a in self.game.id_to_country
                    ]
                    alive_agents.sort(
                        key=lambda a: self.game.id_to_country[a].size, reverse=True
                    )
                    place = alive_agents.index(agent) + 1
                else:
                    place = len(self.game.id_to_country)
                # 1 for first, -1 for last
                rewards[agent] += (
                    2
                    * (
                        ((self.game.n_players - place) / (self.game.n_players - 1))
                        ** self.reward_convexity
                    )
                    - 1
                )
                infos[agent]["place"] = place

        self.truncations = truncations
        self.terminations = terminations

        self.agents = [
            agent
            for agent in self.agents
            if not terminations[agent] and not truncations[agent]
        ]
        return obs, rewards, terminations, truncations, infos

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
