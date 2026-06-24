import warnings
from collections import deque
from copy import deepcopy

from render import GameRenderer
from utility import permute_id
import numpy as np
from typing import Dict, Any, Optional
from pettingzoo import ParallelEnv
from game.game import Game
from gymnasium import spaces
from game.gameFuncs import findNeighbours
from strategy_config import (
    POLICY_COLORS,
    GAME_MAX_TURNS,
    GAMMA_DECAY,
    TERMINAL_REWARD_COEFF,
    N_COMMIT_BINS,
)


class CustomEnvironment(ParallelEnv):
    metadata = {
        "name": "custom_environment_v0",
    }

    def permute_id(self, original_id: int, agent: int) -> int:
        return permute_id(self.id_permutation[agent], original_id)

    def unpermute_id(self, permuted_id: int, agent: int) -> int:
        return int(self.reverse_id_permutation[agent][permuted_id] - 1)

    def __init__(
        self,
        rendering,
        n_players,
        grid_rows,
        grid_columns,
    ):
        super().__init__()
        self.n_players = n_players
        self.n_agents = n_players
        self.id_permutation: Dict[int, np.ndarray] = {}
        self.reverse_id_permutation: Dict[int, np.ndarray] = {}
        self.map_obs_deque: Dict[int, deque[np.ndarray]] = {}
        self.policy_mapping = {i: "p0" for i in range(self.n_players)}
        self.grid_columns = grid_columns
        self.grid_rows = grid_rows
        self.max_steps = GAME_MAX_TURNS
        self.reward_convexity = 1
        self.ticks_delta = 1
        self.render_mode = None
        self.rendering = rendering
        self.obs_stack_size = 4
        self.n_commit_bins = N_COMMIT_BINS
        self.agents = []
        self.next_country_colors = None
        self.country_colors = list(POLICY_COLORS.values())[: self.n_players]
        self._prepare()

        self.terminations = {agent: False for agent in self.possible_agents}
        self.truncations = {agent: False for agent in self.possible_agents}

        self.saved_stats: Dict[int, np.ndarray] = {}
        self.saved_obs_metadata: Dict[int, dict] = {}

        self.n_board_channels = self.game.n_players + 1
        self.n_stats_channels = self.game.n_players * 2
        obs_shape = (
            self.n_board_channels * self.obs_stack_size,
            self.game.n_grid_rows,
            self.game.n_grid_columns,
        )

        self.n_stats = (self.game.n_players * 2) + 1
        stats_shape = (self.n_stats,)

        self.action_spaces = {
            agent: spaces.MultiDiscrete([self.game.n_players + 1, self.n_commit_bins])
            for agent in self.possible_agents
        }

        self.observation_spaces = {
            agent: spaces.Dict(
                {
                    "observations": spaces.Box(
                        low=-1, high=1, shape=obs_shape, dtype=np.int8
                    ),
                    "stats": spaces.Box(
                        low=0, high=1, shape=stats_shape, dtype=np.float32
                    ),
                    "action_mask": spaces.Box(
                        low=0.0,
                        high=1.0,
                        shape=(self.game.n_players + 1 + self.n_commit_bins,),
                        dtype=np.float32,
                    ),
                    "board": spaces.Box(
                        low=-1,
                        high=self.game.n_players,
                        shape=(self.game.n_grid_rows, self.game.n_grid_columns),
                        dtype=np.int16,
                    ),
                    "id_to_money": spaces.Box(
                        low=0.0,
                        high=np.inf,
                        shape=(self.game.n_players,),
                        dtype=np.float32,
                    ),
                    "id_to_size": spaces.Box(
                        low=0, high=np.inf, shape=(self.game.n_players,), dtype=np.int32
                    ),
                    "agent_id": spaces.Box(
                        low=0, high=self.game.n_players - 1, dtype=np.int8
                    ),
                    "agent_aggro": spaces.Box(
                        low=0.0, high=np.inf, shape=(1,), dtype=np.float32
                    ),
                    "tooBig": spaces.Box(
                        low=0.0, high=np.inf, shape=(1,), dtype=np.float32
                    ),
                    "threshold": spaces.Box(
                        low=0.0, high=np.inf, shape=(1,), dtype=np.float32
                    ),
                    "id_permutation": spaces.Box(
                        low=0.0,
                        high=self.game.n_players,
                        shape=(self.game.n_players + 1,),
                        dtype=np.int8,
                    ),
                }
            )
            for agent in self.possible_agents
        }

    def _build_permutations(self, agent):
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
            country_colors=self.country_colors,
        )
        self.possible_agents = self.game.agents
        if self.rendering:
            self.renderer = GameRenderer(self.game.countryColors)

        unstacked_obs_shape = (
            (self.game.n_players + 1),
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

    def _get_observation_frame(self, agent: int):
        board = np.array(self.game.board)
        permuted_board = np.full(board.shape, -1)
        for original_id in range(-1, self.game.n_players):
            permuted_board[board == original_id] = self.permute_id(original_id, agent)

        num_channels = self.game.n_players + 1
        one_hot = np.eye(num_channels, dtype=np.int8)[permuted_board]
        stats = np.zeros(self.n_stats, dtype=np.float32)

        id_to_money = np.zeros(self.game.n_players, dtype=np.float32)
        id_to_size = np.zeros(self.game.n_players, dtype=np.int32)

        for perm_idx in range(1, self.game.n_players + 1):
            original_id = self.unpermute_id(perm_idx, agent)
            stat_idx = perm_idx - 1

            if original_id in self.game.id_to_country:
                c = self.game.id_to_country[original_id]
                id_to_money[original_id] = c.money
                id_to_size[original_id] = c.size
                stats[stat_idx] = (
                    c.money / 1500 / self.game.n_grid_rows / self.game.n_grid_columns
                )
                stats[self.game.n_players + stat_idx] = (
                    c.size / self.game.n_grid_rows / self.game.n_grid_columns
                )
        stats[-1] = float(self.current_step) / float(self.max_steps)
        # avoid roundoff errors that cause negatives
        stats = np.clip(stats, 0.0, 1.0)

        c_agent = self.game.id_to_country.get(agent)

        return {
            "observations": one_hot.transpose(2, 0, 1),
            "stats": stats,
            "action_mask": self.get_action_mask(agent),
            "board": board.astype(np.int8),
            "id_to_money": id_to_money,
            "id_to_size": id_to_size,
            "agent_aggro": np.array(
                [c_agent.aggro if c_agent else 0.0], dtype=np.float32
            ),
            "agent_id": agent,
            "tooBig": np.array([c_agent.tooBig if c_agent else 0.0], dtype=np.float32),
            "threshold": np.array(
                [c_agent.threshold if c_agent else 0.0], dtype=np.float32
            ),
            "id_permutation": self.id_permutation[agent],
        }

    def _get_deltas(self, agent=None):
        deltas = deepcopy(self.map_obs_deque[agent])
        for delta_idx in range(2, self.obs_stack_size + 1):
            deltas[-delta_idx] -= self.map_obs_deque[agent][-1]
        return deltas

    def observe(self, agent=None):
        return {
            "observations": np.concatenate(self._get_deltas(agent)),
            **self.saved_obs_metadata[agent],
        }

    def set_next_game_colors(self, colors):
        assert len(colors) == self.game.n_players
        self.next_country_colors = colors

    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ):
        if self.next_country_colors is not None:
            self.country_colors = self.next_country_colors
            self.next_country_colors = None
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

    def _update_frames(self, agent: int):
        obs = self._get_observation_frame(agent)
        self.map_obs_deque[agent].append(obs.pop("observations"))
        self.saved_obs_metadata[agent] = obs

    def step(self, action: Dict[int, Any]):
        attacks = []
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

            raw_action = action[agent]
            target = raw_action[0]
            committed_bin = raw_action[1]
            committed = (
                self.game.id_to_country[agent].money
                * committed_bin
                / (N_COMMIT_BINS - 1)
            )
            target = self.unpermute_id(target, agent)

            if target == agent:
                warnings.warn(f"Suicide is a sin, player {agent}")
            # warnings.warn(f"{agent} attacks {target} with {committed}")
            attacks.append(
                {"attacker": agent, "target": target, "commit": float(committed)}
            )
            self.game.id_to_country[agent].attackInit(self.game, target, committed)

        for _ in range(self.ticks_delta):
            self.game.tick()

        self.current_step += 1

        for agent in self.agents:
            self._update_frames(agent)

        obs, rewards, terminations, truncations, infos = {}, {}, {}, {}, {}
        is_timeout = self.current_step >= self.max_steps

        for agent in self.agents:
            is_alive = (
                agent in self.game.id_to_country
                and self.game.id_to_country[agent].size > 0
            )
            won = is_alive and len(self.game.id_to_country) == 1

            new_size = self.game.id_to_country[agent].size if is_alive else 0
            rewards[agent] = (GAMMA_DECAY * new_size - old_player_size[agent]) / (
                self.game.n_grid_rows * self.game.n_grid_columns
            )
            terminations[agent] = not is_alive or won
            truncations[agent] = is_timeout
            infos[agent] = {}
            obs[agent] = self.observe(agent)
            if terminations[agent] or truncations[agent]:
                if truncations[agent]:
                    # The game timed out. Rank everyone currently alive by size
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
                ) * TERMINAL_REWARD_COEFF
                infos[agent]["place"] = place

        for agent in self.agents:
            infos[agent]["attacks"] = attacks.copy()

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
