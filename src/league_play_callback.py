import random
import numpy as np
from itertools import combinations
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.algorithms import Algorithm
from ray.rllib.evaluation.episode_v2 import EpisodeV2
from ray.rllib.utils.metrics.metrics_logger import MetricsLogger
from players.bot import BotPolicy
from arena import EloRating
from environment import CustomEnvironment
from players.model import MODEL_NAME

# XXX what the
TOTAL_PLAYERS = 8
NUM_FROZEN_POLICIES = 3


def get_action_space():
    # XXX now we have the same action for any agent.
    return CustomEnvironment().action_spaces[
        next(iter(CustomEnvironment().action_spaces))
    ]


def get_observation_space():
    # XXX now we have the same obs for any agent.
    return CustomEnvironment().observation_spaces[
        next(iter(CustomEnvironment().observation_spaces))
    ]


policies = {
    **{
        f"bot{i}": (
            BotPolicy,
            get_observation_space(),
            get_action_space(),
            {"model": {}},
        )
        for i in range(TOTAL_PLAYERS)
    },
    **{
        "p0": (
            None,
            get_observation_space(),
            get_action_space(),
            {"model": {"custom_model": MODEL_NAME}},
        )
    },
    **{
        f"p0_v{i}": (
            None,
            get_observation_space(),
            get_action_space(),
            {"model": {"custom_model": MODEL_NAME}},
        )
        for i in range(NUM_FROZEN_POLICIES)
    },
}
policy_pool = policies.copy()
policy_pool.pop("p0")


class LeaguePlayCallback(DefaultCallbacks):
    def __init__(self, avg_place_threshold=3, n_trainable_players=2):
        super().__init__()
        self.current_opponent = 0
        self.avg_place_threshold = avg_place_threshold
        self.n_trainable_players = n_trainable_players
        self.num_policies = NUM_FROZEN_POLICIES
        self.available_snapshots = [f"p0_v{i}" for i in range(self.num_policies)]
        self.elo = EloRating()

    def on_episode_start(
        self,
        *,
        episode,
        env_runner=None,
        metrics_logger: MetricsLogger = None,
        env=None,
        env_index,
        rl_module=None,
        worker=None,
        base_env=None,
        policies=None,
        **kwargs,
    ):
        episode.user_data["places"] = {}

    def on_episode_step(self, *, episode: EpisodeV2, **kwargs):
        for agent_id in episode.get_agents():
            info = episode.last_info_for(agent_id=agent_id)
            if "place" in info:
                episode.user_data["places"][agent_id] = info["place"]

    def on_episode_end(
        self, *, worker, base_env, policies, episode: EpisodeV2, env_index, **kwargs
    ):
        places = episode.user_data["places"]

        policy_places = {}
        for agent_id, place in places.items():
            policy_id = episode.policy_for(agent_id)
            if policy_id not in policy_places:
                policy_places[policy_id] = []
            policy_places[policy_id].append(place)

        for policy_id, policy_episode_places in policy_places.items():
            episode.custom_metrics[f"{policy_id}/avg_place"] = float(
                np.mean(policy_episode_places)
            )

        active_policies = list(policy_places.keys())
        if len(active_policies) >= 2:
            # Sort the pair alphabetically so 'p0 vs bot0' and 'bot0 vs p0'
            # always map to the exact same metric key across all workers
            for p_a, p_b in combinations(sorted(active_policies), 2):
                mean_a = np.mean(policy_places[p_a])
                mean_b = np.mean(policy_places[p_b])

                if mean_a < mean_b:
                    score_a = 1.0
                elif mean_a > mean_b:
                    score_a = 0.0
                else:
                    score_a = 0.5

                episode.custom_metrics[f"matchup_score/{p_a}_vs_{p_b}"] = score_a

    def update_league(
        self, algorithm: Algorithm, metrics_logger: MetricsLogger, result=None
    ):
        metrics_logger.log_value(
            key="league_updates",
            value=self.current_opponent,
        )
        slot = self.current_opponent % self.num_policies
        snapshot_id = f"p0_v{slot}"

        print(f"Snapshotting into {snapshot_id}")
        self.current_opponent += 1

        main_policy = algorithm.get_policy("p0")
        snapshot_policy = algorithm.get_policy(snapshot_id)

        snapshot_policy.set_state(main_policy.get_state())
        algorithm.set_weights({snapshot_id: main_policy.get_weights()})
        chosen = random.sample(
            sorted(policy_pool.keys()), 8 - self.n_trainable_players
        )  # XXX what the 8
        # force at least one bot for stability
        if not any([p.startswith("bot") for p in chosen]):
            chosen[0] = "bot0"
        print("Current policy setup:", *chosen)

        def mapping_fn(agent_id, episode, **kwargs):
            if agent_id in range(self.n_trainable_players):
                return "p0"
            return chosen[agent_id - self.n_trainable_players]

        algorithm.env_runner_group.foreach_env_runner(
            lambda w: w.set_policy_mapping_fn(mapping_fn)
        )

    def on_algorithm_init(
        self,
        *,
        algorithm: Algorithm,
        metrics_logger: MetricsLogger | None = None,
        **kwargs,
    ) -> None:
        main_policy = algorithm.get_policy("p0")
        self.update_league(algorithm, metrics_logger)
        for new_policy_id in self.available_snapshots:
            algorithm.set_weights({new_policy_id: main_policy.get_weights()})

    def on_train_result(
        self, *, algorithm: Algorithm, result, metrics_logger, **kwargs
    ):
        metrics_logger.log_value(
            key="league_updates",
            value=self.current_opponent,
        )
        env_runners_dict = result.get("env_runners", {})
        custom_metrics = env_runners_dict.get("custom_metrics", {})
        for key, empirical_win_rate in custom_metrics.items():
            if key.startswith("matchup_score/") and key.endswith("_mean"):
                # Clean the key to extract policy names
                # e.g., "matchup_score/p0_vs_bot0_mean" -> ["p0", "bot0"]
                pair_str = key.replace("matchup_score/", "").replace("_mean", "")
                p_a, p_b = pair_str.split("_vs_")

                self.elo.register(p_a)
                self.elo.register(p_b)

                self.elo.update_from_batch(p_a, p_b, empirical_win_rate)

        for policy_id, rating in self.elo.ratings.items():
            custom_metrics[f"{policy_id}/elo"] = rating

        print(self.elo.summary())

        avg_best_place = custom_metrics.get("p0/avg_place_mean", float("inf"))
        if avg_best_place <= self.avg_place_threshold:
            self.update_league(algorithm, metrics_logger)
