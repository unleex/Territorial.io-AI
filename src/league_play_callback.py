import random
import numpy as np
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.algorithms import Algorithm
from ray.rllib.evaluation.episode_v2 import EpisodeV2
from ray.rllib.utils.metrics.metrics_logger import MetricsLogger
from players.bot import BotPolicy
from environment import CustomEnvironment
from players.model import MODEL_NAME

# XXX what the
TOTAL_PLAYERS = 8
NUM_FROZEN_POLICIES = 10


def get_action_space():
    # XXX now we have the same action for any agent.
    return CustomEnvironment().action_spaces.items()[0][1]


def get_observation_space():
    # XXX now we have the same obs for any agent.
    return CustomEnvironment().observation_spaces.items()[0][1]


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


class LeaguePlayCallback(DefaultCallbacks):
    def __init__(self, avg_place_threshold=3, n_trainable_players=2):
        super().__init__()
        self.current_opponent = 0
        self.avg_place_threshold = avg_place_threshold
        self.n_trainable_players = n_trainable_players
        self.num_policies = 10
        self.available_snapshots = [f"p0_v{i}" for i in range(self.num_policies)]

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
            if episode.policy_for(agent_id) == "p0":
                info = episode.last_info_for(agent_id=agent_id)
                if "place" in info:
                    episode.user_data["places"][agent_id] = info["place"]

    def on_episode_end(
        self, *, worker, base_env, policies, episode: EpisodeV2, env_index, **kwargs
    ):
        places = episode.user_data["places"]
        episode.custom_metrics["avg_place"] = float(np.mean(list(places.values())))

    def update_league(
        self, algorithm: Algorithm, step, metrics_logger: MetricsLogger, result=None
    ):
        metrics_logger.log_value(
            key="league_updates",
            value=self.current_opponent,
        )
        slot = self.current_opponent % self.num_policies
        snapshot_id = f"p0_v{slot}"

        print(f"Snapshotting into {snapshot_id}")

        main_policy = algorithm.get_policy("p0")
        snapshot_policy = algorithm.get_policy(snapshot_id)

        snapshot_policy.set_state(main_policy.get_state())
        self.current_opponent += 1
        chosen = random.sample(policies.keys(), 8)  # XXX what the

        def mapping_fn(agent_id):
            if agent_id in range(self.n_trainable_players):
                return "p0"
            return chosen[agent_id]

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
        # algorithm.env_runner_group.foreach_policy(collect_policy_id)
        # self.current_opponent = max(policy_ids)
        main_policy = algorithm.get_policy("p0")
        for new_policy_id in self.available_snapshots:
            algorithm.add_policy(
                policy_id=new_policy_id,
                policy_cls=type(main_policy),
                observation_space=main_policy.observation_space,
                action_space=main_policy.action_space,
                config=main_policy.config,
            )
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
        avg_best_place = custom_metrics.get("avg_place_mean", float("inf"))
        if avg_best_place <= self.avg_place_threshold:
            self.update_league(algorithm, self.current_opponent + 1, metrics_logger)
