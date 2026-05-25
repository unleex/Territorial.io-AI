import numpy as np
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.algorithms import Algorithm
from ray.rllib.evaluation.episode_v2 import EpisodeV2
from ray.rllib.utils.metrics.metrics_logger import MetricsLogger


class LeaguePlayCallback(DefaultCallbacks):
    def __init__(self, avg_place_threshold=3, n_trainable_players=2):
        super().__init__()
        self.current_opponent = 0
        self.avg_place_threshold = avg_place_threshold
        self.n_trainable_players = n_trainable_players

    def on_episode_start(
        self,
        *,
        episode,
        env_runner=None,
        metrics_logger=None,
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

    def update_league(self, algorithm: Algorithm, step, metrics_logger: MetricsLogger):
        self.current_opponent += 1
        new_policy_id = f"p0_v{self.current_opponent}"
        print(f"Snapshotting {new_policy_id} to league...")
        metrics_logger.log_value("league_updates", 1, reduce="sum")
        main_policy = algorithm.get_policy("p0")

        algorithm.add_policy(
            policy_id=new_policy_id,
            policy_cls=type(main_policy),
            observation_space=main_policy.observation_space,
            action_space=main_policy.action_space,
            config=main_policy.config,
        )

        algorithm.set_weights({new_policy_id: main_policy.get_weights()})

        def mapping_fn(agent_id, episode: EpisodeV2, **kwargs):
            if agent_id in range(self.n_trainable_players):
                return "p0"
            rng = np.random.default_rng(hash(episode.episode_id) + agent_id)
            pool = [f"p0_v{i}" for i in range(1, step + 1)]
            return rng.choice(pool)

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
        self.update_league(algorithm, self.current_opponent + 1, metrics_logger)

    def on_train_result(
        self, *, algorithm: Algorithm, result, metrics_logger, **kwargs
    ):
        env_runners_dict = result.get("env_runners", {})
        custom_metrics = env_runners_dict.get("custom_metrics", {})
        avg_best_place = custom_metrics.get("avg_place_mean", float("inf"))
        if avg_best_place <= self.avg_place_threshold:
            self.update_league(algorithm, self.current_opponent + 1, metrics_logger)
