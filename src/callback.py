from ray.rllib.callbacks.callbacks import RLlibCallback
from log import VideoCallback
from league_play_callback import LeaguePlayCallback


class MergedCallback(RLlibCallback):
    def __init__(self):
        self.callbacks = [
            VideoCallback(),
            LeaguePlayCallback(avg_place_threshold=3, n_trainable_players=2),
        ]

    def on_episode_start(self, **kwargs):
        for cb in self.callbacks:
            cb.on_episode_start(**kwargs)

    def on_episode_end(self, **kwargs):
        for cb in self.callbacks:
            cb.on_episode_end(**kwargs)

    def on_episode_step(self, **kwargs):
        for cb in self.callbacks:
            cb.on_episode_step(**kwargs)

    def on_train_result(self, **kwargs):
        for cb in self.callbacks:
            cb.on_train_result(**kwargs)

    def on_algorithm_init(self, **kwargs):
        for cb in self.callbacks:
            cb.on_algorithm_init(**kwargs)
