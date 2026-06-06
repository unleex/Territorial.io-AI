import random
import numpy as np
from game.gameAI import runAi
from ray.rllib.policy.policy import Policy
from players.base_player import BasePlayer
import warnings


class BotPolicy(Policy, BasePlayer):
    def compute_actions(
        self,
        obs_batch,
        state_batches=None,
        prev_action_batch=None,
        prev_reward_batch=None,
        info_batch=None,
        episodes=None,
        explore=None,
        timestep=None,
        agent_id=None,
        worker=None,
        **kwargs,
    ):
        batch_size = len(obs_batch)
        if worker is None:
            warnings.warn("Bot policy didn't receive the worker, returning.")
            return [self.action_space.sample() for _ in range(len(obs_batch))], [], {}
        for _ in range(batch_size):
            game = worker.env.game
            env_agent_id = agent_id
            actual_id = worker.env.unpermute_id(env_agent_id, env_agent_id)
            agent_country = game.id_to_country.get(actual_id)
            runAi(game, agent_country)

        return [self.action_space.sample() for _ in range(len(obs_batch))], [], {}

    def learn_on_batch(self, samples):
        return {}

    def get_weights(self):
        return {}

    def set_weights(self, weights):
        pass
