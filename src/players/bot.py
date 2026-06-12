from ray.rllib.policy.policy import Policy
from players.base_player import BasePlayer
import warnings
import random
from game.gameAI import findNeighbours
import numpy as np
import typing

if typing.TYPE_CHECKING:
    from environment import CustomEnvironment


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
            warnings.warn(
                "Bot policy compute_actions didn't receive the worker, returning."
            )

            if isinstance(obs_batch, dict):
                batch_size = len(next(iter(obs_batch.values())))
            else:
                batch_size = len(obs_batch)

            return [self.action_space.sample() for _ in range(batch_size)], [], {}

        sub_envs = worker.env.get_sub_environments()
        actions = []
        for i in range(batch_size):
            episode = episodes[i]

            env_id = episode.env_id

            # Grab game instance for this specific observation
            env_instance: "CustomEnvironment" = sub_envs[env_id]
            game = env_instance.game
            env_agent_id = info_batch[i]["agent_id"]
            agent_country = game.id_to_country[env_agent_id]
            target, commit = self.runai(
                game, agent_country
            )  # FIXME bot algo doesn't account for env.tick() in between
            if target is None:
                actions.append(np.array([0, 0], dtype=np.int64))
                continue
            permuted_target = env_instance.permute_id(target, env_agent_id)

            ratio = commit / agent_country.money if agent_country.money > 0 else 0
            commit_bin = int(np.round(np.clip(ratio * 10, 0, 10)))
            if commit > 0 and commit_bin == 0:
                commit_bin = 1

            actions.append(np.array([permuted_target, commit_bin], dtype=np.int64))

        return np.array(actions, dtype=np.int64), [], {}

    def runai(self, game, agent):
        d = findNeighbours(game, agent.id)
        if -1 in d:
            commit = int(5.0 * d[-1]) + 1
            if (
                agent.money > random.randint(agent.size * 10, agent.size * 30)
                and agent.money > commit
            ):
                return -1, commit  # Exact amount for one layer
            return None, None

        # Find smallest neighbour
        smallest = None
        for i in d:
            if (
                smallest is None
                or game.id_to_country[i].money < game.id_to_country[smallest].money
            ):
                smallest = i
        # If the smallest neighbour is signifcantly smaller
        density = game.id_to_country[smallest].money / game.id_to_country[smallest].size
        if (
            game.id_to_country[smallest].money < agent.money * agent.aggro
            and int(d[smallest] * density) + 1 <= agent.money
        ):
            return smallest, int(d[smallest] * density) + 1
        # if attack threshold not met
        if agent.money / agent.size / 1000 < agent.threshold:
            return None, None
        # If the difference between the to countries isn't too big, or growth has stopped
        if (
            game.id_to_country[smallest].money / (agent.money * agent.aggro)
            <= agent.tooBig
            or agent.money >= agent.size * 1000
        ):
            return smallest, int(agent.money * agent.aggro)
        return None, None

    def learn_on_batch(self, samples):
        return {}

    def get_weights(self):
        return {}

    def set_weights(self, weights):
        pass
