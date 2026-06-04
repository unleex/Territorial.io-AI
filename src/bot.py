import random
from ray.rllib.policy.policy import Policy
from game.gameAI import findNeighbours


class BotPolicy(Policy):
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
        **kwargs,
    ):
        actions = []
        for obs in obs_batch:
            game = obs["game_instance"]
            agent = obs["agent"]
            actions.append(self.run_ai(game, agent))
        return actions, [], {}

    def run_ai(self, game, agent):
        d = findNeighbours(game, agent.id)

        if -1 in d:
            commit = int(5.0 * d[-1]) + 1
            if (
                agent.money > random.randint(agent.size * 10, agent.size * 30)
                and agent.money > commit
            ):
                return ("attack", -1, commit)
            return ("noop",)

        smallest = None
        for i in d:
            if (
                smallest is None
                or game.id_to_country[i].money < game.id_to_country[smallest].money
            ):
                smallest = i

        density = game.id_to_country[smallest].money / game.id_to_country[smallest].size

        if (
            game.id_to_country[smallest].money < agent.money * agent.aggro
            and int(d[smallest] * density) + 1 <= agent.money
        ):
            return ("attack", smallest, int(d[smallest] * density) + 1)

        if agent.money / agent.size / 1000 < agent.threshold:
            return ("noop",)

        if (
            game.id_to_country[smallest].money / (agent.money * agent.aggro)
            <= agent.tooBig
            or agent.money >= agent.size * 1000
        ):
            return ("attack", smallest, int(agent.money * agent.aggro))

        return ("noop",)

    def learn_on_batch(self, samples):
        return {}

    def get_weights(self):
        return {}

    def set_weights(self, weights):
        pass
