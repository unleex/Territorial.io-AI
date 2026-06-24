import warnings
from ray.rllib.policy.policy import Policy
from players.base_player import BasePlayer
import random
from game.gameAI import find_neighbours
import numpy as np
from strategy_config import BOT_EXPANSION_BOOST, N_COMMIT_BINS
from utility import permute_id


class BotPolicy(Policy, BasePlayer):
    TARGET_EMA_ALPHA = 0.35
    TARGET_RETARGET_MARGIN = 0.15

    def compute_actions(
        self,
        obs_batch,
        state_batches=None,
        prev_action_batch=None,
        prev_reward_batch=None,
        info_batch=None,
        episodes=None,
        **kwargs,
    ):
        actions = []

        for i in range(len(obs_batch["agent_id"])):
            episode = episodes[i]
            env_agent_id = obs_batch["agent_id"][i]

            if "bot_state" not in episode.user_data:
                episode.user_data["bot_state"] = {}
            bot_state = episode.user_data["bot_state"].setdefault(
                env_agent_id,
                {"target": None, "ema_money": {}},
            )

            target, commit = self.runai(
                agent_id=env_agent_id,
                bot_state=bot_state,
                board=obs_batch["board"][i],
                id_to_money=obs_batch["id_to_money"][i],
                id_to_size=obs_batch["id_to_size"][i],
                agent_aggro=obs_batch["agent_aggro"][i][0],
                tooBig=obs_batch["tooBig"][i][0],
                threshold=obs_batch["threshold"][i][0],
            )

            if target is None:
                actions.append(np.array([0, 0], dtype=np.int64))
                continue

            permuted_target = permute_id(
                obs_batch["id_permutation"][i],
                target,
            )

            money = max(1.0, float(obs_batch["id_to_money"][i][env_agent_id]))
            commit_frac = np.clip(float(commit) / money, 0.0, 1.0)
            commit_bin = np.ceil(commit_frac * (N_COMMIT_BINS - 1))

            actions.append(np.array([int(permuted_target), commit_bin], dtype=np.int64))

        return np.asarray(actions, dtype=np.int64), [], {}

    def runai(
        self,
        *,
        agent_id: int,
        bot_state,
        board,
        id_to_money,
        id_to_size,
        agent_aggro,
        tooBig,
        threshold,
    ):
        d = find_neighbours(board, agent_id)

        # Forget dead / invalid current target immediately
        current_target = bot_state.get("target", None)
        if current_target not in d or current_target not in id_to_money:
            current_target = None

        # If neutral territory is nearby, capture immediately
        if -1 in d:
            commit = int(5.0 * d[-1]) + 1
            if (
                id_to_money[agent_id]
                > random.randint(id_to_size[agent_id] * 10, id_to_size[agent_id] * 30)
                / BOT_EXPANSION_BOOST
                and id_to_money[agent_id] > commit
            ):
                bot_state["target"] = None
                return -1, commit

        ema_money: dict = bot_state["ema_money"]
        for i in d:
            money = id_to_money[i]
            prev = ema_money.get(i, money)
            ema_money[i] = (
                self.TARGET_EMA_ALPHA * money + (1.0 - self.TARGET_EMA_ALPHA) * prev
            )

        if not d:
            bot_state["target"] = None
            return None, None

        # Find smallest neighbour by smoothed money
        smallest = None
        smallest_score = None
        for i in d:
            score = ema_money.get(i, id_to_money[i])
            if smallest is None or score < smallest_score:
                smallest = i
                smallest_score = score

        if smallest_score is None:
            bot_state["target"] = None
            return None, None

        # Keep current target unless another neighbour is clearly better
        if current_target is not None:
            current_score = ema_money.get(current_target, id_to_money[current_target])
            if current_score <= smallest_score * (1.0 + self.TARGET_RETARGET_MARGIN):
                smallest = current_target
                smallest_score = current_score

        bot_state["target"] = smallest

        density = id_to_money[smallest] / max(1, id_to_size[smallest])
        if (
            id_to_money[smallest] < id_to_money[agent_id] * agent_aggro
            and int(d[smallest] * density) + 1 <= id_to_money[agent_id]
        ):
            return smallest, int(d[smallest] * density) + 1

        if id_to_money[agent_id] / max(1, id_to_size[agent_id]) / 1000 < threshold:
            return None, None

        # If the difference between the to countries isn't too big, or growth has stopped
        if (
            id_to_money[smallest] / (id_to_money[agent_id] * agent_aggro) <= tooBig
            or id_to_money[agent_id] >= id_to_size[agent_id] * 1000
        ):
            return smallest, int(id_to_money[agent_id] * agent_aggro)

        return None, None

    def learn_on_batch(self, samples):
        return {}

    def get_weights(self):
        return {}

    def set_weights(self, weights):
        pass
