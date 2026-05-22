from collections import defaultdict
import numpy as np

from ray.rllib.callbacks.callbacks import RLlibCallback
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.rllib.utils.metrics import ENV_RUNNER_RESULTS


class LeaguePlayCallback(RLlibCallback):
    def __init__(self, win_rate_threshold=0.6):
        super().__init__()
        self.current_opponent = 0
        self.win_rate_threshold = win_rate_threshold
        self._matching_stats = defaultdict(int)

    def on_episode_end(
        self,
        *,
        episode,
        metrics_logger,
        **kwargs,
    ) -> None:
        # 1. Identify which of the 8 agents in this episode were the active learning "main" policy
        main_agents = [
            agent_id
            for agent_id in episode.get_agents()
            if episode.module_for(agent_id) == "main"
        ]

        if not main_agents:
            return

        rewards = episode.get_rewards()
        main_won = False

        # 2. Check if ANY of the "main" agents won this FFA match (Reward == 1.0)
        for agent in main_agents:
            if agent in rewards and rewards[agent][-1] == 1.0:
                main_won = True
                break

        # Log the win rate (1.0 if a main agent won, 0.0 if a bot or past snapshot won)
        metrics_logger.log_value(
            "win_rate",
            float(main_won),
            reduce="mean",
            window=100,
        )

    def update_policies(self, *, algorithm, result, **kwargs):
        win_rate = result[ENV_RUNNER_RESULTS].get("win_rate", 0.0)
        print(f"Iter={algorithm.iteration} win-rate={win_rate:.2f} -> ", end="")

        if win_rate > self.win_rate_threshold:
            self.current_opponent += 1
            new_module_id = f"main_v{self.current_opponent}"
            print(f"Snapshotting policy! Adding {new_module_id} to the league.")

            # Define the 8-player FFA mapping function
            def agent_to_module_mapping_fn(agent_id, episode, **kwargs):
                # Slots 0 and 1: Always Hardcoded Bots (The Anchor)
                if agent_id in [0, 1]:
                    return "bot_policy"  # Must match your bot policy ID in config

                # Slots 2 and 3: Always Active Learning Policy
                if agent_id in [2, 3]:
                    self._matching_stats["main"] += 1
                    return "main"

                # Slots 4, 5, 6, 7: Random mix of active policy and historical snapshots
                pool = ["main"] + [
                    f"main_v{i}" for i in range(1, self.current_opponent + 1)
                ]
                selected_opponent = np.random.choice(pool)
                self._matching_stats[selected_opponent] += 1
                return selected_opponent

            # Duplicate the current weights into the new historical module
            main_module = algorithm.get_module("main")
            algorithm.add_module(
                module_id=new_module_id,
                module_spec=RLModuleSpec.from_module(main_module),
                new_agent_to_module_mapping_fn=agent_to_module_mapping_fn,
            )

            # Transfer the state/weights
            algorithm.set_state(
                {
                    "learner_group": {
                        "learner": {
                            "rl_module": {
                                new_module_id: main_module.get_state(),
                            }
                        }
                    }
                }
            )
        else:
            print("Not good enough; will keep learning ...")

        result["league_size"] = self.current_opponent + 2
