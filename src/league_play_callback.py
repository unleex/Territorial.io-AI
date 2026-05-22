from collections import defaultdict
import numpy as np
from ray.rllib.callbacks.callbacks import RLlibCallback


class LeaguePlayCallback(RLlibCallback):
    def __init__(self, win_rate_threshold=0.6):
        super().__init__()
        self.current_opponent = 0
        self.win_rate_threshold = win_rate_threshold
        self._matching_stats = defaultdict(int)

    def on_episode_end(
        self, *, worker, base_env, policies, episode, env_index, **kwargs
    ):
        # Legacy signature uses policy_for() instead of module_for()
        main_agents = [
            agent_id
            for agent_id in episode.get_agents()
            if episode.policy_for(agent_id) == "main"
        ]

        if not main_agents:
            return

        main_won = False
        # Check agent rewards via legacy episode dictionary tracking
        for agent in main_agents:
            # RLlib stores legacy rewards keyed by (agent_id, policy_id)
            if episode.agent_rewards.get((agent, "main"), 0.0) == 1.0:
                main_won = True
                break

        # Populate custom_metrics so Tune automatically routes it to results
        episode.custom_metrics["win_rate"] = float(main_won)

    def on_train_result(self, *, algorithm, result, **kwargs):
        # Legacy RLlib places metrics in the "custom_metrics" root key
        custom_metrics = result.get("custom_metrics", {})
        win_rate = custom_metrics.get("win_rate_mean", 0.0)

        print(f"\n[Tune Callback] Iter={algorithm.iteration} Win Rate={win_rate:.2f}")
        if win_rate > self.win_rate_threshold:
            self.current_opponent += 1
            new_module_id = f"p0_v{self.current_opponent}"  # Renamed to match your p0
            print(f"[Tune Callback] Snapshot to league: {new_module_id}")

            def agent_to_module_mapping_fn(agent_id, episode, **kwargs):
                # We do NOT map bots here, because your environment engine handles them.
                # RLlib only passes the IDs of the agents it is allowed to control.

                # Force at least agent 0 (or your first RLlib slot) to ALWAYS be the active learner
                if agent_id == 0 or self.current_opponent == 0:
                    return "p0"

                # For any other agents RLlib controls, randomize between active and historical
                rng = np.random.default_rng(hash(episode.id_) + agent_id)
                pool = ["p0"] + [
                    f"p0_v{i}" for i in range(1, self.current_opponent + 1)
                ]

                return rng.choice(pool)

            # Weight replication logic (same as before, just using "p0")
            main_module = algorithm.get_module("p0")
            from ray.rllib.core.rl_module.rl_module import RLModuleSpec

            algorithm.add_module(
                module_id=new_module_id,
                module_spec=RLModuleSpec.from_module(main_module),
                new_agent_to_module_mapping_fn=agent_to_module_mapping_fn,
            )

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

            algorithm.workers.foreach_worker(
                lambda worker: worker.set_policy_mapping_fn(agent_to_module_mapping_fn)
            )
        else:
            print("[Tune Callback] Continuing current policy mix.")

        result["league_size"] = self.current_opponent + 2
