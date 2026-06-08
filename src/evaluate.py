from configs.test_config import config
import log
from pprint import pprint

if __name__ == "__main__":
    log.VIDEO_SAVE_FREQ = 1
    log.EVALUATION = True
    config = config.evaluation(
        evaluation_duration=1,
        evaluation_duration_unit="complete_episodes",
        evaluation_num_env_runners=1,
    )
    algo = config.build_algo()
    algo.restore(
        "/home2/mrgaschenko/Territorial.io-AI/logs/custom_env/all_pretrained_agents/PPO_custom_env_dfce1_00000_0_2026-06-08_11-48-52/checkpoint_000027"
    )
    with open("evaluation_results.py", "w+") as out:
        pprint(algo.evaluate(), stream=out)
