from configs.test_config import config
import log
from pprint import pprint

if __name__ == "__main__":
    log.VIDEO_SAVE_FREQ = 1
    log.EVALUATION = True
    config = config.evaluation(
        evaluation_duration=100,
        evaluation_duration_unit="episodes",
    )
    algo = config.build_algo()
    # algo.restore(
    #     "/home2/mrgaschenko/Territorial.io-AI/logs/custom_env/all_pretrained_agents/PPO_custom_env_543a3_00000_0_2026-05-28_18-53-11/checkpoint_000028"
    # )
    with open("evaluation_results.py", "w+") as out:
        pprint(algo.evaluate(), stream=out)
