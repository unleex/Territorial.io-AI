from config import config
import log
from pprint import pprint

if __name__ == "__main__":
    log.VIDEO_SAVE_FREQ = 1
    config = config.evaluation(
        evaluation_duration=100,
        evaluation_duration_unit="episodes",
    )
    algo = config.build_algo()
    algo.restore(
        "/home2/mrgaschenko/Territorial.io-AI/logs/custom_env/ChonkyNet/PPO_custom_env_55c63_00000_0_2026-05-15_06-51-21/checkpoint_000007"
    )
    with open("evaluation_results.py", "w+") as out:
        pprint(algo.evaluate(), stream=out)
