from configs.mac_config import config
import log
from pprint import pprint

if __name__ == "__main__":
    log.VIDEO_SAVE_FREQ = 1
    log.EVALUATION = True
    config = config.evaluation(
        evaluation_duration=10,
        evaluation_duration_unit="episodes",
    )
    algo = config.build_algo()
    # algo.restore(
    #     "/home2/mrgaschenko/Territorial.io-AI/logs/custom_env/Multiagency/PPO_custom_env_6f21c_00000_0_2026-05-20_16-33-06/checkpoint_000000"
    # )
    with open("evaluation_results.py", "w+") as out:
        pprint(algo.evaluate(), stream=out)
