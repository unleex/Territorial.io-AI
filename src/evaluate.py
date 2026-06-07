from configs.a100_config import config
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
    #     "/home2/mrgaschenko/Territorial.io-AI/logs/custom_env/all_pretrained_agents/PPO_custom_env_8d4b7_00000_0_2026-06-02_19-31-48/checkpoint_000007"
    # )
    with open("evaluation_results.py", "w+") as out:
        pprint(algo.evaluate(), stream=out)
