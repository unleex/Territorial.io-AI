import os
from ray.rllib.algorithms.algorithm import Algorithm
from ray.tune import Tuner


def filtered_checkpoint_from_policy_ids(
    checkpoint_path: str,
    keep_policy_ids: list[str],
    out_dir: str | None = None,
):
    algo = Algorithm.from_checkpoint(checkpoint_path, policy_ids=keep_policy_ids)

    all_policy_ids = list(algo.get_weights().keys())

    missing = [pid for pid in keep_policy_ids if pid not in all_policy_ids]
    if missing:
        raise KeyError(f"Policies not found in checkpoint: {missing}")

    for pid in all_policy_ids:
        if pid not in keep_policy_ids:
            algo.remove_policy(pid)

    if out_dir is None:
        out_dir = checkpoint_path + "_filtered"
    os.makedirs(out_dir, exist_ok=True)

    return algo.save(out_dir)


filtered_checkpoint_from_policy_ids(
    checkpoint_path="/home2/mrgaschenko/Territorial.io-AI/logs/custom_env/all_pretrained_agents/PPO_custom_env_a593a_00000_0_2026-06-02_14-46-09/checkpoint_000008",
    keep_policy_ids=["p0"],
)
