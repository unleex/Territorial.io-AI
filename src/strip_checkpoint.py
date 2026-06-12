import os
from ray.rllib.algorithms.algorithm import Algorithm


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


if __name__ == "__main__":
    filtered_checkpoint_from_policy_ids(
        checkpoint_path="/home2/mrgaschenko/Territorial.io-AI/logs/custom_env/time_aware/PPO_custom_env_45afa_00000_0_2026-06-12_11-04-02/checkpoint_000008",
        keep_policy_ids=["p0"],
    )
