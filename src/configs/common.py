from ray.rllib.algorithms.ppo import PPOConfig
from prepare_env import ENV_NAME
from callback import MergedCallback

CONFIG_NAME = "H200"
config = (
    PPOConfig()
    .environment(
        env=ENV_NAME,
        clip_actions=True,
        disable_env_checking=False,
    )
    .multi_agent(
        policies={"p0"},
        policy_mapping_fn=(lambda aid, *args, **kwargs: "p0"),
        count_steps_by="agent_steps",
    )
    .debugging(log_level="WARN")
    .framework(framework="torch")
    .api_stack(
        enable_rl_module_and_learner=False,
        enable_env_runner_and_connector_v2=False,
    )
    .callbacks(MergedCallback)
)
