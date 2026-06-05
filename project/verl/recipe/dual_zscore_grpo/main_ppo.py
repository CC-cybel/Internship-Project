"""PPO entrypoint with recipe-local Dual Z-Score GRPO registered."""

import hydra
import ray

from recipe.dual_zscore_grpo import dual_zscore_advantage  # noqa: F401
from verl.experimental.reward_loop import migrate_legacy_reward_impl
from verl.trainer.main_ppo import TaskRunner, run_ppo
from verl.utils.device import auto_set_device


class DualZScoreTaskRunner(TaskRunner):
    """TaskRunner that registers the recipe-local estimator inside the Ray actor."""

    def run(self, config):
        from recipe.dual_zscore_grpo import dual_zscore_advantage  # noqa: F401

        return super().run(config)


@hydra.main(config_path="../../verl/trainer/config", config_name="ppo_trainer", version_base=None)
def main(config):
    auto_set_device(config)
    config = migrate_legacy_reward_impl(config)
    run_ppo(config, task_runner_class=ray.remote(num_cpus=1)(DualZScoreTaskRunner))


if __name__ == "__main__":
    main()
