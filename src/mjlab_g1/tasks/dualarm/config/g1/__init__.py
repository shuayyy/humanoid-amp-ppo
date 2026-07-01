from mjlab_g1.tasks.registry import register_mjlab_task
from mjlab_g1.tasks.dualarm.rl import DualArmOnPolicyRunner
from .env_cfgs import (
  unitree_g1_dualarm_env_cfg,
)
from .rl_cfg import unitree_g1_dualarm_ppo_runner_cfg



register_mjlab_task(
  task_id="Mjlab-G1-DualArm",
  env_cfg=unitree_g1_dualarm_env_cfg(),
  play_env_cfg=unitree_g1_dualarm_env_cfg(play=True),
  rl_cfg=unitree_g1_dualarm_ppo_runner_cfg(),
  runner_cls=DualArmOnPolicyRunner,
)
