from mjlab_husky.tasks.registry import register_mjlab_task

from .env_cfgs import (
  unitree_g1_wb_grasp_env_cfg,
)
from .rl_cfg import unitree_g1_wb_grasp_ppo_runner_cfg

# Keep the custom AMP-based WB runner available for future use, but do not
# wire it into the current PPO task registration.
# from mjlab_husky.tasks.wb_grasp.rl import WbGraspOnPolicyRunner


register_mjlab_task(
  task_id="Mjlab-WB-Grasp-Unitree-G1",
  env_cfg=unitree_g1_wb_grasp_env_cfg(),
  play_env_cfg=unitree_g1_wb_grasp_env_cfg(play=True),
  rl_cfg=unitree_g1_wb_grasp_ppo_runner_cfg(),
  runner_cls=None,
)

""" 
Register this task in the MJLab-HUSKY task registry. (src/mjlab_husky/tasks/registry.py)
This stores task_id, env_cfg, play_env_cfg, rl_cfg, and runner_cls in:
  src/mjlab_husky/tasks/registry.py
Later, train/play scripts use the task_id to find and load this task:
  src/mjlab_husky/scripts/train.py
  src/mjlab_husky/scripts/play.py 
"""


""""
A **registry** is a software design pattern used to store things by name and retrieve them later. In many codebases, it is used for tasks, models, datasets, environments, rewards, or plugins. Instead of writing many `if/else` conditions in the training script, the code registers each option once and later loads it using a string name.

For example, in this codebase, `register_mjlab_task(...)` stores the `task_id`, `env_cfg`, `play_env_cfg`, `rl_cfg`, and `runner_cls` inside the task registry. Later, `train.py` or `play.py` uses the `task_id` to find the correct environment config, RL config, and runner class.

This makes the code more modular. To add a new task such as toaster grasp, we only register a new task in the task folder. We do not need to modify the main training script with new `if/else` logic.
 
 """
