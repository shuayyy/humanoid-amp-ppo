from mjlab_g1.tasks.registry import register_mjlab_task
from mjlab_g1.tasks.locomotion.rl import LocomotionOnPolicyRunner
from .env_cfgs import (
  unitree_g1_locomotion_env_cfg,
)
from .rl_cfg import unitree_g1_locomotion_ppo_runner_cfg



register_mjlab_task(
  task_id="Mjlab-G1-Locomotion",
  env_cfg=unitree_g1_locomotion_env_cfg(),
  play_env_cfg=unitree_g1_locomotion_env_cfg(play=True),
  rl_cfg=unitree_g1_locomotion_ppo_runner_cfg(),
  runner_cls=LocomotionOnPolicyRunner,
)

""" 
Register this task in the MJLab-HUSKY task registry. (src/mjlab_g1/tasks/registry.py)
This stores task_id, env_cfg, play_env_cfg, rl_cfg, and runner_cls in:
  src/mjlab_g1/tasks/registry.py
Later, train/play scripts use the task_id to find and load this task:
  src/mjlab_g1/scripts/train.py
  src/mjlab_g1/scripts/play.py 
"""


""""
A **registry** is a software design pattern used to store things by name and retrieve them later. In many codebases, it is used for tasks, models, datasets, environments, rewards, or plugins. Instead of writing many `if/else` conditions in the training script, the code registers each option once and later loads it using a string name.

For example, in this codebase, `register_mjlab_task(...)` stores the `task_id`, `env_cfg`, `play_env_cfg`, `rl_cfg`, and `runner_cls` inside the task registry. Later, `train.py` or `play.py` uses the `task_id` to find the correct environment config, RL config, and runner class.

This makes the code more modular. To add a new task such as toaster grasp, we only register a new task in the task folder. We do not need to modify the main training script with new `if/else` logic.
 
 """
