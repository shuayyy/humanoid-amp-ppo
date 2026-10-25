"""SMTrack: shared G1 motion-tracking tasks for the SuperMimic vs PPO comparison.

A tiny registry of its own, separate from the AMP `mjlab_g1.tasks.registry`.
`mjlab_g1.tasks.register_tasks()` imports this package while walking the AMP tasks, so it
stays import-light: configs are built on demand.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
  from mjlab.rl import RslRlOnPolicyRunnerCfg
  from mjlab_g1.envs.smtrack_env import SMTrackEnvCfg

TASKS: dict[str, str] = {
  "G1-SMTrack-Walk": "walk",
  "G1-SMTrack-Squat": "squat",
  "G1-SMTrack-Lift": "lift",
}


def list_tasks() -> list[str]:
  return sorted(TASKS)


def task_kind(task_id: str) -> str:
  if task_id not in TASKS:
    raise KeyError(f"unknown SMTrack task '{task_id}'; choose from {list_tasks()}")
  return TASKS[task_id]


def make_env_cfg(task_id: str, play: bool = False, motion_file: str | None = None) -> SMTrackEnvCfg:
  from mjlab_g1.tasks.smtrack.env_cfgs import make_env_cfg as _make

  return _make(task_kind(task_id), play=play, motion_file=motion_file)  # type: ignore[arg-type]


def make_ppo_cfg(task_id: str) -> RslRlOnPolicyRunnerCfg:
  from mjlab_g1.tasks.smtrack.rl_cfgs import ppo_runner_cfg

  task_kind(task_id)
  cfg = ppo_runner_cfg()
  cfg.experiment_name = f"smtrack_{task_kind(task_id)}_ppo"
  return cfg
