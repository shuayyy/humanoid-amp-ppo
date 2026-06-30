"""Task registry system for managing environment registration and creation."""
from __future__ import annotations


"""MANAGERS - MJLAB BUILTIN MODULE
Managers are MJLab modules that handle separate parts of the env: actions, observations, rewards, terminations, commands, and reset/events.

Your task config tells these managers what to compute for your specific task.

"""

from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnvCfg
  from mjlab_g1.rl import RslRlOnPolicyRunnerCfg


@dataclass
class _TaskCfg:
  env_cfg: ManagerBasedRlEnvCfg # training environment config
  play_env_cfg: ManagerBasedRlEnvCfg # evaluation/play environment config
  rl_cfg: RslRlOnPolicyRunnerCfg  # PPO/AMP training config
  runner_cls: type | None  # runner class used to train


# Private module-level registry: task_id -> task config.
_REGISTRY: dict[str, _TaskCfg] = {}


def _ensure_registered() -> None:
  from mjlab_g1.tasks import register_tasks

  register_tasks()


def register_mjlab_task(
  task_id: str,
  env_cfg: ManagerBasedRlEnvCfg,
  play_env_cfg: ManagerBasedRlEnvCfg,
  rl_cfg: RslRlOnPolicyRunnerCfg,
  runner_cls: type | None = None,
) -> None:
  """Register an environment task.

  Args:
    task_id: Unique task identifier (e.g., "Mjlab-Velocity-Rough-Unitree-Go1").
    env_cfg: Environment configuration used for training.
    play_env_cfg: Environment configuration in "play" mode.
    rl_cfg: RL runner configuration.
    runner_cls: Optional custom runner class. If None, uses OnPolicyRunner.
  """
  if task_id in _REGISTRY:
    raise ValueError(f"Task '{task_id}' is already registered")
  _REGISTRY[task_id] = _TaskCfg(env_cfg, play_env_cfg, rl_cfg, runner_cls)


def list_tasks() -> list[str]:
  """List all registered task IDs."""
  _ensure_registered()
  return sorted(_REGISTRY.keys())


def load_env_cfg(task_name: str, play: bool = False) -> ManagerBasedRlEnvCfg:
  """Load environment configuration for a task.

  Returns a deep copy to prevent mutation of the registered config.
  """
  _ensure_registered()
  return deepcopy(
    _REGISTRY[task_name].env_cfg if not play else _REGISTRY[task_name].play_env_cfg
  )


def load_rl_cfg(task_name: str) -> RslRlOnPolicyRunnerCfg:
  """Load RL configuration for a task.

  Returns a deep copy to prevent mutation of the registered config.
  """
  _ensure_registered()
  return deepcopy(_REGISTRY[task_name].rl_cfg)


def load_runner_cls(task_name: str) -> type | None:
  """Load the runner class for a task.

  If None, the default OnPolicyRunner will be used.
  """
  _ensure_registered()
  return _REGISTRY[task_name].runner_cls
