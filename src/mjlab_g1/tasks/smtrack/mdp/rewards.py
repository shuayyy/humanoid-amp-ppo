"""Box tracking rewards for the lift task (brief §0, ours: the paper has no object terms).

Both read the raw box qpos, not the one-substep-stale derived pose, so the learner's
analytic reward on s_{t+1} reproduces them exactly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from mjlab.utils.lab_api.math import quat_error_magnitude

from .commands import SMMotionCommand
from .state import free_body_state

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def box_pose_error(
  env: ManagerBasedRlEnv, command_name: str
) -> tuple[torch.Tensor, torch.Tensor]:
  """(squared position error [N], orientation angle [N]) vs the current box reference."""
  cmd = cast(SMMotionCommand, env.command_manager.get_term(command_name))
  assert cmd.box is not None, "box rewards need a command with box_entity_name"
  box = free_body_state(env, cmd.box)
  ref_pos = cmd.box_ref_pos[cmd.time_steps]  # env-local, like free_body_state
  pos_err_sq = torch.sum(torch.square(ref_pos - box[:, :3]), dim=-1)
  ang = quat_error_magnitude(cmd.box_quat_w, box[:, 3:7])
  return pos_err_sq, ang


def box_position_error_exp(
  env: ManagerBasedRlEnv, command_name: str, std: float
) -> torch.Tensor:
  pos_err_sq, _ = box_pose_error(env, command_name)
  return torch.exp(-pos_err_sq / std**2)


def box_orientation_error_exp(
  env: ManagerBasedRlEnv, command_name: str, std: float
) -> torch.Tensor:
  _, ang = box_pose_error(env, command_name)
  return torch.exp(-torch.square(ang) / std**2)
