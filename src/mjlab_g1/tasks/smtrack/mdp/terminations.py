from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from .commands import SMMotionCommand
from .rewards import box_pose_error

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def motion_clip_end(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """True on the step whose reward was scored against the last clip frame.

  Terminations run before the command advances, so `time_steps` here is the frame this
  step's reward used. Registered with time_out=True: the clip end is a truncation.
  """
  cmd = cast(SMMotionCommand, env.command_manager.get_term(command_name))
  return cmd.time_steps >= cmd.num_frames - 1


def bad_box_pos(env: ManagerBasedRlEnv, command_name: str, threshold: float) -> torch.Tensor:
  pos_err_sq, _ = box_pose_error(env, command_name)
  return pos_err_sq > threshold**2
