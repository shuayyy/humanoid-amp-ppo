"""SMTrack observation terms (brief §2).

actor_hist robot terms are mjlab's own (joint_pos_rel biased, joint_vel_rel, imu gyro,
projected gravity, last action), which `supermimic.features.obs_frame` reproduces from raw
state. The box terms and the goal call the learner's `features` functions directly, so the
simulator and imagination compute them with the same code (brief §2).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.utils.lab_api.math import matrix_from_quat, quat_apply_inverse, quat_inv, quat_mul
from mjlab_g1.rl.supermimic import features

if TYPE_CHECKING:
  from mjlab_g1.envs.smtrack_env import SMTrackEnv

_ROBOT_FRAME_DIM = features.obs_frame_dim(False)  # 93; box terms follow it in obs_frame


def _box_obs(env: SMTrackEnv) -> torch.Tensor:
  frame = features.obs_frame(
    env.get_state(), None, env.action_manager.action, env.get_xi(), has_box=True
  )
  return frame[:, _ROBOT_FRAME_DIM:]


def box_pos_b(env: SMTrackEnv) -> torch.Tensor:
  """Box position in the pelvis frame [N,3]."""
  return _box_obs(env)[:, :3]


def box_rot6d_b(env: SMTrackEnv) -> torch.Tensor:
  """rot6d(R_pelvisᵀ R_box) [N,6]."""
  return _box_obs(env)[:, 3:9]


def goal_frames(env: SMTrackEnv) -> torch.Tensor:
  """K=10 future reference frames relative to the robot, 640 (walk) / 730 (lift)."""
  s = env.get_state()
  kin = features.forward_kin(env.torch_fk, s, with_vel=False)
  return features.goal(
    s,
    kin,
    env.motion_lib,
    env.get_ref_index(),
    K=features.GOAL_K,
    offset=env.cfg.reward_ref_offset,
    has_box=env.has_box,
  )


def box_state_b(env: SMTrackEnv) -> torch.Tensor:
  """Critic-only box state in the pelvis frame: pos 3, rot6d 6, lin vel 3, ang vel 3."""
  s = env.get_state()
  q_pel = s[:, 3:7]
  box = s[:, features.ROBOT_STATE_DIM :]
  rot = matrix_from_quat(quat_mul(quat_inv(q_pel), box[:, 3:7]))[..., :2]
  return torch.cat(
    [
      quat_apply_inverse(q_pel, box[:, :3] - s[:, :3]),
      rot.reshape(-1, 6),
      quat_apply_inverse(q_pel, box[:, 7:10]),
      quat_apply_inverse(q_pel, box[:, 10:13]),
    ],
    dim=-1,
  )
