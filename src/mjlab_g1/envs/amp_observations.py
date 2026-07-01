"""AMP observation helpers shared by G1 tasks."""

from __future__ import annotations

import torch

from mjlab.utils.lab_api.math import quat_apply_inverse


def g1_rich_amp_observations(robot, num_envs: int, device: str) -> torch.Tensor:
    """Return a motion-style AMP state derived from robot kinematics.

    Layout:
      joint_pos[29], joint_vel[29], root_lin_vel_b[3],
      root_ang_vel_b[3], projected_gravity_b[3]
    """
    gravity_w = torch.zeros(num_envs, 3, device=device)
    gravity_w[:, 2] = -1.0
    projected_gravity_b = quat_apply_inverse(
        robot.data.root_link_quat_w,
        gravity_w,
    )

    return torch.cat(
        (
            robot.data.joint_pos,
            robot.data.joint_vel,
            robot.data.root_link_lin_vel_b,
            robot.data.root_link_ang_vel_b,
            projected_gravity_b,
        ),
        dim=-1,
    )
