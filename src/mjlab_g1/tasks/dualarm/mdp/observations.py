from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.sensor import ContactSensor

if TYPE_CHECKING:
  from mjlab_g1.envs.g1_dualarm_rl_env import G1DualarmManagerBasedRlEnv

from mjlab.utils.lab_api.math import (
  quat_mul,
  quat_apply_inverse,
  quat_conjugate,
)

def get_depth(
    env: G1DualarmManagerBasedRlEnv,
    sensor_name: str = "head_depth",
) -> torch.Tensor:
    sensor = env.scene[sensor_name]
    depth = sensor.data.depth

    if depth is None:
        raise RuntimeError(
            f"Camera sensor '{sensor_name}' has no depth output."
        )

    return depth


def get_depth_features(
    env: G1DualarmManagerBasedRlEnv,
) -> torch.Tensor:
    """Return cached frozen DeFM features with shape [B, 192]."""
    features = env.depth_feature_buf

    expected_shape = (env.num_envs, 192)
    if tuple(features.shape) != expected_shape:
        raise RuntimeError(
            "Expected cached DeFM features with shape "
            f"{expected_shape}, got {tuple(features.shape)}."
        )

    return features


def foot_air_time(env: G1DualarmManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
    sensor: ContactSensor = env.scene[sensor_name]
    sensor_data = sensor.data
    current_air_time = sensor_data.current_air_time
    assert current_air_time is not None
    return current_air_time

def foot_contact(env: G1DualarmManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
    sensor: ContactSensor = env.scene[sensor_name]
    sensor_data =sensor.data
    assert sensor_data.found is not None
    return (sensor_data.found > 0).float()

def foot_contact_forces(env: G1DualarmManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
    sensor: ContactSensor = env.scene[sensor_name]
    sensor_data = sensor.data
    assert sensor_data.force is not None
    forces_flat = sensor_data.force.flatten(start_dim=1)  # [B, N*3]
    return torch.sign(forces_flat) * torch.log1p(torch.abs(forces_flat))

def object_pose(env: G1DualarmManagerBasedRlEnv) -> torch.Tensor:
    obj_pos_w = env.toaster.data.root_link_pos_w[:, :3]
    obj_quat_w = env.toaster.data.root_link_quat_w

    base_pos_w = env.robot.data.root_link_pos_w[:, :3]
    base_quat_w = env.robot.data.root_link_quat_w

    # Vector from robot base to object, expressed in world frame.
    obj_pos_rel_w = obj_pos_w - base_pos_w

    # Convert object position and orientation into robot/base frame.
    obj_pos_b = quat_apply_inverse(base_quat_w, obj_pos_rel_w)
    obj_quat_b = quat_mul(quat_conjugate(base_quat_w), obj_quat_w)

    return torch.cat([obj_pos_b, obj_quat_b], dim=-1)

def left_grasp_marker_pos(env: G1DualarmManagerBasedRlEnv) -> torch.Tensor:
    marker_pos_w = env.toaster.data.site_pos_w[:, env.grasp_site_ids[0], :3]
    base_pos_w = env.robot.data.root_link_pos_w[:, :3]
    base_quat_w = env.robot.data.root_link_quat_w

    marker_rel_w = marker_pos_w - base_pos_w
    return quat_apply_inverse(base_quat_w, marker_rel_w)

def right_grasp_marker_pos(env: G1DualarmManagerBasedRlEnv) -> torch.Tensor:
    marker_pos_w = env.toaster.data.site_pos_w[:, env.grasp_site_ids[1], :3]
    base_pos_w = env.robot.data.root_link_pos_w[:, :3]
    base_quat_w = env.robot.data.root_link_quat_w

    marker_rel_w = marker_pos_w - base_pos_w
    return quat_apply_inverse(base_quat_w, marker_rel_w)

def trajectory_reference_pos(
    env: G1DualarmManagerBasedRlEnv,
) -> torch.Tensor:
    """
    Current reference toaster position expressed in the robot base frame.
    """
    reference_pos_w, _ = env.get_object_trajectory_reference()

    base_pos_w = env.robot.data.root_link_pos_w[:, :3]
    base_quat_w = env.robot.data.root_link_quat_w

    reference_rel_w = reference_pos_w - base_pos_w
    return quat_apply_inverse(base_quat_w, reference_rel_w)
