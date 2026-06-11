from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor

if TYPE_CHECKING:
  from mjlab_husky.envs.g1_wb_grasp_rl_env import G1GraspManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")
from mjlab.utils.lab_api.math import (
  quat_apply,
  quat_mul,
  quat_apply_inverse,
  quat_conjugate,
  wrap_to_pi,
)


def root_pos(env: G1GraspManagerBasedRlEnv) -> torch.Tensor:
    return env.robot.data.root_link_pos_w[:, :3]

def root_ori(env: G1GraspManagerBasedRlEnv) -> torch.Tensor:
    return env.robot.data.root_link_quat_w

def foot_air_time(env: G1GraspManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
    sensor: ContactSensor = env.scene[sensor_name]
    sensor_data = sensor.data
    current_air_time = sensor_data.current_air_time
    assert current_air_time is not None
    return current_air_time

def foot_contact(env: G1GraspManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
    sensor: ContactSensor = env.scene[sensor_name]
    sensor_data =sensor.data
    assert sensor_data.found is not None
    return (sensor_data.found > 0).float()

def foot_contact_forces(env: G1GraspManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
    sensor: ContactSensor = env.scene[sensor_name]
    sensor_data = sensor.data
    assert sensor_data is not None
    forces_flat = sensor_data.force.flatten(start_dim=1)  # [B, N*3]
    return torch.sign(forces_flat) * torch.log1p(torch.abs(forces_flat))


""" Gtrasping observations are being commented """

"""

def vision(env: G1GraspManagerBasedRlEnv) -> torch.Tensor:
# Placeholder only. Do not use this in env_cfg until camera obs is implemented.
    return torch.zeros(env.num_envs, 1, device=env.device)


def heading(env: G1GraspManagerBasedRlEnv) -> torch.Tensor:
    return env.robot.data.heading_w.unsqueeze(-1)


def object_pose(env: G1GraspManagerBasedRlEnv) -> torch.Tensor:
    obj_pos_w = env.toaster.data.root_link_pos_w[:, :3]
    obj_quat_w = env.toaster.data.root_link_quat_w

    base_pos_w = env.robot.data.root_link_pos_w[:, :3]
    base_quat_w = env.robot.data.root_link_quat_w

    # Vector from robot base to object, still expressed in world frame.
    obj_pos_rel_w = obj_pos_w - base_pos_w

    # Convert object position and orientation into robot/base frame.
    obj_pos_b = quat_apply_inverse(base_quat_w, obj_pos_rel_w)
    obj_quat_b = quat_mul(quat_conjugate(base_quat_w), obj_quat_w)

    return torch.cat([obj_pos_b, obj_quat_b], dim=-1)

def left_grasp_marker_pos(env: G1GraspManagerBasedRlEnv) -> torch.Tensor:
    marker_pos_w = env.toaster.data.site_pos_w[:, env.grasp_site_ids[0], :3]
    base_pos_w = env.robot.data.root_link_pos_w[:, :3]
    base_quat_w = env.robot.data.root_link_quat_w

    marker_rel_w = marker_pos_w - base_pos_w
    return quat_apply_inverse(base_quat_w, marker_rel_w)

def right_grasp_marker_pos(env: G1GraspManagerBasedRlEnv) -> torch.Tensor:
    marker_pos_w = env.toaster.data.site_pos_w[:, env.grasp_site_ids[1], :3]
    base_pos_w = env.robot.data.root_link_pos_w[:, :3]
    base_quat_w = env.robot.data.root_link_quat_w

    marker_rel_w = marker_pos_w - base_pos_w
    return quat_apply_inverse(base_quat_w, marker_rel_w)

def dist_to_object(env: G1GraspManagerBasedRlEnv) -> torch.Tensor:
    # Placeholder object distance scalar.
    obj_pos_w = env.toaster.data.root_link_pos_w[:, :3]
    base_pos_w = env.robot.data.root_link_pos_w[:, :3]
    return torch.norm(obj_pos_w - base_pos_w, dim=-1, keepdim=True)

def place_pos(env: G1GraspManagerBasedRlEnv) -> torch.Tensor:
    # Placeholder target place position: [x, y, z]
    return torch.zeros(env.num_envs, 3, device=env.device)

    
"""