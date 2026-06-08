from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor

from mjlab.utils.lab_api.math import (
  quat_apply,
  wrap_to_pi,
  quat_mul,
  quat_apply_inverse,
  quat_error_magnitude,
  euler_xyz_from_quat
)
if TYPE_CHECKING:
  from mjlab_husky.envs.g1_wb_grasp_rl_env import G1GraspManagerBasedRlEnv


_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")
"""Hekper fucntions for rewards"""

def get_object_pose(env: G1GraspManagerBasedRlEnv) -> torch.Tensor:
    obj_pos_w = env.toaster.data.root_link_pos_w[:, :3]
    obj_quat_w = env.toaster.data.root_link_quat_w

    return torch.cat([obj_pos_w, obj_quat_w], dim=-1)

"""Hekper fucntions for rewards"""

#### reach phase rewards ####
def hand_to_toaster(env: G1GraspManagerBasedRlEnv, d_scale: float = 1.5) -> torch.Tensor:
  dis = env._get_hand_toaster_dis()
  dist = torch.norm(dis, dim=-1)  # [num_envs, 2]

  reward_per_hand = torch.exp(-dist / d_scale)
  return reward_per_hand.mean(dim=-1)

def dist_to_toaster(env: G1GraspManagerBasedRlEnv, d_scale: float = 1.5) -> torch.Tensor:
  root_pos = env.robot.data.root_link_pos_w[:, :3]
  toaster_pos = env.toaster.data.root_link_pos_w[:, :3]

  dist = torch.norm(root_pos - toaster_pos, dim=-1)
  reward = torch.exp(-dist / d_scale)
  return reward

#### grasp phase rewards ####
def hands_contact(env: G1GraspManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  contact_sensor: ContactSensor = env.scene[sensor_name]

  assert contact_sensor.data.found is not None

  contact = torch.any(contact_sensor.data.found > 0, dim=-1)

  return contact.float()

def hands_at_markers(env: G1GraspManagerBasedRlEnv, left_sensor: str, right_sensor: str) -> torch.Tensor:
  """return whether the specified contact sensor detected contact, which indicates whether the hand is at the marker."""
  sensor1: ContactSensor = env.scene[left_sensor]
  sensor2: ContactSensor = env.scene[right_sensor]
  assert sensor1.data.found is not None
  assert sensor2.data.found is not None

  sensor1_contact = torch.any(sensor1.data.found > 0, dim=-1)
  sensor2_contact = torch.any(sensor2.data.found > 0, dim=-1)

  return (sensor1_contact & sensor2_contact).float()

def lift(
  env: G1GraspManagerBasedRlEnv,
  left_sensor: str,
  right_sensor: str,
  xy_penalty_weight: float = 0.1,
) -> torch.Tensor:
  marker_contact = hands_at_markers(
    env,
    left_sensor=left_sensor,
    right_sensor=right_sensor,
  )
  if not torch.any(marker_contact):
    return torch.zeros(env.num_envs, device=env.device)

  obj_pose = get_object_pose(env)
  obj_pos = obj_pose[:, :3]
  target_pos = env.object_lift_target_pos_w

  height_error = torch.abs(obj_pos[:, 2] - target_pos[:, 2])
  height_reward = 1.0 - torch.clamp(
    height_error / max(env.cfg.lift_height_thresh, 1.0e-6),
    min=0.0,
    max=1.0,
  )

  xy_error = torch.norm(obj_pos[:, :2] - target_pos[:, :2], dim=-1)
  reward = torch.clamp(height_reward - xy_penalty_weight * xy_error, min=0.0, max=1.0)

  return reward * marker_contact

#### regularization rewards ####

def torso_upright(env: G1GraspManagerBasedRlEnv, d_scale: float = 1.5) -> torch.Tensor:
  root_quat = env.robot.data.root_link_quat_w

  local_up = torch.zeros(env.num_envs, 3, device=env.device)
  local_up[:, 2] = 1.0

  torso_up_w = quat_apply(root_quat, local_up)

  upright_error = 1.0 - torso_up_w[:, 2]
  reward = torch.exp(-upright_error / d_scale)

  return reward

def feet_stumble(
  env: G1GraspManagerBasedRlEnv,
  left_sensor_name: str = "left_feet_ground_contact",
  right_sensor_name: str = "right_feet_ground_contact",
) -> torch.Tensor:
  left_sensor: ContactSensor = env.scene[left_sensor_name]
  right_sensor: ContactSensor = env.scene[right_sensor_name]

  left_force = left_sensor.data.force
  right_force = right_sensor.data.force
  assert left_force is not None
  assert right_force is not None

  force = torch.cat([left_force, right_force], dim=1)

  horizontal = torch.norm(force[..., :2], dim=-1)
  vertical = torch.abs(force[..., 2])

  return torch.any(horizontal > 4.0 * vertical, dim=1).float()

def feet_slip(
  env: G1GraspManagerBasedRlEnv,
  left_sensor_name: str = "left_feet_ground_contact",
  right_sensor_name: str = "right_feet_ground_contact",
  contact_force_threshold: float = 5.0,
) -> torch.Tensor:
  left_sensor: ContactSensor = env.scene[left_sensor_name]
  right_sensor: ContactSensor = env.scene[right_sensor_name]

  left_force = left_sensor.data.force
  right_force = right_sensor.data.force
  assert left_force is not None
  assert right_force is not None

  force = torch.cat([left_force, right_force], dim=1)
  contact = torch.abs(force[..., 2]) > contact_force_threshold

  foot_vel_xy = env.robot.data.body_link_lin_vel_w[:, env.feet_body_ids, :2]
  foot_speed = torch.norm(foot_vel_xy, dim=-1)

  slip = torch.sqrt(torch.clamp(foot_speed, min=0.0))
  return torch.sum(slip * contact.float(), dim=1)

def at_least_one_foot_contact(
  env: G1GraspManagerBasedRlEnv,
  left_sensor_name: str = "left_feet_ground_contact",
  right_sensor_name: str = "right_feet_ground_contact",
  contact_force_threshold: float = 5.0,
  illegal_sensor_names: tuple[str, ...] = (),
) -> torch.Tensor:
  left_sensor: ContactSensor = env.scene[left_sensor_name]
  right_sensor: ContactSensor = env.scene[right_sensor_name]

  left_force = left_sensor.data.force
  right_force = right_sensor.data.force
  assert left_force is not None
  assert right_force is not None

  left_contact = torch.norm(left_force, dim=-1) > contact_force_threshold
  right_contact = torch.norm(right_force, dim=-1) > contact_force_threshold

  contact = torch.cat([left_contact, right_contact], dim=1)

  reward = torch.any(contact, dim=1).float()
  if illegal_sensor_names:
    reward *= (~_illegal_contact_mask(env, illegal_sensor_names)).float()
  return reward

def self_collision_cost(env: G1GraspManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  """Cost that returns whether self-collision was detected by a sensor."""
  sensor: ContactSensor = env.scene[sensor_name]
  assert sensor.data.found is not None
  return sensor.data.found.squeeze(-1).float()

def _illegal_contact_mask(
  env: G1GraspManagerBasedRlEnv,
  sensor_names: tuple[str, ...],
) -> torch.Tensor:
  illegal = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
  for sensor_name in sensor_names:
    sensor: ContactSensor = env.scene[sensor_name]
    assert sensor.data.found is not None
    illegal |= torch.any(sensor.data.found > 0, dim=-1)
  return illegal

def illegal_contact_penalty(
  env: G1GraspManagerBasedRlEnv,
  sensor_names: tuple[str, ...],
) -> torch.Tensor:
  return _illegal_contact_mask(env, sensor_names).float()
