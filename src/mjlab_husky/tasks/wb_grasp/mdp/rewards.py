from __future__ import annotations

from typing import TYPE_CHECKING

import torch
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


def get_object_pose(env: G1GraspManagerBasedRlEnv) -> torch.Tensor:
    obj_pos_w = env.toaster.data.root_link_pos_w[:, :3]
    obj_quat_w = env.toaster.data.root_link_quat_w

    return torch.cat([obj_pos_w, obj_quat_w], dim=-1)

"""Hekper fucntions for rewards"""

#### locomotion phase rewards ####

def track_lin_vel(
  env: G1GraspManagerBasedRlEnv,
  command_name: str,
  std: float,
  y_deadzone: tuple[float, float] = (-0.0075, 0.075),
  z_deadzone: tuple[float, float] = (-0.0075, 0.075),
) -> torch.Tensor:
  """Reward tracking commanded body-frame linear velocity."""
  root_lin_vel = env.robot.data.root_link_lin_vel_b
  vx = root_lin_vel[:, 0]
  vy = root_lin_vel[:, 1]
  vz = root_lin_vel[:, 2]
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  target_vx = command[:, 0]
  x_error = torch.square(vx - target_vx)

  y_low, y_high = y_deadzone
  z_low, z_high = z_deadzone
  y_error = torch.where(
    vy < y_low,
    torch.square(vy - y_low),
    torch.where(vy > y_high, torch.square(vy - y_high), torch.zeros_like(vy)),
  )
  z_error = torch.where(
    vz < z_low,
    torch.square(vz - z_low),
    torch.where(vz > z_high, torch.square(vz - z_high), torch.zeros_like(vz)),
  )
  error = x_error + y_error + z_error
  return torch.exp(-error / std**2)


def yaw_rate_penalty(
  env: G1GraspManagerBasedRlEnv,
  command_name: str,
  threshold: float = 0.1,
) -> torch.Tensor:
  """Penalty for body-frame yaw rate outside a deadzone around the command."""
  yaw_rate = env.robot.data.root_link_ang_vel_b[:, 2]
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  target_yaw_rate = command[:, 2]
  excess = torch.clamp(torch.abs(yaw_rate - target_yaw_rate) - threshold, min=0.0)
  return torch.square(excess)

def feet_air_time(
  env: G1GraspManagerBasedRlEnv,
  sensor_name: str,
  command_name: str,
  threshold_min: float = 0.2,
  threshold_max: float = 0.5,
  command_threshold: float = 0.5,
) -> torch.Tensor:
  """Reward feet air time."""
  sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = sensor.data

  current_air_time = sensor_data.current_air_time
  assert current_air_time is not None

  in_range = (current_air_time > threshold_min) & (current_air_time < threshold_max)
  reward = torch.sum(in_range.float(), dim=1)

  command = env.command_manager.get_command(command_name)
  if command is not None:
    vx = torch.abs(command[:, 0])
    scale = (vx > command_threshold).float()
    reward *= scale

  return reward

def feet_slip(
  env: G1GraspManagerBasedRlEnv,
  left_sensor_name: str = "left_feet_ground_contact",
  right_sensor_name: str = "right_feet_ground_contact",
  threshold_min: float = 0.0,
) -> torch.Tensor:
  left_sensor: ContactSensor = env.scene[left_sensor_name]
  right_sensor: ContactSensor = env.scene[right_sensor_name]

  left_contact = left_sensor.data.found
  right_contact = right_sensor.data.found
  assert left_contact is not None
  assert right_contact is not None

  contact = torch.cat([left_contact, right_contact], dim=1)

  feet_body_ids = torch.as_tensor(
    env.feet_body_ids, device=env.device, dtype=torch.long
  )
  foot_vel_xy = env.robot.data.body_link_lin_vel_w[:, feet_body_ids, :2]
  foot_speed = torch.norm(foot_vel_xy, dim=-1)

  slip = torch.clamp(foot_speed - threshold_min, min=0.0)
  return torch.sum(slip * (contact > 0).float(), dim=1)

def soft_landing(
  env: G1GraspManagerBasedRlEnv,
  sensor_name: str,
  threshold_min: float = 0.05,
) -> torch.Tensor:
  """Penalize high impact forces at landing to encourage soft footfalls."""
  contact_sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = contact_sensor.data
  assert sensor_data.force is not None

  forces = sensor_data.force  # [B, N, 3]
  force_magnitude = torch.norm(forces, dim=-1)  # [B, N]

  first_contact = contact_sensor.compute_first_contact(dt=env.step_dt)  # [B, N]
  landing_impact = (
    torch.clamp(force_magnitude - threshold_min, min=0.0) * first_contact.float()
  )  # [B, N]

  cost = torch.sum(landing_impact, dim=1)  # [B]
  return cost

def torso_upright(
  env: G1GraspManagerBasedRlEnv,
  std: float = 1.5,
  tilt_threshold: float = 0.0,
) -> torch.Tensor:
  root_quat = env.robot.data.root_link_quat_w

  local_up = torch.zeros(env.num_envs, 3, device=env.device)
  local_up[:, 2] = 1.0

  torso_up_w = quat_apply(root_quat, local_up)
  xy_norm = torch.norm(torso_up_w[:, :2], dim=1)
  tilt_error = torch.clamp(xy_norm - tilt_threshold, min=0.0)
  xy_squared = torch.square(tilt_error)

  return torch.exp(-xy_squared / std**2)


#### regularization rewards ####





  
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



"""" Commented functions for the grasp rewards, which are not used in the current version of the environment."""


"""
#### grasp phase rewards ####

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


def hands_contact(env: G1GraspManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  contact_sensor: ContactSensor = env.scene[sensor_name]

  assert contact_sensor.data.found is not None

  contact = torch.any(contact_sensor.data.found > 0, dim=-1)

  return contact.float()

def hands_at_markers(env: G1GraspManagerBasedRlEnv, left_sensor: str, right_sensor: str) -> torch.Tensor:
  ###return whether the specified contact sensor detected contact, which indicates whether the hand is at the marker
  sensor1: ContactSensor = env.scene[left_sensor]
  sensor2: ContactSensor = env.scene[right_sensor]
  assert sensor1.data.found is not None
  assert sensor2.data.found is not None

  sensor1_contact = torch.any(sensor1.data.found > 0, dim=-1)
  sensor2_contact = torch.any(sensor2.data.found > 0, dim=-1)

  return (sensor1_contact & sensor2_contact).float()

# def lift(
#   env: G1GraspManagerBasedRlEnv,
#   left_sensor: str,
#   right_sensor: str,
#   xy_penalty_weight: float = 0.1,
# ) -> torch.Tensor:
#   marker_contact = hands_at_markers(
#     env,
#     left_sensor=left_sensor,
#     right_sensor=right_sensor,
#   )
#   if not torch.any(marker_contact):
#     return torch.zeros(env.num_envs, device=env.device)
#
#   obj_pose = get_object_pose(env)
#   obj_pos = obj_pose[:, :3]
#   target_pos = env.object_lift_target_pos_w
#
#   height_error = torch.abs(obj_pos[:, 2] - target_pos[:, 2])
#   height_reward = 1.0 - torch.clamp(
#     height_error / max(env.cfg.lift_height_thresh, 1.0e-6),
#     min=0.0,
#     max=1.0,
#   )
#
#   xy_error = torch.norm(obj_pos[:, :2] - target_pos[:, :2], dim=-1)
#   reward = torch.clamp(height_reward - xy_penalty_weight * xy_error, min=0.0, max=1.0)
#
#   return reward * marker_contact

def lift(
  env: G1GraspManagerBasedRlEnv,
  left_sensor: str,
  right_sensor: str,
  sensor_name: str = "toaster_contact",
  xy_penalty_weight: float = 0.1,
) -> torch.Tensor:
  contact = hands_contact(env, sensor_name=sensor_name)
  if not torch.any(contact):
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

  return reward * contact
  
  """
