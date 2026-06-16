from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.sensor import ContactSensor
from mjlab.tasks.manipulation.mdp.commands import LiftingCommand
if TYPE_CHECKING:
  from mjlab_g1.envs.g1_dualarm_rl_env import G1DualarmManagerBasedRlEnv


def get_object_pose(env: G1DualarmManagerBasedRlEnv) -> torch.Tensor:
    obj_pos_w = env.toaster.data.root_link_pos_w[:, :3]
    obj_quat_w = env.toaster.data.root_link_quat_w

    return torch.cat([obj_pos_w, obj_quat_w], dim=-1)
#### dual-arm manipulation rewards ####

def hand_to_toaster(
  env: G1DualarmManagerBasedRlEnv, d_scale: float = 1.5
) -> torch.Tensor:
  dis = env._get_hand_toaster_dis()
  dist = torch.norm(dis, dim=-1)  # [num_envs, 2]

  reward_per_hand = torch.exp(-dist / d_scale)
  return reward_per_hand.mean(dim=-1)


def dist_to_toaster(
  env: G1DualarmManagerBasedRlEnv, d_scale: float = 1.5
) -> torch.Tensor:
  root_pos = env.robot.data.root_link_pos_w[:, :3]
  toaster_pos = env.toaster.data.root_link_pos_w[:, :3]

  dist = torch.norm(root_pos - toaster_pos, dim=-1)
  return torch.exp(-dist / d_scale)


def hands_contact(
  env: G1DualarmManagerBasedRlEnv,
  sensor_name: str,
  min_reward_time_s: float = 2.0,
) -> torch.Tensor:
  contact_sensor: ContactSensor = env.scene[sensor_name]

  assert contact_sensor.data.found is not None

  contact = torch.any(contact_sensor.data.found > 0, dim=-1)
  elapsed_s = env.episode_length_buf.float() * env.step_dt
  reward_enabled = elapsed_s >= min_reward_time_s
  return contact.float() * reward_enabled.float()

def hands_at_markers(
  env: G1DualarmManagerBasedRlEnv,
  left_sensor: str,
  right_sensor: str,
  min_reward_time_s: float = 2.0,
) -> torch.Tensor:
  """Return whether both hand-marker contact sensors are active."""
  sensor1: ContactSensor = env.scene[left_sensor]
  sensor2: ContactSensor = env.scene[right_sensor]
  assert sensor1.data.found is not None
  assert sensor2.data.found is not None

  sensor1_contact = torch.any(sensor1.data.found > 0, dim=-1)
  sensor2_contact = torch.any(sensor2.data.found > 0, dim=-1)
  elapsed_s = env.episode_length_buf.float() * env.step_dt
  reward_enabled = elapsed_s >= min_reward_time_s
  return (sensor1_contact & sensor2_contact).float() * reward_enabled.float()


def marker_force(
  env: G1DualarmManagerBasedRlEnv,
  left_sensor: str,
  right_sensor: str,
  min_reward_time_s: float = 2.0,
  target_force: float = 10.0,
) -> torch.Tensor:
  """Reward applying force at both grasp markers while both contacts are active."""
  left_contact_sensor: ContactSensor = env.scene[left_sensor]
  right_contact_sensor: ContactSensor = env.scene[right_sensor]
  assert left_contact_sensor.data.found is not None
  assert right_contact_sensor.data.found is not None
  assert left_contact_sensor.data.force is not None
  assert right_contact_sensor.data.force is not None

  left_contact = torch.any(left_contact_sensor.data.found > 0, dim=-1)
  right_contact = torch.any(right_contact_sensor.data.found > 0, dim=-1)
  both_contact = left_contact & right_contact

  left_force = torch.norm(left_contact_sensor.data.force, dim=-1)
  right_force = torch.norm(right_contact_sensor.data.force, dim=-1)
  left_force = left_force.reshape(env.num_envs, -1).amax(dim=1)
  right_force = right_force.reshape(env.num_envs, -1).amax(dim=1)
  avg_force = 0.5 * (left_force + right_force)
  force_reward = torch.clamp(avg_force / max(target_force, 1.0e-6), 0.0, 1.0)

  elapsed_s = env.episode_length_buf.float() * env.step_dt
  reward_enabled = elapsed_s >= min_reward_time_s
  return force_reward * both_contact.float() * reward_enabled.float()


def lift(
  env: G1DualarmManagerBasedRlEnv,
  left_sensor: str,
  right_sensor: str,
  position_tolerance: float = 0.075,
  command_name: str = "place_pos",
  min_reward_time_s: float = 2.5,
  early_lift_penalty: float = -1.0,
) -> torch.Tensor:
  left_contact_sensor: ContactSensor = env.scene[left_sensor]
  right_contact_sensor: ContactSensor = env.scene[right_sensor]
  assert left_contact_sensor.data.found is not None
  assert right_contact_sensor.data.found is not None

  left_contact = torch.any(left_contact_sensor.data.found > 0, dim=-1)
  right_contact = torch.any(right_contact_sensor.data.found > 0, dim=-1)
  both_contact = left_contact & right_contact
  if not torch.any(both_contact):
    return torch.zeros(env.num_envs, device=env.device)

  command = env.command_manager.get_term(command_name)
  if not isinstance(command, LiftingCommand):
    raise TypeError(
      f"Command '{command_name}' must be a LiftingCommand, got {type(command)}"
    )

  obj_pose = get_object_pose(env)
  obj_pos = obj_pose[:, :3]
  target_pos = command.target_pos

  position_error = torch.norm(obj_pos - target_pos, dim=-1)
  reward = torch.exp(
    -position_error / max(position_tolerance, 1.0e-6)
  )
  reward = torch.where(
    position_error <= position_tolerance,
    torch.ones_like(reward),
    reward,
  )

  elapsed_s = env.episode_length_buf.float() * env.step_dt
  reward_enabled = elapsed_s >= min_reward_time_s
  lifted_early = both_contact & env._object_lifted() & (~reward_enabled)

  gated_reward = reward * both_contact.float() * reward_enabled.float()
  early_penalty = lifted_early.float() * early_lift_penalty
  return gated_reward + early_penalty
  
#### Stability  rewards ####


def yaw_rate_penalty(
  env: G1DualarmManagerBasedRlEnv,
  threshold: float = 0.075,
) -> torch.Tensor:
  """Penalty for body-frame yaw rate outside a deadzone around zero."""
  yaw_rate = env.robot.data.root_link_ang_vel_b[:, 2]
  excess = torch.clamp(torch.abs(yaw_rate) - threshold, min=0.0)
  return torch.square(excess)


def angular_vel_penalty(
  env: G1DualarmManagerBasedRlEnv,
  threshold: float = 0.05,
) -> torch.Tensor:
  """Penalty for base angular motion outside a small deadzone."""
  ang_vel = env.robot.data.root_link_ang_vel_b
  speed = torch.norm(ang_vel, dim=-1)
  excess = torch.clamp(speed - threshold, min=0.0)
  return torch.square(excess)


def linear_vel_penalty(
  env: G1DualarmManagerBasedRlEnv,
  threshold: float = 0.075,
) -> torch.Tensor:
  """Penalty for base linear motion outside a small deadzone."""
  root_lin_vel = env.robot.data.root_link_lin_vel_b
  speed = torch.norm(root_lin_vel, dim=-1)
  excess = torch.clamp(speed - threshold, min=0.0)
  return torch.square(excess)

def feet_slip(
  env: G1DualarmManagerBasedRlEnv,
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

def feet_contact(
  env: G1DualarmManagerBasedRlEnv,
  left_sensor_name: str = "left_feet_ground_contact",
  right_sensor_name: str = "right_feet_ground_contact",
) -> torch.Tensor:
  left_sensor: ContactSensor = env.scene[left_sensor_name]
  right_sensor: ContactSensor = env.scene[right_sensor_name]

  left_contact = left_sensor.data.found
  right_contact = right_sensor.data.found
  assert left_contact is not None
  assert right_contact is not None

  left_in_contact = torch.any(left_contact > 0, dim=-1)
  right_in_contact = torch.any(right_contact > 0, dim=-1)

  return (left_in_contact & right_in_contact).float()

def self_collision_cost(
  env: G1DualarmManagerBasedRlEnv, sensor_name: str
) -> torch.Tensor:
  """Cost that returns whether self-collision was detected by a sensor."""
  sensor: ContactSensor = env.scene[sensor_name]
  assert sensor.data.found is not None
  return sensor.data.found.squeeze(-1).float()

def _illegal_contact_mask(
  env: G1DualarmManagerBasedRlEnv,
  sensor_names: tuple[str, ...],
) -> torch.Tensor:
  illegal = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
  for sensor_name in sensor_names:
    sensor: ContactSensor = env.scene[sensor_name]
    assert sensor.data.found is not None
    illegal |= torch.any(sensor.data.found > 0, dim=-1)
  return illegal

def illegal_contact_penalty(
  env: G1DualarmManagerBasedRlEnv,
  sensor_names: tuple[str, ...],
) -> torch.Tensor:
  return _illegal_contact_mask(env, sensor_names).float()
