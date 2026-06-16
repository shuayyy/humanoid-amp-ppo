from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.sensor import ContactSensor

if TYPE_CHECKING:
  from mjlab_g1.envs.g1_locomotion_rl_env import G1LocomotionManagerBasedRlEnv


def root_pos(env: G1LocomotionManagerBasedRlEnv) -> torch.Tensor:
    return env.robot.data.root_link_pos_w[:, :3]

def root_ori(env: G1LocomotionManagerBasedRlEnv) -> torch.Tensor:
    return env.robot.data.root_link_quat_w

def foot_air_time(env: G1LocomotionManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
    sensor: ContactSensor = env.scene[sensor_name]
    sensor_data = sensor.data
    current_air_time = sensor_data.current_air_time
    assert current_air_time is not None
    return current_air_time

def foot_contact(env: G1LocomotionManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
    sensor: ContactSensor = env.scene[sensor_name]
    sensor_data =sensor.data
    assert sensor_data.found is not None
    return (sensor_data.found > 0).float()

def foot_contact_forces(env: G1LocomotionManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
    sensor: ContactSensor = env.scene[sensor_name]
    sensor_data = sensor.data
    assert sensor_data is not None
    forces_flat = sensor_data.force.flatten(start_dim=1)  # [B, N*3]
    return torch.sign(forces_flat) * torch.log1p(torch.abs(forces_flat))
