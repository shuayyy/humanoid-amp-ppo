from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.sensor import ContactSensor

if TYPE_CHECKING:
  from mjlab_g1.envs.g1_dualarm_rl_env import G1DualarmManagerBasedRlEnv


   

def illegal_contact(
  env: G1DualarmManagerBasedRlEnv,
  sensor_names: tuple[str, ...],
) -> torch.Tensor:
  illegal = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
  for sensor_name in sensor_names:
    sensor: ContactSensor = env.scene[sensor_name]
    assert sensor.data.found is not None
    illegal |= torch.any(sensor.data.found > 0, dim=-1)
  return illegal


def grasp_success_held(
  env: G1DualarmManagerBasedRlEnv,
  left_sensor: str,
  right_sensor: str,
) -> torch.Tensor:
  left_contact_sensor: ContactSensor = env.scene[left_sensor]
  right_contact_sensor: ContactSensor = env.scene[right_sensor]
  assert left_contact_sensor.data.found is not None
  assert right_contact_sensor.data.found is not None

  left_contact = torch.any(left_contact_sensor.data.found > 0, dim=-1)
  right_contact = torch.any(right_contact_sensor.data.found > 0, dim=-1)
  raw_success = left_contact & right_contact & env._object_lifted()

  env.success_hold_buf = torch.where(
    raw_success,
    env.success_hold_buf + 1,
    torch.zeros_like(env.success_hold_buf),
  )

  return env.success_hold_buf >= env.cfg.hold_steps
  
