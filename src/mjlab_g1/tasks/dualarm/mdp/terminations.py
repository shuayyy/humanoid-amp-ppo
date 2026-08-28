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


def grasp_success_held(env: G1DualarmManagerBasedRlEnv) -> torch.Tensor:
  """Sustained success: object at goal with both grasp contacts for hold_steps.

  Read-only on success_hold_buf -- env.step() owns the increment. The earlier
  version also incremented it here, which double-counted every holding step
  and halved the effective hold requirement.
  """
  return env.success_hold_buf >= env.cfg.hold_steps

