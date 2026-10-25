"""Raw-state readers shared by the SMTrack env, its MDP terms and the eval script.

Everything reads `sim.data.qpos/qvel` directly: derived quantities (xpos, cvel, sensors)
lag qpos by one physics substep inside `step()` until `sim.forward()` runs, while the
learner's state contract (brief §1) is the raw generalized coordinates.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.utils.lab_api.math import quat_apply

if TYPE_CHECKING:
  from mjlab.entity import Entity
  from mjlab.envs import ManagerBasedRlEnv


def free_body_state(
  env: ManagerBasedRlEnv, entity: Entity, env_ids: torch.Tensor | slice | None = None
) -> torch.Tensor:
  """[N,13] = [pos (env-local) 3, quat wxyz 4, lin vel world 3, ang vel WORLD 3].

  MuJoCo's free-joint qvel[3:6] is body-local; the contract wants world frame.
  """
  ids = slice(None) if env_ids is None else env_ids
  ix = entity.indexing
  q = env.sim.data.qpos[ids][:, ix.free_joint_q_adr]
  v = env.sim.data.qvel[ids][:, ix.free_joint_v_adr]
  pos = q[:, :3] - env.scene.env_origins[ids]
  quat = q[:, 3:7]
  ang_w = quat_apply(quat, v[:, 3:6])
  return torch.cat([pos, quat, v[:, :3], ang_w], dim=-1)


def robot_qpos_qvel(
  env: ManagerBasedRlEnv, robot: Entity, env_ids: torch.Tensor | slice | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
  """qpos [N,36] (root pos env-local), qvel [N,35] (root ang vel world)."""
  ids = slice(None) if env_ids is None else env_ids
  root = free_body_state(env, robot, env_ids)
  jp = env.sim.data.qpos[ids][:, robot.indexing.joint_q_adr]
  jv = env.sim.data.qvel[ids][:, robot.indexing.joint_v_adr]
  qpos = torch.cat([root[:, :7], jp], dim=-1)
  qvel = torch.cat([root[:, 7:], jv], dim=-1)
  return qpos, qvel
