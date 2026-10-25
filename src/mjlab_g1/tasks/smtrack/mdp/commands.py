"""MotionCommand variant for SMTrack: no clip wrap, optional box reference.

mjlab's command teleports the robot to a freshly sampled frame when `time_steps` passes
the clip end (`_update_command`), with no done signal. SMTrack ends episodes at the clip
end instead (`terminations.motion_clip_end`, a time-out), so here the frame index only
clamps at F-1; the clamp is a guard for the reset step and for auto_reset=False callers.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch

from mjlab.tasks.tracking.mdp import MotionCommand, MotionCommandCfg
from mjlab.utils.lab_api.math import quat_apply, quat_inv, quat_mul, sample_uniform, yaw_quat

from .box import BOX_HALF_EXTENTS

if TYPE_CHECKING:
  from mjlab.entity import Entity
  from mjlab.envs import ManagerBasedRlEnv


class SMMotionCommand(MotionCommand):
  cfg: SMMotionCommandCfg

  def __init__(self, cfg: SMMotionCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self.num_frames = int(self.motion.time_step_total)
    self.box: Entity | None = None
    if cfg.box_entity_name is not None:
      self.box = env.scene[cfg.box_entity_name]
      data = np.load(cfg.motion_file)
      missing = [k for k in ("box_pos_w", "box_quat_w") if k not in data.files]
      if missing:
        raise KeyError(
          f"{cfg.motion_file} lacks {missing}; run src/mjlab_g1/scripts/make_box_reference.py"
        )
      self.box_ref_pos = torch.tensor(data["box_pos_w"], dtype=torch.float32, device=self.device)
      self.box_ref_quat = torch.tensor(data["box_quat_w"], dtype=torch.float32, device=self.device)
      self.grasp_frame = int(data["grasp_frame"]) if "grasp_frame" in data.files else -1
      # The reference (spawn height, floor clamp) was built for one box size.
      if "box_half_extents" in data.files and not np.allclose(
        data["box_half_extents"], BOX_HALF_EXTENTS, atol=1e-6
      ):
        raise ValueError(
          f"{cfg.motion_file} was made for box half extents {data['box_half_extents']}, "
          f"the box entity has {BOX_HALF_EXTENTS}"
        )

  @property
  def box_pos_w(self) -> torch.Tensor:
    """Reference box position (world, env origin added) at the current frame [N,3]."""
    return self.box_ref_pos[self.time_steps] + self._env.scene.env_origins

  @property
  def box_quat_w(self) -> torch.Tensor:
    return self.box_ref_quat[self.time_steps]

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    super()._resample_command(env_ids)
    self._place_box(env_ids)

  def reset_to_frame(self, env_ids: torch.Tensor, frame: int) -> None:
    super().reset_to_frame(env_ids, frame)
    self._place_box(env_ids)

  def _place_box(self, env_ids: torch.Tensor) -> None:
    """Box at its reference pose for the start frame, at rest (in hand if k >= grasp).

    The robot was just written at ref[k] plus mjlab's reset pose noise (xy +-5 cm, yaw
    +-0.2 rad). For in-hand starts the box follows that xy + yaw offset (mjlab's
    re-anchoring convention); at the world reference it would sit up to ~10 cm into one
    palm and get kicked out on the first substep. Before the grasp it stays at the world
    reference on the floor, where the robot has to walk its own offset back.
    """
    if self.box is None or len(env_ids) == 0:
      return
    k = self.time_steps[env_ids]
    origins = self._env.scene.env_origins[env_ids]
    pos = self.box_ref_pos[k] + origins
    quat = self.box_ref_quat[k]
    if self.grasp_frame >= 0:
      held = (k >= self.grasp_frame)[:, None]
      # Raw qpos: the root state was written just now, derived poses are not updated yet.
      root = self._env.sim.data.qpos[env_ids][:, self.robot.indexing.free_joint_q_adr]
      ref_root_pos = self.motion.body_pos_w[k, 0] + origins  # body 0 = pelvis (root)
      dyaw = yaw_quat(quat_mul(root[:, 3:7], quat_inv(self.motion.body_quat_w[k, 0])))
      rel = quat_apply(dyaw, pos - ref_root_pos)
      moved = torch.cat([root[:, :2] + rel[:, :2], pos[:, 2:3]], dim=-1)
      pos = torch.where(held, moved, pos)
      quat = torch.where(held, quat_mul(dyaw, quat), quat)
    vel = torch.zeros(len(env_ids), 6, device=self.device)
    self.box.write_root_state_to_sim(torch.cat([pos, quat, vel], dim=-1), env_ids)
    self.box.reset(env_ids=env_ids)

  def _adaptive_sampling(self, env_ids: torch.Tensor) -> None:
    """Start frames from a true uniform / failure-weighted mixture.

    mjlab's `adaptive_uniform_ratio` adds ratio / bins to *raw* failure counts, so with
    thousands of envs failing every step the uniform part vanishes and starts pile into the
    hardest bin (lift: 25-27% of starts in the final hold second), starving the clip start
    that a frame-0 evaluation always passes through. The paper's sampler is 50% uniform
    (App. D.1); `adaptive_uniform_mix` is that weight (0 keeps mjlab's sampler).
    """
    mix = self.cfg.adaptive_uniform_mix
    if mix <= 0.0:
      super()._adaptive_sampling(env_ids)
      return
    failed = self._env.termination_manager.terminated[env_ids]
    if torch.any(failed):
      bins = torch.clamp(
        (self.time_steps * self.bin_count) // max(self.motion.time_step_total, 1),
        0,
        self.bin_count - 1,
      )
      self._current_bin_failed[:] = torch.bincount(bins[env_ids][failed], minlength=self.bin_count)
    fail = self.bin_failed_count + 1e-6
    probs = mix / self.bin_count + (1.0 - mix) * fail / fail.sum()
    sampled = torch.multinomial(probs, len(env_ids), replacement=True)
    self.time_steps[env_ids] = (
      (sampled + sample_uniform(0.0, 1.0, (len(env_ids),), device=self.device))
      / self.bin_count
      * (self.motion.time_step_total - 1)
    ).long()
    entropy = -(probs * (probs + 1e-12).log()).sum()
    pmax, imax = probs.max(dim=0)
    self.metrics["sampling_entropy"][:] = entropy / math.log(self.bin_count) if self.bin_count > 1 else 1.0
    self.metrics["sampling_top1_prob"][:] = pmax
    self.metrics["sampling_top1_bin"][:] = imax.float() / self.bin_count

  def _update_command(self) -> None:
    # In place: other modules may hold a reference to this buffer.
    self.time_steps.add_(1).clamp_(max=self.num_frames - 1)
    self.update_relative_body_poses()
    if self.cfg.sampling_mode == "adaptive":
      self.bin_failed_count = (
        self.cfg.adaptive_alpha * self._current_bin_failed
        + (1 - self.cfg.adaptive_alpha) * self.bin_failed_count
      )
      self._current_bin_failed.zero_()


@dataclass(kw_only=True)
class SMMotionCommandCfg(MotionCommandCfg):
  box_entity_name: str | None = None
  adaptive_uniform_mix: float = 0.0  # weight of the uniform part of the start sampler

  def build(self, env: ManagerBasedRlEnv) -> SMMotionCommand:
    return SMMotionCommand(self, env)
