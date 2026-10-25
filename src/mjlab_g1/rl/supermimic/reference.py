"""Reference motion library shared by the env, the PPO baseline and the SuperMimic learner.

Reads an mjlab MotionCommand npz (brief §3): joint_pos/joint_vel [F,29] and body_{pos,quat,
lin_vel,ang_vel}_w [F,B,·] for ALL robot bodies in mjlab entity order (B=30 for mjlab's G1; the
env's MotionCommand slices its 14 tracked bodies out of the same arrays). Positions are env-local
(mjlab adds env_origins at runtime). The lift clip adds box_{pos,quat,lin_vel,ang_vel}_w and
grasp_frame (written by scripts/make_box_reference.py).

One clip per task, so clip_start = 0 and clip_last = F-1. Frame indices past the end clamp to the
last frame (goal frames near the end repeat it); `alive(k)` says whether k is still on the clip.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

_ROBOT_KEYS = (
  "joint_pos",
  "joint_vel",
  "body_pos_w",
  "body_quat_w",
  "body_lin_vel_w",
  "body_ang_vel_w",
)
_BOX_KEYS = ("box_pos_w", "box_quat_w", "box_lin_vel_w", "box_ang_vel_w")


@dataclass
class RefFrame:
  """Reference quantities gathered at frame indices k of any shape [*]."""

  joint_pos: torch.Tensor  # [*, 29]
  joint_vel: torch.Tensor  # [*, 29]
  body_pos_w: torch.Tensor  # [*, B, 3]
  body_quat_w: torch.Tensor  # [*, B, 4]
  body_lin_vel_w: torch.Tensor  # [*, B, 3]
  body_ang_vel_w: torch.Tensor  # [*, B, 3]
  box_pos_w: torch.Tensor | None = None  # [*, 3]
  box_quat_w: torch.Tensor | None = None  # [*, 4]
  box_lin_vel_w: torch.Tensor | None = None  # [*, 3]
  box_ang_vel_w: torch.Tensor | None = None  # [*, 3]


class MotionLibrary:
  def __init__(self, npz_path: str | Path, device: str | torch.device = "cpu") -> None:
    self.path = Path(npz_path)
    self.device = torch.device(device)
    with np.load(self.path) as data:  # NpzFile keeps the zip open until closed
      self._load(data)
    self.clip_start = 0
    self.clip_last = self.num_frames - 1

  def _load(self, data: np.lib.npyio.NpzFile) -> None:
    missing = [k for k in _ROBOT_KEYS if k not in data.files]
    if missing:
      raise KeyError(f"{self.path}: missing keys {missing}")

    def t(key: str) -> torch.Tensor:
      return torch.as_tensor(np.asarray(data[key], dtype=np.float32), device=self.device)

    self.joint_pos = t("joint_pos")
    self.joint_vel = t("joint_vel")
    self.body_pos_w = t("body_pos_w")
    self.body_quat_w = t("body_quat_w")
    self.body_lin_vel_w = t("body_lin_vel_w")
    self.body_ang_vel_w = t("body_ang_vel_w")
    self.num_frames = int(self.joint_pos.shape[0])
    self.num_bodies = int(self.body_pos_w.shape[1])
    for key in _ROBOT_KEYS:
      if getattr(self, key).shape[0] != self.num_frames:
        raise ValueError(f"{self.path}: {key} has {getattr(self, key).shape[0]} frames")

    present = [k for k in _BOX_KEYS if k in data.files]
    if present and len(present) != len(_BOX_KEYS):
      raise KeyError(f"{self.path}: partial box reference, has {present}")
    self.has_box = bool(present)
    self.box_pos_w = t("box_pos_w") if self.has_box else None
    self.box_quat_w = t("box_quat_w") if self.has_box else None
    self.box_lin_vel_w = t("box_lin_vel_w") if self.has_box else None
    self.box_ang_vel_w = t("box_ang_vel_w") if self.has_box else None
    for key in _BOX_KEYS if self.has_box else ():
      if getattr(self, key).shape[0] != self.num_frames:
        raise ValueError(f"{self.path}: {key} has {getattr(self, key).shape[0]} frames")
    self.grasp_frame: int | None = (
      int(np.asarray(data["grasp_frame"]).reshape(-1)[0]) if "grasp_frame" in data.files else None
    )
    # Informational only: mjlab advances one frame per env step and never reads fps.
    self.fps: int | None = (
      int(np.asarray(data["fps"]).reshape(-1)[0]) if "fps" in data.files else None
    )

  # ----- indexing -----

  def _k(self, k: torch.Tensor | int) -> torch.Tensor:
    return torch.as_tensor(k, dtype=torch.long, device=self.device)

  def clamp(self, k: torch.Tensor | int) -> torch.Tensor:
    return self._k(k).clamp(0, self.clip_last)

  def frame(self, k: torch.Tensor | int, body_ids: torch.Tensor | None = None) -> RefFrame:
    """Gather frame(s) k (any shape), clamped to [0, F-1]; `body_ids` optionally subsets bodies."""
    kc = self.clamp(k)

    def bodies(x: torch.Tensor) -> torch.Tensor:
      x = x[kc]
      return x if body_ids is None else x.index_select(-2, body_ids.to(self.device))

    box = self.has_box
    return RefFrame(
      joint_pos=self.joint_pos[kc],
      joint_vel=self.joint_vel[kc],
      body_pos_w=bodies(self.body_pos_w),
      body_quat_w=bodies(self.body_quat_w),
      body_lin_vel_w=bodies(self.body_lin_vel_w),
      body_ang_vel_w=bodies(self.body_ang_vel_w),
      box_pos_w=self.box_pos_w[kc] if box else None,
      box_quat_w=self.box_quat_w[kc] if box else None,
      box_lin_vel_w=self.box_lin_vel_w[kc] if box else None,
      box_ang_vel_w=self.box_ang_vel_w[kc] if box else None,
    )

  def alive(self, k: torch.Tensor | int) -> torch.Tensor:
    """True while k is still on the clip (k <= F-1)."""
    return self._k(k) <= self.clip_last

  def t_left(self, k: torch.Tensor | int, horizon: int = 64) -> torch.Tensor:
    """Remaining clip steps, clamped to [0, horizon] and scaled to [0, 1] (critic input)."""
    return (self.clip_last - self._k(k)).clamp(0, horizon).float() / float(horizon)

  def to(self, device: str | torch.device) -> "MotionLibrary":
    """Move all tensors in place (returns self, like nn.Module.to)."""
    self.device = torch.device(device)
    for key in _ROBOT_KEYS + _BOX_KEYS:
      x = getattr(self, key)
      if x is not None:
        setattr(self, key, x.to(self.device))
    return self

  @property
  def duration_s(self) -> float:
    return self.num_frames * 0.02  # one frame per 0.02 s env step

  def __repr__(self) -> str:
    box = f", box, grasp_frame={self.grasp_frame}" if self.has_box else ""
    return f"MotionLibrary({self.path.name}, F={self.num_frames}, B={self.num_bodies}{box})"
