"""Analytic, differentiable tracking reward and termination errors (design §4.2, brief §0).

Reproduces the SMTrackEnv reward term by term: the mjlab tracking terms
(`mjlab/tasks/tracking/mdp/rewards.py`), `action_rate_l2` / `joint_pos_limits`
(`mjlab/envs/mdp/rewards.py`) and the lift box terms, each times weight x dt.

Two mjlab details reproduced on purpose:
- Body pos/ori terms compare against the reference re-anchored to the robot anchor's xy and yaw
  (`MotionCommand.update_relative_body_poses`). mjlab refreshes that re-anchoring at the END of
  the previous step, so the reward of s_{t+1} is re-anchored on the anchor of s_t. Pass
  `kin_prev` (FK of s_t) to get that; without it we re-anchor on s_{t+1} (<~1e-4 per step off).
- Sums over xyz, means over bodies, squared `quat_error_magnitude` exactly as mjlab.

Stock mjlab scores rewards on body kinematics one physics substep stale (`manager_based_rl_env.py`
step docstring), which no function of s_{t+1} can reproduce: measured on SMTrack walk, up to
~6e-3 per step (body ang/lin vel terms) and ~1/3 of terminations flipped. SMTrackEnv therefore
refreshes kinematics before terminations/rewards (`fresh_reward_kinematics=True`); with it the
analytic reward matches the env to ~1e-8 and terminations exactly (tests/supermimic/test_reward.py).

Quaternions are wxyz; the state s is [qpos 36, qvel 35, (box qpos 7, box qvel 6)], env-local.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal, Sequence

import torch
from torch import nn

from mjlab_g1.rl.supermimic.geometry import (
  quat_apply,
  quat_apply_inverse,
  quat_conj,
  quat_mul,
  yaw_quat,
)
from mjlab_g1.rl.supermimic.kinematics import (
  G1_ANCHOR_BODY_NAME,
  G1_BODY_NAMES,
  G1_SOFT_LIMIT_FACTOR,
  G1_TRACKED_BODY_NAMES,
  g1_xml_path,
)

NQ, NV, NJ = 36, 35, 29
JOINT_POS = slice(7, NQ)
BOX_POS = slice(NQ + NV, NQ + NV + 3)
BOX_QUAT = slice(NQ + NV + 3, NQ + NV + 7)


def quat_inv(q: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
  """mjlab quat_inv (conjugate / |q|^2), used by the re-anchoring yaw."""
  return quat_conj(q) / q.pow(2).sum(-1, keepdim=True).clamp(min=eps)


def quat_error_magnitude(q1: torch.Tensor, q2: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
  """|log(q1 q2^-1)| exactly as mjlab computes it (incl. its |q| factor for non-unit inputs).

  mjlab's torch.where evaluates sin(a/2)/a at a=0 as well, which makes the gradient NaN when
  robot and reference coincide; the untaken branch gets a finite stand-in here.
  """
  qd = quat_mul(q1, quat_conj(q2))
  qd = torch.where(qd[..., :1] < 0, -qd, qd)
  mag = torch.linalg.vector_norm(qd[..., 1:], dim=-1)
  angle = 2.0 * torch.atan2(mag, qd[..., 0])
  small = angle.abs() <= eps
  safe = torch.where(small, torch.ones_like(angle), angle)
  ratio = torch.where(small, 0.5 - angle * angle / 48, torch.sin(safe / 2) / safe)
  return mag / ratio


# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class RewardTerm:
  name: str  # env reward term name (keys of the returned terms dict)
  kind: str  # one of KINDS
  weight: float
  std: float = 1.0


KINDS = (
  "anchor_pos", "anchor_ori", "body_pos", "body_ori", "body_lin_vel", "body_ang_vel",
  "action_rate", "joint_limit", "box_pos", "box_ori",
)  # fmt: skip

# Paper Table A.2 = mjlab tracking minus self_collisions, joint_limit -10 -> -2 (brief §0).
TRACKING_TERMS: tuple[RewardTerm, ...] = (
  RewardTerm("motion_global_root_pos", "anchor_pos", 0.5, 0.3),
  RewardTerm("motion_global_root_ori", "anchor_ori", 0.5, 0.4),
  RewardTerm("motion_body_pos", "body_pos", 1.0, 0.3),
  RewardTerm("motion_body_ori", "body_ori", 1.0, 0.4),
  RewardTerm("motion_body_lin_vel", "body_lin_vel", 1.0, 1.0),
  RewardTerm("motion_body_ang_vel", "body_ang_vel", 1.0, 3.14),
  RewardTerm("action_rate_l2", "action_rate", -0.1),
  RewardTerm("joint_limit", "joint_limit", -2.0),
)
# Lift box terms (ours, brief §0).
BOX_TERMS: tuple[RewardTerm, ...] = (
  RewardTerm("box_pos", "box_pos", 1.0, 0.1),
  RewardTerm("box_ori", "box_ori", 0.5, 0.4),
)

# env reward function name -> analytic kind (for TrackingRewardCfg.from_env).
_FUNC_KINDS = {
  "motion_global_anchor_position_error_exp": "anchor_pos",
  "motion_global_anchor_orientation_error_exp": "anchor_ori",
  "motion_relative_body_position_error_exp": "body_pos",
  "motion_relative_body_orientation_error_exp": "body_ori",
  "motion_global_body_linear_velocity_error_exp": "body_lin_vel",
  "motion_global_body_angular_velocity_error_exp": "body_ang_vel",
  "action_rate_l2": "action_rate",
  "joint_pos_limits": "joint_limit",
}
_DEFAULT_STD = {t.kind: t.std for t in TRACKING_TERMS + BOX_TERMS}


@dataclass
class TrackingRewardCfg:
  terms: tuple[RewardTerm, ...] = TRACKING_TERMS
  dt: float = 0.02  # mjlab scales every term by step_dt
  has_box: bool = False
  body_names: tuple[str, ...] = G1_TRACKED_BODY_NAMES  # bodies the body_* terms average over
  anchor_body: str = G1_ANCHOR_BODY_NAME  # mjlab G1 anchor (paper text says pelvis: brief §0)
  lib_body_names: tuple[str, ...] = G1_BODY_NAMES  # body axis of MotionLibrary arrays
  kin_body_names: tuple[str, ...] | None = None  # body axis of BodyKin; None = lib_body_names
  soft_joint_pos_limits: Any = None  # [29, 2]; None = g1.xml ranges x soft_limit_factor
  soft_limit_factor: float = G1_SOFT_LIMIT_FACTOR
  reanchor: Literal["prev", "next"] = "prev"  # "prev" = mjlab (see module docstring)
  term_thresholds: tuple[float, ...] = (0.25, 0.8, 0.3)  # anchor z, tilt, (lift) box pos

  @classmethod
  def walk(cls, **kw: Any) -> "TrackingRewardCfg":
    return cls(**kw)

  @classmethod
  def lift(cls, **kw: Any) -> "TrackingRewardCfg":
    kw.setdefault("terms", TRACKING_TERMS + BOX_TERMS)
    return cls(has_box=True, **kw)

  @classmethod
  def from_env(
    cls, env: Any, kin_body_names: Sequence[str] | None = None, **kw: Any
  ) -> "TrackingRewardCfg":
    """Read term weights/stds, bodies, soft limits and thresholds from a built mjlab env.

    Raises if the env has a nonzero-weight reward term with no analytic counterpart, since the
    parity check could never pass then.
    """
    env = getattr(env, "unwrapped", env)
    rm = env.reward_manager
    terms: list[RewardTerm] = []
    command_name = None
    for name in rm.active_terms:
      tc = rm.get_term_cfg(name)
      fn = getattr(tc.func, "__name__", type(tc.func).__name__)
      kind = _FUNC_KINDS.get(fn)
      if kind is None:
        kind = next((k for k in ("box_pos", "box_ori") if k in name or k in fn), None)
      if kind is None:
        if tc.weight != 0.0:
          raise ValueError(f"reward term {name!r} ({fn}) has no analytic counterpart")
        continue
      params = dict(tc.params or {})
      command_name = command_name or params.get("command_name")
      if params.get("body_names") is not None:
        raise ValueError(f"reward term {name!r} uses a body subset; not supported")
      std = float(params.get("std", _DEFAULT_STD[kind]))
      terms.append(RewardTerm(name, kind, float(tc.weight), std))

    cmd = env.command_manager.get_term(command_name or "motion")
    robot = env.scene["robot"]
    limits = robot.data.soft_joint_pos_limits[0].detach().clone()
    dt = float(env.step_dt) if getattr(env.cfg, "scale_rewards_by_dt", True) else 1.0
    # A criterion the env does not have never fires (inf); an env failure term we cannot
    # reproduce raises, since the analytic `terminated` labels/parity would silently disagree.
    thresholds = [math.inf, math.inf, math.inf]
    tm = env.termination_manager
    for name in tm.active_terms:
      tc = tm.get_term_cfg(name)
      if getattr(tc, "time_out", False):
        continue  # truncations (time limit, clip end) are not failures
      fn = getattr(tc.func, "__name__", type(tc.func).__name__)
      th = (tc.params or {}).get("threshold")
      if fn == "bad_anchor_pos_z_only":
        slot = 0
      elif fn == "bad_anchor_ori":
        slot = 1
      elif "box" in name or "box" in fn:
        slot = 2
      else:
        raise ValueError(f"termination term {name!r} ({fn}) has no analytic counterpart")
      if th is None:
        raise ValueError(f"termination term {name!r} ({fn}) has no 'threshold' param")
      thresholds[slot] = float(th)
    has_box = bool(getattr(env, "has_box", any(t.kind.startswith("box") for t in terms)))
    out = cls(
      terms=tuple(terms),
      dt=dt,
      has_box=has_box,
      body_names=tuple(cmd.cfg.body_names),
      anchor_body=cmd.cfg.anchor_body_name,
      lib_body_names=tuple(robot.body_names),
      kin_body_names=tuple(kin_body_names) if kin_body_names is not None else None,
      soft_joint_pos_limits=limits,
      term_thresholds=tuple(thresholds),
    )
    return replace(out, **kw) if kw else out


@lru_cache(maxsize=4)
def _xml_soft_limits(xml: str, factor: float) -> torch.Tensor:
  import mujoco

  model = mujoco.MjModel.from_xml_path(xml)
  ids = [j for j in range(model.njnt) if model.jnt_type[j] != mujoco.mjtJoint.mjJNT_FREE]
  rng = torch.tensor(model.jnt_range[ids], dtype=torch.float32)  # float32 like mjlab/warp
  mean, half = (rng[:, 0] + rng[:, 1]) / 2, 0.5 * (rng[:, 1] - rng[:, 0]) * factor
  return torch.stack((mean - half, mean + half), dim=-1)


def soft_joint_limits_from_xml(
  xml: str | Path | None = None, factor: float = G1_SOFT_LIMIT_FACTOR
) -> torch.Tensor:
  """[29, 2] soft limits as mjlab's Entity computes them (mid +- factor x half range).

  TorchFK.soft_joint_limits holds the same numbers; pass either as soft_joint_pos_limits.
  """
  return _xml_soft_limits(str(xml or g1_xml_path()), float(factor)).clone()


def _get(obj: Any, name: str) -> torch.Tensor:
  return obj[name] if isinstance(obj, dict) else getattr(obj, name)


class TrackingReward(nn.Module):
  """r_{t+1} = sum_i w_i * dt * term_i(s_{t+1}, ref[k_t], a_t, a_{t-1}) (all torch, differentiable)."""

  def __init__(self, cfg: TrackingRewardCfg, device: str | torch.device | None = None):
    super().__init__()
    self.cfg = cfg
    self.terms = tuple(cfg.terms)
    unknown = [t.kind for t in self.terms if t.kind not in KINDS]
    if unknown:
      raise ValueError(f"unknown reward kinds {unknown}")
    if not cfg.has_box and any(t.kind.startswith("box") and t.weight != 0 for t in self.terms):
      raise ValueError("box reward terms need has_box=True")
    lib_names = list(cfg.lib_body_names)
    kin_names = list(cfg.kin_body_names or cfg.lib_body_names)
    self.register_buffer("lib_ids", _ids(lib_names, cfg.body_names), persistent=False)
    self.register_buffer("kin_ids", _ids(kin_names, cfg.body_names), persistent=False)
    self.lib_anchor = lib_names.index(cfg.anchor_body)
    self.kin_anchor = kin_names.index(cfg.anchor_body)
    limits = cfg.soft_joint_pos_limits
    if limits is None:
      limits = soft_joint_limits_from_xml(factor=cfg.soft_limit_factor)
    limits = torch.as_tensor(limits, dtype=torch.float32)
    if limits.shape != (NJ, 2):
      raise ValueError(f"soft_joint_pos_limits must be [{NJ}, 2], got {tuple(limits.shape)}")
    self.register_buffer("soft_limits", limits, persistent=False)
    self.names = [t.name for t in self.terms]
    self._warned = False
    if device is not None:
      self.to(device)

  # ---- reward -------------------------------------------------------------------------------

  def forward(
    self,
    s_next: torch.Tensor,
    kin_next: Any,
    ref: Any,
    a: torch.Tensor,
    a_prev: torch.Tensor,
    kin_prev: Any = None,
  ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Returns (r [*], {term name: weight * dt * value [*]}); sum of the dict = r."""
    if kin_prev is None and self.cfg.reanchor == "prev" and not self._warned:
      warnings.warn(
        "TrackingReward: kin_prev not given; re-anchoring on s_{t+1} instead of s_t (mjlab).",
        stacklevel=2,
      )
      self._warned = True
    kinds = {t.kind for t in self.terms if t.weight != 0.0}
    v: dict[str, torch.Tensor] = {}

    ref_pos = _get(ref, "body_pos_w")
    ref_quat = _get(ref, "body_quat_w")
    ref_a_pos = ref_pos[..., self.lib_anchor, :]
    ref_a_quat = ref_quat[..., self.lib_anchor, :]
    rob_a_pos = kin_next.pos[..., self.kin_anchor, :]
    rob_a_quat = kin_next.quat[..., self.kin_anchor, :]

    if "anchor_pos" in kinds:
      err = (ref_a_pos - rob_a_pos).square().sum(-1)
      v["anchor_pos"] = (-err / self._std("anchor_pos") ** 2).exp()
    if "anchor_ori" in kinds:
      err = quat_error_magnitude(ref_a_quat, rob_a_quat).square()
      v["anchor_ori"] = (-err / self._std("anchor_ori") ** 2).exp()
    if kinds & {"body_pos", "body_ori"}:
      src = kin_prev if (kin_prev is not None and self.cfg.reanchor == "prev") else kin_next
      rel_pos, rel_quat = self.reanchored_reference(ref, src)
      if "body_pos" in kinds:
        err = (rel_pos - kin_next.pos[..., self.kin_ids, :]).square().sum(-1)
        v["body_pos"] = (-err.mean(-1) / self._std("body_pos") ** 2).exp()
      if "body_ori" in kinds:
        err = quat_error_magnitude(rel_quat, kin_next.quat[..., self.kin_ids, :]).square()
        v["body_ori"] = (-err.mean(-1) / self._std("body_ori") ** 2).exp()
    if "body_lin_vel" in kinds:
      ref_v = _get(ref, "body_lin_vel_w")[..., self.lib_ids, :]
      err = (ref_v - kin_next.lin_vel[..., self.kin_ids, :]).square().sum(-1)
      v["body_lin_vel"] = (-err.mean(-1) / self._std("body_lin_vel") ** 2).exp()
    if "body_ang_vel" in kinds:
      ref_w = _get(ref, "body_ang_vel_w")[..., self.lib_ids, :]
      err = (ref_w - kin_next.ang_vel[..., self.kin_ids, :]).square().sum(-1)
      v["body_ang_vel"] = (-err.mean(-1) / self._std("body_ang_vel") ** 2).exp()
    if "action_rate" in kinds:
      v["action_rate"] = (a - a_prev).square().sum(-1)
    if "joint_limit" in kinds:
      q = s_next[..., JOINT_POS]
      lim = self.soft_limits.to(q)
      out = -(q - lim[:, 0]).clip(max=0.0) + (q - lim[:, 1]).clip(min=0.0)
      v["joint_limit"] = out.sum(-1)
    if "box_pos" in kinds:
      err = (_get(ref, "box_pos_w") - s_next[..., BOX_POS]).square().sum(-1)
      v["box_pos"] = (-err / self._std("box_pos") ** 2).exp()
    if "box_ori" in kinds:
      err = quat_error_magnitude(_get(ref, "box_quat_w"), s_next[..., BOX_QUAT]).square()
      v["box_ori"] = (-err / self._std("box_ori") ** 2).exp()

    zero = torch.zeros(s_next.shape[:-1], dtype=s_next.dtype, device=s_next.device)
    terms: dict[str, torch.Tensor] = {}
    total = zero
    for t in self.terms:
      if t.weight == 0.0:  # mjlab skips zero-weight terms and logs 0
        terms[t.name] = zero
        continue
      # nan_to_num as mjlab's RewardManager does (corrupted physics -> 0).
      val = torch.nan_to_num(v[t.kind] * (t.weight * self.cfg.dt), nan=0.0, posinf=0.0, neginf=0.0)
      terms[t.name] = val
      total = total + val
    return total, terms

  def reanchored_reference(self, ref: Any, kin_anchor: Any) -> tuple[torch.Tensor, torch.Tensor]:
    """mjlab update_relative_body_poses: reference bodies moved to the robot anchor's xy + yaw."""
    ref_pos = _get(ref, "body_pos_w")
    ref_quat = _get(ref, "body_quat_w")
    ref_a_pos = ref_pos[..., self.lib_anchor, :]
    ref_a_quat = ref_quat[..., self.lib_anchor, :]
    rob_a_pos = kin_anchor.pos[..., self.kin_anchor, :]
    rob_a_quat = kin_anchor.quat[..., self.kin_anchor, :]
    delta_pos = torch.cat((rob_a_pos[..., :2], ref_a_pos[..., 2:3]), dim=-1)
    delta_ori = yaw_quat(quat_mul(rob_a_quat, quat_inv(ref_a_quat)))[..., None, :]
    body_pos = ref_pos[..., self.lib_ids, :]
    body_quat = ref_quat[..., self.lib_ids, :]
    rel_quat = quat_mul(delta_ori, body_quat)
    rel_pos = delta_pos[..., None, :] + quat_apply(delta_ori, body_pos - ref_a_pos[..., None, :])
    return rel_pos, rel_quat

  # ---- terminations -------------------------------------------------------------------------

  def term_errors(
    self, kin_next: Any, ref: Any, s_next: torch.Tensor | None = None
  ) -> tuple[torch.Tensor, ...]:
    """(e_pz, e_tilt[, e_box]) as mjlab bad_anchor_pos_z_only / bad_anchor_ori (+ box distance).

    e_box is returned only when cfg.has_box (needs s_next for the box position).
    """
    ref_a_pos = _get(ref, "body_pos_w")[..., self.lib_anchor, :]
    ref_a_quat = _get(ref, "body_quat_w")[..., self.lib_anchor, :]
    rob_a_pos = kin_next.pos[..., self.kin_anchor, :]
    rob_a_quat = kin_next.quat[..., self.kin_anchor, :]
    e_pz = (ref_a_pos[..., 2] - rob_a_pos[..., 2]).abs()
    g = torch.tensor((0.0, 0.0, -1.0), dtype=rob_a_quat.dtype, device=rob_a_quat.device)
    g_ref = quat_apply_inverse(ref_a_quat, g.expand(ref_a_quat.shape[:-1] + (3,)))[..., 2]
    g_rob = quat_apply_inverse(rob_a_quat, g.expand(rob_a_quat.shape[:-1] + (3,)))[..., 2]
    e_tilt = (g_ref - g_rob).abs()
    if not self.cfg.has_box:
      return e_pz, e_tilt
    if s_next is None:
      raise ValueError("term_errors needs s_next for the box error (has_box=True)")
    e_box = torch.linalg.vector_norm(s_next[..., BOX_POS] - _get(ref, "box_pos_w"), dim=-1)
    return e_pz, e_tilt, e_box

  def terminated(self, kin_next: Any, ref: Any, s_next: torch.Tensor | None = None) -> torch.Tensor:
    """Hard analytic termination (for parity checks and labels), bool [*]."""
    errs = self.term_errors(kin_next, ref, s_next)
    out = torch.zeros_like(errs[0], dtype=torch.bool)
    for e, th in zip(errs, self.cfg.term_thresholds, strict=False):
      out = out | (e > th)
    return out

  @property
  def max_step_reward(self) -> float:
    """Sum of positive weights x dt: the per-step maximum (normalized-return denominator)."""
    return sum(max(t.weight, 0.0) for t in self.terms) * self.cfg.dt

  def _std(self, kind: str) -> float:
    return next(t.std for t in self.terms if t.kind == kind)


def _ids(axis: list[str], names: Sequence[str]) -> torch.Tensor:
  missing = [n for n in names if n not in axis]
  if missing:
    raise ValueError(f"bodies {missing} not on the body axis {axis}")
  return torch.tensor([axis.index(n) for n in names], dtype=torch.long)
