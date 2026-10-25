"""SuperMimic world model (design_learner.md §3; paper §3.2, Eq. 1).

f_ω predicts a residual on generalized coordinates and a semi-implicit Euler step integrates it, so
the zero-initialized network starts out as a constant-velocity model (SuperTrack §3.4, §6.3). The
feature map φ(s) and the FK are injected (features.py / kinematics.py) so the dynamics core can be
unit-tested on a toy system with the same integrator the robot uses.

Conventions (brief §1): quaternions wxyz; s = [qpos, qvel(, box qpos, box qvel)] with root and box
angular velocity in the WORLD frame; the network works in body-frame ω_b = Rᵀω.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from mjlab_g1.rl.supermimic import geometry as geo

if TYPE_CHECKING:
  from mjlab_g1.rl.supermimic.replay import Window

KinFn = Callable[[torch.Tensor], Any]
"""s[*,S] -> kinematics object (e.g. features.BodyKin); opaque to the world model."""
PhiFn = Callable[[torch.Tensor, Any], torch.Tensor]
"""(s[*,S], kin | None) -> φ(s)[*,P]."""
TermExtraFn = Callable[[torch.Tensor, Any, Any], torch.Tensor]
"""(s_next[*,S], kin_next, ref) -> extra termination features [*,E] (e_pz, e_tilt, ref anchor z,
(lift) ‖e_box‖). `ref` is whatever the caller scores against, e.g. `lib.frame(k + offset)`."""
RefFn = Callable[[torch.Tensor], Any]
"""Replay ref_idx [*] -> the `ref` object handed to TermExtraFn (used by window_loss)."""

# Squared-angle threshold below which exp/log use Taylor branches. Zero-init output plus a robot at
# rest gives exactly zero angles, where sin(θ)/θ has a NaN gradient that torch.where cannot mask.
_SMALL_ANGLE_SQ = 1e-8


# --------------------------------------------------------------------------------------------------
# Quaternion helpers (wxyz). Local copies so this module has no dependency on features.py.


def quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
  aw, ax, ay, az = a.unbind(-1)
  bw, bx, by, bz = b.unbind(-1)
  return torch.stack(
    (
      aw * bw - ax * bx - ay * by - az * bz,
      aw * bx + ax * bw + ay * bz - az * by,
      aw * by - ax * bz + ay * bw + az * bx,
      aw * bz + ax * by - ay * bx + az * bw,
    ),
    dim=-1,
  )


def quat_conj(q: torch.Tensor) -> torch.Tensor:
  return torch.cat((q[..., :1], -q[..., 1:]), dim=-1)


def quat_rotate(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
  """R(q) v."""
  w, u = q[..., :1], q[..., 1:]
  t = 2.0 * geo.cross(u, v)
  return v + w * t + geo.cross(u, t)


def quat_rotate_inv(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
  """R(q)ᵀ v."""
  return quat_rotate(quat_conj(q), v)


def quat_exp(v: torch.Tensor) -> torch.Tensor:
  """Rotation vector (axis·angle) -> unit quaternion, gradient-safe at zero."""
  th2 = v.square().sum(-1, keepdim=True)
  small = th2 < _SMALL_ANGLE_SQ
  th = torch.sqrt(torch.where(small, torch.ones_like(th2), th2))
  w = torch.where(small, 1.0 - th2 / 8.0, torch.cos(0.5 * th))
  k = torch.where(small, 0.5 - th2 / 48.0, torch.sin(0.5 * th) / th)
  return torch.cat((w, k * v), dim=-1)


def quat_log(q: torch.Tensor) -> torch.Tensor:
  """Unit quaternion -> rotation vector with angle in [0, π] (q and -q map to the same)."""
  q = torch.where(q[..., :1] < 0.0, -q, q)
  w, u = q[..., :1], q[..., 1:]
  s2 = u.square().sum(-1, keepdim=True)
  small = s2 < _SMALL_ANGLE_SQ
  s = torch.sqrt(torch.where(small, torch.ones_like(s2), s2))
  # Both branches must stay finite everywhere, or the masked one poisons the backward pass.
  w_s = torch.where(small, w.clamp_min(1e-6), torch.ones_like(w))
  f = torch.where(small, 2.0 / w_s * (1.0 - s2 / (3.0 * w_s.square())), 2.0 * torch.atan2(s, w) / s)
  return f * u


def heading_cs(q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
  """cos/sin [*,1] of the yaw of R(q) (projection of the body x axis onto the ground plane)."""
  w, x, y, z = q.unbind(-1)
  hx = 1.0 - 2.0 * (y * y + z * z)
  hy = 2.0 * (x * y + w * z)
  n = torch.sqrt((hx * hx + hy * hy).clamp_min(1e-12))
  return (hx / n).unsqueeze(-1), (hy / n).unsqueeze(-1)


def rot_z(c: Any, s: Any, v: torch.Tensor) -> torch.Tensor:
  """R_H v for a yaw rotation given by (cos, sin)."""
  x, y = v[..., :1], v[..., 1:2]
  return torch.cat((c * x - s * y, s * x + c * y, v[..., 2:]), dim=-1)


def rot_z_inv(c: Any, s: Any, v: torch.Tensor) -> torch.Tensor:
  """R_Hᵀ v."""
  return rot_z(c, -s, v)


# --------------------------------------------------------------------------------------------------
# State layout.

Layout = list[tuple[str, int]]


@dataclass(frozen=True)
class StateSpec:
  """Layout of s, of the network residual y and of the error vector e.

  A free body (robot root, box) contributes pos 3 + quat 4 + lin vel 3 + ang vel 3 to s and
  Δv 3, Δω_b 3, δp 3, δθ 3 to y. Joints contribute q, q̇ to s and Δq̇, δq to y. With
  `robot_root=False` the same integrator runs on a plain joint chain (toy tests).
  """

  num_joints: int = 29
  robot_root: bool = True
  has_box: bool = False

  @classmethod
  def g1(cls, has_box: bool = False) -> "StateSpec":
    return cls(num_joints=29, robot_root=True, has_box=has_box)

  @property
  def qpos_dim(self) -> int:
    return (7 if self.robot_root else 0) + self.num_joints

  @property
  def qvel_dim(self) -> int:
    return (6 if self.robot_root else 0) + self.num_joints

  @property
  def state_dim(self) -> int:
    return self.qpos_dim + self.qvel_dim + (13 if self.has_box else 0)

  @property
  def out_dim(self) -> int:
    return 12 * (int(self.robot_root) + int(self.has_box)) + 2 * self.num_joints

  @property
  def err_dim(self) -> int:
    return self.out_dim

  def state_layout(self) -> Layout:
    nj, root, box = self.num_joints, self.robot_root, self.has_box
    return (
      ([("p", 3), ("quat", 4)] if root else [])
      + [("q", nj)]
      + ([("v", 3), ("w", 3)] if root else [])
      + [("qd", nj)]
      + ([("bp", 3), ("bquat", 4), ("bv", 3), ("bw", 3)] if box else [])
    )

  def residual_layout(self) -> Layout:
    """y = [Δν, δx]: velocity deltas first, then position residuals (design §3.1)."""
    nj, root, box = self.num_joints, self.robot_root, self.has_box
    vel = (
      ([("dv", 3), ("dw", 3)] if root else [])
      + [("dqd", nj)]
      + ([("bdv", 3), ("bdw", 3)] if box else [])
    )
    pos = (
      ([("dp", 3), ("dth", 3)] if root else [])
      + [("dq", nj)]
      + ([("bdp", 3), ("bdth", 3)] if box else [])
    )
    return vel + pos

  def error_layout(self) -> Layout:
    """e = [p, θ, q, v, ω, q̇(, box p, θ, v, ω)] (design §3.3)."""
    nj, root, box = self.num_joints, self.robot_root, self.has_box
    return (
      ([("ep", 3), ("eth", 3)] if root else [])
      + [("eq", nj)]
      + ([("ev", 3), ("ew", 3)] if root else [])
      + [("eqd", nj)]
      + ([("bep", 3), ("beth", 3), ("bev", 3), ("bew", 3)] if box else [])
    )

  def identity_state(self, like: torch.Tensor) -> torch.Tensor:
    """[S] finite placeholder state: zeros with identity quaternions (dtype/device of `like`)."""
    out = torch.zeros(self.state_dim, dtype=like.dtype, device=like.device)
    i = 0
    for name, n in self.state_layout():
      if name in ("quat", "bquat"):
        out[i] = 1.0
      i += n
    return out

  @staticmethod
  def split(x: torch.Tensor, layout: Layout) -> dict[str, torch.Tensor]:
    out, i = {}, 0
    for name, n in layout:
      out[name] = x[..., i : i + n]
      i += n
    return out

  @staticmethod
  def join(d: dict[str, torch.Tensor], layout: Layout) -> torch.Tensor:
    return torch.cat([d[name] for name, _ in layout], dim=-1)


def _free_step(p, quat, v, w, dv, dw, dp, dth, c, s, dt: float):
  """Semi-implicit Euler for one free body: velocities first, then positions with the new ones."""
  w_b1 = quat_rotate_inv(quat, w) + dw
  v1 = v + rot_z(c, s, dv)
  p1 = p + dt * v1 + rot_z(c, s, dp)
  quat1 = F.normalize(quat_mul(quat, quat_exp(dt * w_b1 + dth)), dim=-1)
  # ω_b' is expressed in the pre-step body frame, so it maps back to world with the pre-step R.
  # Written as ω + RΔω_b (= R ω_b') so a zero residual leaves world ω bit-exactly unchanged.
  w1 = w + quat_rotate(quat, dw)
  return p1, quat1, v1, w1


def _free_residual(p, quat, v, w, p1, quat1, v1, w1, c, s, dt: float):
  """Exact inverse of `_free_step`."""
  w_b1 = quat_rotate_inv(quat, w1)
  dv = rot_z_inv(c, s, v1 - v)
  dw = quat_rotate_inv(quat, w1 - w)
  dp = rot_z_inv(c, s, p1 - p - dt * v1)
  dth = quat_log(quat_mul(quat_conj(quat), quat1)) - dt * w_b1
  return dv, dw, dp, dth


def _free_error(p_hat, q_hat, v_hat, w_hat, p, quat, v, w, c, s):
  """Error of a predicted free body against a reference, in the reference's heading/body frame."""
  ep = rot_z_inv(c, s, p_hat - p)
  eth = quat_log(quat_mul(quat_conj(quat), q_hat))
  ev = rot_z_inv(c, s, v_hat - v)
  ew = quat_rotate_inv(quat, w_hat - w)
  return ep, eth, ev, ew


# --------------------------------------------------------------------------------------------------
# Running statistics.


class RunningStats(nn.Module):
  """Cumulative mean/std (Chan's parallel update, float64 accumulators).

  Unlike rsl_rl's EmpiricalNormalization, `update` does not depend on `self.training`, so an
  `.eval()` elsewhere cannot silently freeze the statistics. Updates happen only through explicit
  calls (outside autograd), which is what keeps them frozen inside a gradient update.
  """

  def __init__(self, dim: int):
    super().__init__()
    self.dim = dim
    self.register_buffer("count", torch.zeros((), dtype=torch.float64))
    self.register_buffer("_mean64", torch.zeros(dim, dtype=torch.float64))
    self.register_buffer("_m2", torch.zeros(dim, dtype=torch.float64))
    self.register_buffer("mean", torch.zeros(dim))
    self.register_buffer("std", torch.ones(dim))

  @torch.no_grad()
  def update(self, x: torch.Tensor) -> None:
    x = x.detach().reshape(-1, self.dim).to(torch.float64)
    n = x.shape[0]
    if n == 0:
      return
    batch_mean = x.mean(0)
    batch_m2 = (x - batch_mean).square().sum(0)
    total = self.count + n
    delta = batch_mean - self._mean64
    self._mean64 += delta * (n / total)
    self._m2 += batch_m2 + delta.square() * (self.count * n / total)
    self.count.copy_(total)
    self.mean.copy_(self._mean64.to(self.mean.dtype))
    self.std.copy_(torch.sqrt(self._m2 / total).to(self.std.dtype))


def _mlp(in_dim: int, hidden: Sequence[int], out_dim: int, activation: str) -> nn.Sequential:
  layers: list[nn.Module] = []
  d = in_dim
  for h in hidden:
    layers.append(nn.Linear(d, h))
    if activation == "silu_ln":
      layers += [nn.LayerNorm(h), nn.SiLU()]
    elif activation == "elu":
      layers.append(nn.ELU())
    else:
      raise ValueError(f"unknown activation {activation!r} (expected 'silu_ln' or 'elu')")
    d = h
  layers.append(nn.Linear(d, out_dim))
  return nn.Sequential(*layers)


def motion_action_scale(
  joint_pos_ref: torch.Tensor,
  default_joint_pos: torch.Tensor,
  action_scale: torch.Tensor,
  floor: float = 0.5,
) -> torch.Tensor:
  """σ_a (design §2.2): per-joint std over the motion library of (q_ref − q0)/α, floored.

  Running stats of a near-zero initial policy would be degenerate, so the action input scale is
  fixed from the reference instead.
  """
  a_ref = (joint_pos_ref - default_joint_pos) / action_scale
  return a_ref.reshape(-1, a_ref.shape[-1]).std(0, unbiased=False).clamp_min(floor)


# --------------------------------------------------------------------------------------------------
# World model.


class WorldModel(nn.Module):
  """Deterministic ξ-conditioned dynamics ŝ' = f_ω(ŝ, a, ξ) plus a learned termination head.

  Input to f_ω: [norm φ(s), a / σ_a, enc(ξ)]. Output: residual y (layout
  `StateSpec.residual_layout`) in units of σ_out, integrated by `integrate`. Termination:
  σ(MLP([norm φ(s'), extra])).

  The injected callables are stored as plain attributes (not submodules): they must already live on
  the model's device, and their parameters/buffers are not part of this module's state_dict.
  """

  def __init__(
    self,
    spec: StateSpec,
    phi_fn: PhiFn,
    phi_dim: int,
    xi_dim: int,
    action_dim: int | None = None,
    *,
    kin_fn: KinFn | None = None,
    term_extra_fn: TermExtraFn | None = None,
    term_extra_dim: int = 0,
    ref_fn: RefFn | None = None,
    action_scale: torch.Tensor | None = None,
    xi_lo: torch.Tensor | None = None,
    xi_hi: torch.Tensor | None = None,
    hidden_dims: Sequence[int] = (2048, 2048, 2048),
    term_hidden_dims: Sequence[int] = (256, 128),
    activation: str = "silu_ln",
    dt: float = 0.02,
    phi_eps: float = 1e-2,
    out_floor: float = 1e-4,
    nom_floor: float = 1e-3,
    term_init_prob: float = 0.01,
    detach_term_input: bool = False,
  ):
    super().__init__()
    if term_extra_dim > 0 and term_extra_fn is None:
      raise ValueError("term_extra_dim > 0 requires term_extra_fn")
    self.spec = spec
    self.phi_dim = phi_dim
    self.xi_dim = xi_dim
    self.action_dim = spec.num_joints if action_dim is None else action_dim
    self.term_extra_dim = term_extra_dim
    self.dt = dt
    self.phi_eps = phi_eps
    self.out_floor = out_floor
    self.nom_floor = nom_floor
    # Eq. 1 has no detach; this flag only exists as an ablation knob.
    self.detach_term_input = detach_term_input
    # Critique M2 option: the caller may freeze φ/σ_out/σ_nom (e.g. after prefill) so σ_out stops
    # changing what the output layer's weights mean. Not saved in the state_dict.
    self.normalizers_frozen = False
    self._fns = {"phi": phi_fn, "kin": kin_fn, "term_extra": term_extra_fn, "ref": ref_fn}

    self.dynamics = _mlp(phi_dim + self.action_dim + xi_dim, hidden_dims, spec.out_dim, activation)
    last = self.dynamics[-1]
    nn.init.zeros_(last.weight)
    nn.init.zeros_(last.bias)

    self.term_head = _mlp(phi_dim + term_extra_dim, term_hidden_dims, 1, "elu")
    nn.init.constant_(self.term_head[-1].bias, math.log(term_init_prob / (1.0 - term_init_prob)))

    self.phi_norm = RunningStats(phi_dim)
    self.out_stats = RunningStats(spec.out_dim)
    self.nom_stats = RunningStats(spec.err_dim)

    def _vec(x: torch.Tensor | None, fill: float, n: int) -> torch.Tensor:
      if x is None:
        return torch.full((n,), fill)
      return torch.as_tensor(x, dtype=torch.float32).reshape(n).clone()

    self.register_buffer("action_scale", _vec(action_scale, 1.0, self.action_dim))
    # Defaults lo=-1, hi=1 make encode_xi the identity.
    self.register_buffer("xi_lo", _vec(xi_lo, -1.0, xi_dim))
    self.register_buffer("xi_hi", _vec(xi_hi, 1.0, xi_dim))

  @classmethod
  def from_cfg(
    cls, cfg: Any, spec: StateSpec, phi_fn: PhiFn, phi_dim: int, xi_dim: int, **kwargs: Any
  ) -> "WorldModel":
    """Build from a `SuperMimicWMCfg`-like object (hidden_dims, term_hidden_dims, activation)."""
    return cls(
      spec,
      phi_fn,
      phi_dim,
      xi_dim,
      hidden_dims=tuple(cfg.hidden_dims),
      term_hidden_dims=tuple(cfg.term_hidden_dims),
      activation=cfg.activation,
      **kwargs,
    )

  # ---- scales ------------------------------------------------------------------------------------

  @property
  def out_std(self) -> torch.Tensor:
    """σ_out [out_dim]: residual output scale (cumulative std of the exact residual targets)."""
    return self.out_stats.std.clamp_min(self.out_floor)

  @property
  def nom_std(self) -> torch.Tensor:
    """σ_nom [err_dim]: nominal error scale (cumulative std of the one-step change)."""
    return self.nom_stats.std.clamp_min(self.nom_floor)

  # ---- features ----------------------------------------------------------------------------------

  def kin(self, s: torch.Tensor) -> Any:
    fn = self._fns["kin"]
    return None if fn is None else fn(s)

  def phi(self, s: torch.Tensor, kin: Any = None) -> torch.Tensor:
    """Raw φ(s) [*,P]; computes the FK when `kin` is not given."""
    if kin is None:
      kin = self.kin(s)
    return self._fns["phi"](s, kin)

  def normalize_phi(self, phi: torch.Tensor) -> torch.Tensor:
    return (phi - self.phi_norm.mean) / (self.phi_norm.std + self.phi_eps)

  def encode_xi(self, xi: torch.Tensor) -> torch.Tensor:
    """enc(ξ) = 2(ξ − lo)/(hi − lo) − 1, and 0 where hi == lo (brief §1)."""
    span = self.xi_hi - self.xi_lo
    enc = 2.0 * (xi - self.xi_lo) / span.clamp_min(1e-12) - 1.0
    return torch.where(span > 0, enc, torch.zeros_like(enc))

  # ---- dynamics ----------------------------------------------------------------------------------

  def residual(
    self,
    s: torch.Tensor,
    a: torch.Tensor,
    xi_enc: torch.Tensor,
    kin: Any = None,
    phi: torch.Tensor | None = None,
  ) -> torch.Tensor:
    """Network residual y [*,out_dim] in physical units (y_net ⊙ σ_out)."""
    if phi is None:
      phi = self.phi(s, kin)
    x = torch.cat((self.normalize_phi(phi), a / self.action_scale, xi_enc), dim=-1)
    return self.dynamics(x) * self.out_std

  def step(
    self,
    s: torch.Tensor,
    a: torch.Tensor,
    xi_enc: torch.Tensor,
    kin: Any = None,
    phi: torch.Tensor | None = None,
  ) -> torch.Tensor:
    """ŝ_{t+1} = f_ω(ŝ_t, a_t, ξ), differentiable in s, a and the parameters.

    `kin` (FK of `s`) or `phi` (raw φ(s)) can be passed to avoid recomputing them.
    """
    return self.integrate(s, self.residual(s, a, xi_enc, kin, phi))

  def _heading(self, st: dict[str, torch.Tensor]) -> tuple[Any, Any]:
    if self.spec.robot_root:
      return heading_cs(st["quat"])
    return 1.0, 0.0

  def integrate(self, s: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Semi-implicit Euler (dt) with residual y (physical units): design §3.1."""
    spec, dt = self.spec, self.dt
    st = spec.split(s, spec.state_layout())
    d = spec.split(y, spec.residual_layout())
    c, sn = self._heading(st)
    out: dict[str, torch.Tensor] = {}
    if spec.robot_root:
      out["p"], out["quat"], out["v"], out["w"] = _free_step(
        st["p"], st["quat"], st["v"], st["w"], d["dv"], d["dw"], d["dp"], d["dth"], c, sn, dt
      )
    out["qd"] = st["qd"] + d["dqd"]
    out["q"] = st["q"] + dt * out["qd"] + d["dq"]
    if spec.has_box:
      # Box linear terms live in the robot heading frame (as φ), angular terms in the box frame.
      out["bp"], out["bquat"], out["bv"], out["bw"] = _free_step(
        st["bp"], st["bquat"], st["bv"], st["bw"],
        d["bdv"], d["bdw"], d["bdp"], d["bdth"], c, sn, dt,
      )
    return spec.join(out, spec.state_layout())

  def residual_targets(self, s: torch.Tensor, s_next: torch.Tensor) -> torch.Tensor:
    """The y (physical units) for which `integrate(s, y) == s_next` exactly (quats up to sign)."""
    spec, dt = self.spec, self.dt
    a = spec.split(s, spec.state_layout())
    b = spec.split(s_next, spec.state_layout())
    c, sn = self._heading(a)
    out: dict[str, torch.Tensor] = {}
    if spec.robot_root:
      out["dv"], out["dw"], out["dp"], out["dth"] = _free_residual(
        a["p"], a["quat"], a["v"], a["w"], b["p"], b["quat"], b["v"], b["w"], c, sn, dt
      )
    out["dqd"] = b["qd"] - a["qd"]
    out["dq"] = b["q"] - a["q"] - dt * b["qd"]
    if spec.has_box:
      out["bdv"], out["bdw"], out["bdp"], out["bdth"] = _free_residual(
        a["bp"], a["bquat"], a["bv"], a["bw"], b["bp"], b["bquat"], b["bv"], b["bw"], c, sn, dt
      )
    return spec.join(out, spec.residual_layout())

  def state_error(self, s_hat: torch.Tensor, s_ref: torch.Tensor) -> torch.Tensor:
    """e [*,err_dim] of s_hat against s_ref, in s_ref's heading frame (positions, linear
    velocities) and body frame (orientation, angular velocity). Not yet divided by σ_nom."""
    spec = self.spec
    h = spec.split(s_hat, spec.state_layout())
    r = spec.split(s_ref, spec.state_layout())
    c, sn = self._heading(r)
    out: dict[str, torch.Tensor] = {}
    if spec.robot_root:
      out["ep"], out["eth"], out["ev"], out["ew"] = _free_error(
        h["p"], h["quat"], h["v"], h["w"], r["p"], r["quat"], r["v"], r["w"], c, sn
      )
    out["eq"] = h["q"] - r["q"]
    out["eqd"] = h["qd"] - r["qd"]
    if spec.has_box:
      out["bep"], out["beth"], out["bev"], out["bew"] = _free_error(
        h["bp"], h["bquat"], h["bv"], h["bw"], r["bp"], r["bquat"], r["bv"], r["bw"], c, sn
      )
    return spec.join(out, spec.error_layout())

  # ---- termination head --------------------------------------------------------------------------

  def term_logit(
    self,
    s_next: torch.Tensor,
    ref: Any = None,
    kin: Any = None,
    phi: torch.Tensor | None = None,
    extra_feats: torch.Tensor | None = None,
  ) -> torch.Tensor:
    """Termination logit [*] on [norm φ(s'), extra], extra = term_extra_fn(s', kin', ref) unless
    `extra_feats` is given. Pass `kin` (FK of s') or `phi` to avoid recomputing them."""
    if kin is None and (phi is None or (extra_feats is None and self.term_extra_dim > 0)):
      kin = self.kin(s_next)
    if phi is None:
      phi = self._fns["phi"](s_next, kin)
    x = self.normalize_phi(phi)
    if self.term_extra_dim > 0:
      if extra_feats is None:
        if ref is None:
          raise ValueError("term head needs `ref` (or extra_feats) for its extra features")
        extra_feats = self._fns["term_extra"](s_next, kin, ref)
      x = torch.cat((x, extra_feats), dim=-1)
    if self.detach_term_input:
      x = x.detach()
    return self.term_head(x).squeeze(-1)

  def term_prob(
    self,
    s_next: torch.Tensor,
    ref: Any = None,
    kin: Any = None,
    phi: torch.Tensor | None = None,
    extra_feats: torch.Tensor | None = None,
  ) -> torch.Tensor:
    """d̂_{t+1} [*] in [0,1]; `ref` = the frame r_{t+1} is scored against (design §6)."""
    return torch.sigmoid(self.term_logit(s_next, ref, kin, phi, extra_feats))

  def ref_from_idx(self, ref_idx: torch.Tensor) -> Any:
    fn = self._fns["ref"]
    return ref_idx if fn is None else fn(ref_idx)

  # ---- statistics --------------------------------------------------------------------------------

  @torch.no_grad()
  def update_normalizers(
    self, s: torch.Tensor | Mapping[str, torch.Tensor], s_next: torch.Tensor | None = None
  ) -> None:
    """Update φ, σ_out and σ_nom from collected transitions (s_t, pre-reset s_{t+1}).

    Accepts (s, s_next) or one replay transition dict with keys "s" and "s_next". Call once per
    collected step, never inside a gradient update. No-op while `normalizers_frozen`.
    """
    if isinstance(s, Mapping):
      s, s_next = s["s"], s["s_next"]
    if s_next is None:
      raise ValueError("update_normalizers needs s_next")
    if self.normalizers_frozen:
      return
    S = self.spec.state_dim
    s = s.detach().reshape(-1, S).float()
    s_next = s_next.detach().reshape(-1, S).float()
    ok = torch.isfinite(s).all(-1) & torch.isfinite(s_next).all(-1)
    s, s_next = s[ok], s_next[ok]
    if s.shape[0] == 0:
      return
    self.phi_norm.update(self.phi(torch.cat((s, s_next), dim=0)))
    self.out_stats.update(self.residual_targets(s, s_next))
    self.nom_stats.update(self.state_error(s_next, s))

  # ---- Eq. 1 -------------------------------------------------------------------------------------

  def window_loss(self, window: "Window") -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """k-step autoregressive loss on logged actions (Eq. 1, design §3.3).

    L = mean_B Σ_j m_j [mean_d (e_jd / σ_nom,d)² + BCE(d̂_j, d_j)], no detach along the chain.
    Step j (1-based) predicts window.target[:, j-1] (pre-reset s_next) from action a_{j-1}; the
    termination features use ref_fn(ref_idx[:, j-1]) (the frame r_{t+1}/d_{t+1} are scored
    against, so ref_fn applies reward_ref_offset).

    Non-finite stored data (a physics blow-up) is masked from its step to the window end and
    replaced by a finite placeholder: masking alone would still put NaN into every gradient,
    because where()'s backward multiplies 0 by the NaN local derivative.

    Metrics (detached 0-d tensors): loss, loss_state, loss_bce, err_norm_step{j} = masked batch
    mean of mean_d(e²/σ_nom²) at step j (squared, same units as the loss), err_rms_step{j} = its
    square root, term_acc, term_rate, term_brier, term_p_pos / term_p_neg (mean d̂ on true
    terminations / non-terminations; 0 when there are none), mask_frac.
    """
    a, target, term, xi = window.a, window.target, window.term, window.xi
    k = a.shape[1]
    s = window.s0
    ok0 = torch.isfinite(s).all(-1) & torch.isfinite(xi).all(-1)
    ok_t = torch.isfinite(target).all(-1) & torch.isfinite(a).all(-1)
    ok = ok0[:, None] & (ok_t.to(s.dtype).cumprod(dim=1) > 0)
    mask = window.mask.to(s.dtype) * ok.to(s.dtype)
    safe = self.spec.identity_state(s)
    s = torch.where(ok0[:, None], s, safe)
    target = torch.where(ok_t[..., None], target, safe)
    a = torch.where(torch.isfinite(a), a, torch.zeros_like(a))
    xi_enc = self.encode_xi(torch.where(torch.isfinite(xi), xi, torch.zeros_like(xi)))
    nom = self.nom_std
    phi = self.phi(s)
    loss_state = s.new_zeros(())
    loss_bce = s.new_zeros(())
    metrics: dict[str, torch.Tensor] = {}
    n_correct = s.new_zeros(())
    n_valid = s.new_zeros(())
    n_term = s.new_zeros(())
    brier = s.new_zeros(())
    p_pos = s.new_zeros(())
    p_neg = s.new_zeros(())
    for j in range(k):
      s = self.step(s, a[:, j], xi_enc, phi=phi)
      kin = self.kin(s)
      phi = self._fns["phi"](s, kin)
      m = mask[:, j] > 0
      mse = (self.state_error(s, target[:, j]) / nom).square().mean(-1)
      ref = self.ref_from_idx(window.ref_idx[:, j]) if self.term_extra_dim > 0 else None
      logit = self.term_logit(s, ref, kin=kin, phi=phi)
      label = term[:, j].to(logit.dtype)
      bce = F.binary_cross_entropy_with_logits(logit, label, reduction="none")
      zero = torch.zeros_like(mse)
      # where (not multiply): the masked continuation past a reset is never part of the loss.
      loss_state = loss_state + torch.where(m, mse, zero).mean()
      loss_bce = loss_bce + torch.where(m, bce, zero).mean()
      with torch.no_grad():
        mf = m.to(mse.dtype)
        n = mf.sum()
        err = torch.where(m, mse, zero).sum() / n.clamp_min(1.0)
        metrics[f"err_norm_step{j + 1}"] = err
        metrics[f"err_rms_step{j + 1}"] = err.sqrt()
        prob = torch.sigmoid(logit)
        n_correct += (((logit > 0).to(label.dtype) == label).to(mf.dtype) * mf).sum()
        n_term += (label * mf).sum()
        n_valid += n
        brier += ((prob - label).square() * mf).sum()
        p_pos += (prob * label * mf).sum()
        p_neg += (prob * (1.0 - label) * mf).sum()
    loss = loss_state + loss_bce
    metrics["loss"] = loss.detach()
    metrics["loss_state"] = loss_state.detach()
    metrics["loss_bce"] = loss_bce.detach()
    metrics["term_acc"] = n_correct / n_valid.clamp_min(1.0)
    metrics["term_rate"] = n_term / n_valid.clamp_min(1.0)
    metrics["term_brier"] = brier / n_valid.clamp_min(1.0)
    metrics["term_p_pos"] = p_pos / n_term.clamp_min(1.0)
    metrics["term_p_neg"] = p_neg / (n_valid - n_term).clamp_min(1.0)
    metrics["mask_frac"] = n_valid / float(mask.numel())
    return loss, metrics
