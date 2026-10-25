"""Quaternion / rotation helpers in torch (wxyz), batched over leading dims, differentiable.

Conventions match mjlab (`mjlab.utils.lab_api.math`) so env-side and learner-side features agree:
quaternions are (w, x, y, z); `quat_apply(q, v)` rotates v from the q-frame into the parent frame;
`rot6d` is mjlab's `matrix[..., :2].reshape(..., 6)` (row-major over the first two columns:
[R00, R01, R10, R11, R20, R21]), the layout of mjlab's `motion_anchor_ori_b`.

The exp/log maps stay finite (values and gradients) at zero rotation: the world model integrates
orientations through `quat_exp` every imagined step and the reward differentiates `quat_log`.
"""

from __future__ import annotations

import torch

# Below this squared angle the exp/log maps switch to Taylor expansions (float32-safe).
_SMALL_SQ = 1e-8


def quat_normalize(q: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
  return q / q.norm(dim=-1, keepdim=True).clamp_min(eps)


def quat_conj(q: torch.Tensor) -> torch.Tensor:
  return torch.cat([q[..., :1], -q[..., 1:]], dim=-1)


def quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
  """Hamilton product a ⊗ b (broadcasts over leading dims)."""
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


def cross(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
  """a x b over the last dim, written out per component.

  torch.linalg.cross lowers to modular gather indices whose compiled backward trips an
  out-of-bounds device assert on Blackwell (sm_120) under torch.compile; the explicit form
  computes the same values and fuses to plain elementwise kernels.
  """
  a, b = torch.broadcast_tensors(a, b)
  ax, ay, az = a.unbind(-1)
  bx, by, bz = b.unbind(-1)
  return torch.stack((ay * bz - az * by, az * bx - ax * bz, ax * by - ay * bx), dim=-1)


_cross = cross


def quat_apply(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
  """R(q) v for unit q (same formula as mjlab)."""
  xyz = q[..., 1:]
  t = 2.0 * _cross(xyz, v)
  return v + q[..., :1] * t + _cross(xyz, t)


def quat_apply_inverse(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
  """R(q)ᵀ v for unit q."""
  xyz = q[..., 1:]
  t = 2.0 * _cross(xyz, v)
  return v - q[..., :1] * t + _cross(xyz, t)


def quat_from_axis_angle(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
  """Rotation of `angle` [...] about unit `axis` [..., 3]."""
  half = 0.5 * angle.unsqueeze(-1)
  xyz = axis * torch.sin(half)
  w = torch.cos(half) + torch.zeros_like(xyz[..., :1])
  return torch.cat([w, xyz], dim=-1)


def quat_exp(rotvec: torch.Tensor) -> torch.Tensor:
  """Exp map: rotation vector θ·n [..., 3] -> unit quaternion [..., 4]."""
  theta_sq = (rotvec * rotvec).sum(-1, keepdim=True)
  small = theta_sq < _SMALL_SQ
  # The unsafe branch gets a finite stand-in so torch.where does not leak NaN gradients.
  theta = torch.sqrt(torch.where(small, torch.ones_like(theta_sq), theta_sq))
  half = 0.5 * theta
  k = torch.where(small, 0.5 - theta_sq / 48.0, torch.sin(half) / theta)
  w = torch.where(small, 1.0 - theta_sq / 8.0, torch.cos(half))
  return quat_normalize(torch.cat([w, rotvec * k], dim=-1))


def quat_log(q: torch.Tensor) -> torch.Tensor:
  """Log map: unit quaternion -> rotation vector with angle in [0, π] (shortest arc)."""
  q = torch.where(q[..., :1] < 0.0, -q, q)
  w = q[..., :1]
  v = q[..., 1:]
  s_sq = (v * v).sum(-1, keepdim=True)
  small = s_sq < _SMALL_SQ
  s = torch.sqrt(torch.where(small, torch.ones_like(s_sq), s_sq))
  # θ/s with θ = 2 atan2(s, w); near s=0: 2/w · (1 − s²/(3w²)). The unselected Taylor branch
  # must stay finite too: with w = 0 (a 180° rotation) it overflowed and leaked 0·inf = NaN
  # into the gradient through torch.where.
  w_safe = torch.where(small, w.clamp_min(1e-6), torch.ones_like(w))
  scale = torch.where(
    small, 2.0 / w_safe * (1.0 - s_sq / (3.0 * w_safe * w_safe)), 2.0 * torch.atan2(s, w) / s
  )
  return v * scale


quat_to_axis_angle = quat_log
axis_angle_to_quat = quat_exp


def quat_error_magnitude(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
  """Angle of q1 ⊗ q2⁻¹ in radians, as mjlab `quat_error_magnitude` (smooth at 0 here)."""
  return quat_angle(quat_mul(q1, quat_conj(q2)))


def quat_angle(q: torch.Tensor) -> torch.Tensor:
  """Rotation angle in [0, π] of a unit quaternion; differentiable at 0 (gradient 0)."""
  q = torch.where(q[..., :1] < 0.0, -q, q)
  s_sq = (q[..., 1:] ** 2).sum(-1)
  small = s_sq < _SMALL_SQ
  s = torch.sqrt(torch.where(small, torch.ones_like(s_sq), s_sq))
  w = q[..., 0]
  near_zero = 2.0 * torch.sqrt(s_sq + 1e-30) / w.clamp_min(1e-6)
  return torch.where(small, near_zero, 2.0 * torch.atan2(s, w))


def quat_angle_sq(q: torch.Tensor) -> torch.Tensor:
  """Squared rotation angle (smooth everywhere, unlike angle² through a sqrt)."""
  r = quat_log(q)
  return (r * r).sum(-1)


def yaw_angle(q: torch.Tensor) -> torch.Tensor:
  """Heading angle, mjlab `yaw_quat` formula: atan2(R10, R00).

  Forward value is exactly mjlab's everywhere. At gimbal lock (body x axis vertical, both
  atan2 args → 0) the heading is undefined and atan2's gradient ~1/r blows up (NaN at r = 0);
  there the gradient is zeroed so one fallen imagined robot cannot NaN or swamp a clipped step.
  """
  w, x, y, z = q.unbind(-1)
  sy = 2.0 * (w * z + x * y)
  cy = 1.0 - 2.0 * (y * y + z * z)
  singular = sy * sy + cy * cy < _SMALL_SQ
  safe = torch.atan2(torch.where(singular, 0.0, sy), torch.where(singular, 1.0, cy))
  return torch.where(singular, torch.atan2(sy, cy).detach(), safe)


def yaw_quat(q: torch.Tensor) -> torch.Tensor:
  """Yaw-only part of q (heading rotation about world z), mjlab `yaw_quat` formula."""
  half = 0.5 * yaw_angle(q)
  zero = torch.zeros_like(half)
  return torch.stack((torch.cos(half), zero, zero, torch.sin(half)), dim=-1)


def quat_from_yaw(yaw: torch.Tensor) -> torch.Tensor:
  half = 0.5 * yaw
  zero = torch.zeros_like(half)
  return torch.stack((torch.cos(half), zero, zero, torch.sin(half)), dim=-1)


def quat_to_matrix(q: torch.Tensor) -> torch.Tensor:
  """Rotation matrix [..., 3, 3] (same formula as mjlab `matrix_from_quat`)."""
  r, i, j, k = q.unbind(-1)
  two_s = 2.0 / (q * q).sum(-1)
  o = torch.stack(
    (
      1 - two_s * (j * j + k * k),
      two_s * (i * j - k * r),
      two_s * (i * k + j * r),
      two_s * (i * j + k * r),
      1 - two_s * (i * i + k * k),
      two_s * (j * k - i * r),
      two_s * (i * k - j * r),
      two_s * (j * k + i * r),
      1 - two_s * (i * i + j * j),
    ),
    dim=-1,
  )
  return o.reshape(q.shape[:-1] + (3, 3))


def matrix_to_rot6d(m: torch.Tensor) -> torch.Tensor:
  """First two columns, mjlab order: m[..., :, :2] flattened row-major -> [..., 6]."""
  return m[..., :, :2].reshape(m.shape[:-2] + (6,))


def quat_to_rot6d(q: torch.Tensor) -> torch.Tensor:
  return matrix_to_rot6d(quat_to_matrix(q))


rot6d = quat_to_rot6d


def quat_rel(q_from: torch.Tensor, q_to: torch.Tensor) -> torch.Tensor:
  """q_fromᵀ q_to: orientation of q_to expressed in the q_from frame (unit quats)."""
  return quat_mul(quat_conj(q_from), q_to)


def heading_frame(q: torch.Tensor) -> torch.Tensor:
  """Alias kept for readability at call sites: the yaw-only (heading) frame of q."""
  return yaw_quat(q)
