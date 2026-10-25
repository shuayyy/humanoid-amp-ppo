"""State packing and the features both the env obs terms and the SuperMimic learner compute.

Everything is plain differentiable torch on raw state tensors (no env access), so imagined
rollouts and the simulator produce identical observations, goals and world-model inputs.

State (brief §1): s = [qpos 36, qvel 35 (, box_qpos 7, box_qvel 6)], 71 walk / 84 lift.
qpos root position is env-local; qvel root (and box) angular velocity is in the WORLD frame.
`kin` arguments are `TorchFK.g1()(qpos, qvel)` outputs: all 30 bodies in mjlab order plus the
sites (left_palm, right_palm, imu_in_pelvis).

Layouts (dims are asserted in tests/supermimic/test_features.py):
- wm_features   169 / 191  design §2.2, heading-invariant
- obs_frame      93 / 102  brief §2 actor_hist terms, one frame, clean (noise added separately)
- goal          640 / 730  K=10 frames, frame-major [frame0 | frame1 | ...]
- history       930 / 1020 see PolicyObsHistory (mjlab per-term layout by default)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, NamedTuple, Sequence

import torch

from mjlab_g1.rl.supermimic.geometry import (
  quat_apply_inverse,
  quat_conj,
  quat_mul,
  quat_normalize,
  quat_to_rot6d,
  yaw_quat,
)
from mjlab_g1.rl.supermimic.kinematics import (
  ANCHOR_BODY_ID,
  FOOT_BODY_IDS,
  G1_DEFAULT_JOINT_POS,
  G1_SITE_NAMES,
  LEFT_PALM_SITE_ID,
  NUM_BODIES,
  NUM_JOINTS,
  RIGHT_PALM_SITE_ID,
  TRACKED_BODY_IDS,
  BodyKin,
  TorchFK,
)

if TYPE_CHECKING:
  from mjlab_g1.rl.supermimic.reference import MotionLibrary

##
# Dimensions and layouts.
##

QPOS_DIM, QVEL_DIM = 36, 35
BOX_QPOS_DIM, BOX_QVEL_DIM = 7, 6
ROBOT_STATE_DIM = QPOS_DIM + QVEL_DIM  # 71
STATE_DIM = {False: ROBOT_STATE_DIM, True: ROBOT_STATE_DIM + BOX_QPOS_DIM + BOX_QVEL_DIM}  # 71 / 84

# xi (raw DR context, brief §1): [foot_friction 1, encoder_bias 29, torso_com_offset 3]
# (+ lift [box_mass_scale 1, box_friction_scale 1]).
XI_FOOT_FRICTION = slice(0, 1)
XI_ENCODER_BIAS = slice(1, 1 + NUM_JOINTS)
XI_TORSO_COM = slice(1 + NUM_JOINTS, 4 + NUM_JOINTS)
XI_BOX_MASS = slice(4 + NUM_JOINTS, 5 + NUM_JOINTS)
XI_BOX_FRICTION = slice(5 + NUM_JOINTS, 6 + NUM_JOINTS)
XI_DIM = {False: 33, True: 35}

# actor_hist terms per frame, in env obs-group order (brief §2).
OBS_TERMS_WALK: tuple[tuple[str, int], ...] = (
  ("joint_pos_rel", NUM_JOINTS),
  ("joint_vel_rel", NUM_JOINTS),
  ("base_ang_vel", 3),
  ("projected_gravity", 3),
  ("last_action", NUM_JOINTS),
)
OBS_TERMS_LIFT: tuple[tuple[str, int], ...] = OBS_TERMS_WALK + (
  ("box_pos_b", 3),
  ("box_rot6d_b", 6),
)
# Uniform sensor noise half-widths η (Table A.1; box terms ours). last_action is noise-free.
OBS_NOISE: dict[str, float] = {
  "joint_pos_rel": 0.01,
  "joint_vel_rel": 0.5,
  "base_ang_vel": 0.2,
  "projected_gravity": 0.05,
  "last_action": 0.0,
  "box_pos_b": 0.01,
  "box_rot6d_b": 0.02,
}
HISTORY_LEN = 10
GOAL_K = 10
GOAL_FRAME_DIM = {False: 2 * NUM_JOINTS + 6, True: 2 * NUM_JOINTS + 6 + 9}  # 64 / 73
WM_FEATURE_DIM = {False: 169, True: 191}


def obs_terms(has_box: bool) -> tuple[tuple[str, int], ...]:
  return OBS_TERMS_LIFT if has_box else OBS_TERMS_WALK


def obs_term_dims(has_box: bool) -> tuple[int, ...]:
  return tuple(d for _, d in obs_terms(has_box))


def obs_frame_dim(has_box: bool) -> int:
  return sum(obs_term_dims(has_box))  # 93 / 102


def goal_dim(has_box: bool, K: int = GOAL_K) -> int:
  return K * GOAL_FRAME_DIM[has_box]  # 640 / 730


##
# State packing.
##


class State(NamedTuple):
  """Views into a packed state; box fields are None for the walk task."""

  qpos: torch.Tensor
  qvel: torch.Tensor
  box_qpos: torch.Tensor | None
  box_qvel: torch.Tensor | None

  @property
  def root_pos(self) -> torch.Tensor:
    return self.qpos[..., 0:3]

  @property
  def root_quat(self) -> torch.Tensor:
    return self.qpos[..., 3:7]

  @property
  def joint_pos(self) -> torch.Tensor:
    return self.qpos[..., 7:]

  @property
  def root_lin_vel(self) -> torch.Tensor:
    return self.qvel[..., 0:3]

  @property
  def root_ang_vel(self) -> torch.Tensor:
    return self.qvel[..., 3:6]

  @property
  def joint_vel(self) -> torch.Tensor:
    return self.qvel[..., 6:]

  @property
  def box_pos(self) -> torch.Tensor:
    assert self.box_qpos is not None
    return self.box_qpos[..., 0:3]

  @property
  def box_quat(self) -> torch.Tensor:
    assert self.box_qpos is not None
    return self.box_qpos[..., 3:7]

  @property
  def box_lin_vel(self) -> torch.Tensor:
    assert self.box_qvel is not None
    return self.box_qvel[..., 0:3]

  @property
  def box_ang_vel(self) -> torch.Tensor:
    assert self.box_qvel is not None
    return self.box_qvel[..., 3:6]


def pack_state(
  qpos: torch.Tensor,
  qvel: torch.Tensor,
  box_qpos: torch.Tensor | None = None,
  box_qvel: torch.Tensor | None = None,
) -> torch.Tensor:
  parts = [qpos, qvel]
  if (box_qpos is None) != (box_qvel is None):
    raise ValueError("pass both box_qpos and box_qvel, or neither")
  if box_qpos is not None:
    parts += [box_qpos, box_qvel]
  return torch.cat(parts, dim=-1)


def unpack_state(s: torch.Tensor, has_box: bool | None = None) -> State:
  if has_box is None:
    has_box = s.shape[-1] == STATE_DIM[True]
  if s.shape[-1] != STATE_DIM[has_box]:
    raise ValueError(f"state dim {s.shape[-1]} != {STATE_DIM[has_box]} (has_box={has_box})")
  qpos = s[..., :QPOS_DIM]
  qvel = s[..., QPOS_DIM:ROBOT_STATE_DIM]
  if not has_box:
    return State(qpos, qvel, None, None)
  b = ROBOT_STATE_DIM
  return State(qpos, qvel, s[..., b : b + BOX_QPOS_DIM], s[..., b + BOX_QPOS_DIM :])


def forward_kin(fk: TorchFK, s: torch.Tensor, with_vel: bool = True) -> BodyKin:
  """Robot FK of a packed state (the box is not part of the tree)."""
  return fk(s[..., :QPOS_DIM], s[..., QPOS_DIM:ROBOT_STATE_DIM] if with_vel else None)


_CONSTS: dict[tuple[str, torch.device, torch.dtype], torch.Tensor] = {}


def _const(name: str, ref: torch.Tensor) -> torch.Tensor:
  # Cached per device/dtype so per-step calls do not re-upload small constants.
  key = (name, ref.device, ref.dtype)
  t = _CONSTS.get(key)
  if t is None:
    if name == "q0":
      t = torch.tensor(G1_DEFAULT_JOINT_POS, dtype=ref.dtype, device=ref.device)
    elif name == "gravity":
      t = torch.tensor([0.0, 0.0, -1.0], dtype=ref.dtype, device=ref.device)
    else:
      raise KeyError(name)
    _CONSTS[key] = t
  return t


def _check(s: torch.Tensor, has_box: bool) -> None:
  if s.shape[-1] != STATE_DIM[has_box]:
    raise ValueError(f"state dim {s.shape[-1]} does not match has_box={has_box}")


def _check_kin(kin: BodyKin, need_palms: bool = False) -> None:
  if kin.pos.shape[-2] != NUM_BODIES:
    raise ValueError("features expect FK over all 30 G1 bodies (TorchFK.g1())")
  # Palm ids assume G1_SITE_NAMES order; an FK built with other sites would pick wrong ones.
  if need_palms and kin.site_pos.shape[-2] != len(G1_SITE_NAMES):
    raise ValueError(f"features expect FK sites {G1_SITE_NAMES} (TorchFK.g1())")


def _check_lib(lib: "MotionLibrary") -> None:
  # ANCHOR_BODY_ID indexes the 30-body axis; a 14-body (tracked-only) npz would hit the
  # wrong body silently.
  if lib.num_bodies != NUM_BODIES:
    raise ValueError(f"motion library has {lib.num_bodies} bodies, expected all {NUM_BODIES}")


_INDEX: dict[tuple[tuple[int, ...], torch.device], torch.Tensor] = {}


def _rows(x: torch.Tensor, ids: tuple[int, ...]) -> torch.Tensor:
  # Cached device index: indexing with a Python list would re-upload it on every call.
  key = (ids, x.device)
  idx = _INDEX.get(key)
  if idx is None:
    idx = _INDEX[key] = torch.tensor(ids, dtype=torch.long, device=x.device)
  return x.index_select(-2, idx)


##
# World-model input features φ(s) (design §2.2).
##


def wm_features(
  s: torch.Tensor,
  kin: BodyKin,
  has_box: bool,
  default_joint_pos: torch.Tensor | None = None,
) -> torch.Tensor:
  """Heading-invariant φ(s): invariant to world xy translation and yaw. 169 walk / 191 lift."""
  _check(s, has_box)
  _check_kin(kin, need_palms=has_box)
  if kin.lin_vel is None:
    raise ValueError("wm_features needs FK velocities (call fk with qvel)")
  st = unpack_state(s, has_box)
  q = quat_normalize(st.root_quat)
  yaw = yaw_quat(q)
  yaw_b = yaw.unsqueeze(-2)
  root = st.root_pos
  q0 = _const("q0", s) if default_joint_pos is None else default_joint_pos

  tracked, feet = TRACKED_BODY_IDS, FOOT_BODY_IDS
  rel_pos = quat_apply_inverse(yaw_b, _rows(kin.pos, tracked) - root.unsqueeze(-2))
  rel_vel = quat_apply_inverse(yaw_b, _rows(kin.lin_vel, tracked))
  foot_rot = quat_to_rot6d(quat_mul(quat_conj(yaw_b), _rows(kin.quat, feet)))
  lead = s.shape[:-1]
  parts = [
    root[..., 2:3],
    quat_to_rot6d(quat_mul(quat_conj(yaw), q)),
    quat_apply_inverse(yaw, st.root_lin_vel),
    quat_apply_inverse(q, st.root_ang_vel),
    st.joint_pos - q0,
    st.joint_vel,
    rel_pos.reshape(lead + (-1,)),
    rel_vel.reshape(lead + (-1,)),
    foot_rot.reshape(lead + (-1,)),
    _rows(kin.pos, feet)[..., 2],
  ]
  if has_box:
    bpos = st.box_pos
    bq = quat_normalize(st.box_quat)
    palms = _rows(kin.site_pos, (LEFT_PALM_SITE_ID, RIGHT_PALM_SITE_ID))
    palms_b = quat_apply_inverse(bq.unsqueeze(-2), palms - bpos.unsqueeze(-2))
    parts += [
      quat_apply_inverse(yaw, bpos - root),
      quat_to_rot6d(quat_mul(quat_conj(yaw), bq)),
      quat_apply_inverse(yaw, st.box_lin_vel),
      quat_apply_inverse(yaw, st.box_ang_vel),
      bpos[..., 2:3],
      palms_b.reshape(lead + (-1,)),
    ]
  return torch.cat(parts, dim=-1)


##
# Policy observation frame (brief §2 actor_hist, design §2.3).
##


def obs_frame(
  s: torch.Tensor,
  kin: BodyKin | None,
  a_prev: torch.Tensor,
  xi_raw: torch.Tensor,
  has_box: bool,
  default_joint_pos: torch.Tensor | None = None,
) -> torch.Tensor:
  """One clean actor_hist frame: [q−q0+bias 29, q̇ 29, gyro 3, gravity_b 3, a_prev 29
  (, box pos in pelvis frame 3, rot6d(R_pelvisᵀ R_box) 6)] = 93 / 102.

  gyro = mjlab `imu_ang_vel` (gyro on site imu_in_pelvis, whose frame is the pelvis frame) =
  R_pelvisᵀ ω_w. `kin` is unused for the G1 (everything comes from s) and may be None.
  """
  _check(s, has_box)
  st = unpack_state(s, has_box)
  q = quat_normalize(st.root_quat)
  q0 = _const("q0", s) if default_joint_pos is None else default_joint_pos
  bias = xi_raw[..., XI_ENCODER_BIAS]
  parts = [
    st.joint_pos - q0 + bias,
    st.joint_vel,  # mjlab joint_vel_rel: default joint vel is 0
    quat_apply_inverse(q, st.root_ang_vel),
    quat_apply_inverse(q, _const("gravity", s).expand_as(st.root_ang_vel)),
    a_prev,
  ]
  if has_box:
    parts += [
      quat_apply_inverse(q, st.box_pos - st.root_pos),
      quat_to_rot6d(quat_mul(quat_conj(q), quat_normalize(st.box_quat))),
    ]
  return torch.cat(parts, dim=-1)


def obs_noise_scale(
  has_box: bool, device: str | torch.device = "cpu", dtype: torch.dtype = torch.float32
) -> torch.Tensor:
  """Per-dim uniform noise half-width for one obs frame, in obs_frame order."""
  return torch.cat(
    [torch.full((d,), OBS_NOISE[n], dtype=dtype, device=device) for n, d in obs_terms(has_box)]
  )


def sample_obs_noise(
  lead_shape: Sequence[int],
  has_box: bool,
  device: str | torch.device = "cpu",
  dtype: torch.dtype = torch.float32,
  generator: torch.Generator | None = None,
) -> torch.Tensor:
  """η ~ U(−scale, scale) per dim, shape [*lead_shape, F] (mjlab UniformNoiseCfg)."""
  scale = obs_noise_scale(has_box, device, dtype)
  u = torch.rand((*lead_shape, scale.numel()), device=device, dtype=dtype, generator=generator)
  return (2.0 * u - 1.0) * scale


##
# Goal (design §2.4, brief §2 actor_goal).
##


def goal(
  s: torch.Tensor,
  kin: BodyKin,
  lib: "MotionLibrary",
  ref_index: torch.Tensor,
  K: int = GOAL_K,
  offset: int = 0,
  has_box: bool = False,
) -> torch.Tensor:
  """K future reference frames k = ref_index + offset + {0..K−1}, clamped at the clip end.

  Per frame: [q_ref 29, q̇_ref 29, rot6d(R_robot_anchorᵀ R_ref_anchor) 6
  (, R_Hᵀ(p_boxref − p_pelvis) 3, rot6d(R_pelvisᵀ R_boxref) 6)], frames concatenated in time
  order. The anchor term is recomputed from s (via kin), so it carries gradients in imagination.
  """
  _check(s, has_box)
  _check_kin(kin)
  _check_lib(lib)
  st = unpack_state(s, has_box)
  steps = torch.arange(K, device=ref_index.device, dtype=torch.long)
  k = lib.clamp(ref_index.long().unsqueeze(-1) + offset + steps)  # [*, K]
  robot_anchor = kin.quat[..., ANCHOR_BODY_ID, :].unsqueeze(-2)
  ref_anchor = lib.body_quat_w[k, ANCHOR_BODY_ID]
  parts = [
    lib.joint_pos[k],
    lib.joint_vel[k],
    quat_to_rot6d(quat_mul(quat_conj(robot_anchor), ref_anchor)),
  ]
  if has_box:
    if not lib.has_box:
      raise ValueError("lift goal needs a motion library with box keys")
    q = quat_normalize(st.root_quat).unsqueeze(-2)
    parts += [
      quat_apply_inverse(yaw_quat(q), lib.box_pos_w[k] - st.root_pos.unsqueeze(-2)),
      quat_to_rot6d(quat_mul(quat_conj(q), lib.box_quat_w[k])),
    ]
  out = torch.cat(parts, dim=-1)  # [*, K, 64|73]
  return out.reshape(out.shape[:-2] + (-1,))


##
# DR context encoding (design §2.5).
##


def encode_xi(xi: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor) -> torch.Tensor:
  """2(ξ−lo)/(hi−lo)−1, and 0 on dims that are not randomized (hi == lo)."""
  span = hi - lo
  valid = span > 0
  safe = torch.where(valid, span, torch.ones_like(span))
  return torch.where(valid, 2.0 * (xi - lo) / safe - 1.0, torch.zeros_like(xi))


##
# Observation history.
##

HistoryLayout = Literal["mjlab", "frame"]


def flatten_history(
  frames: torch.Tensor,
  term_dims: Sequence[int] | None = None,
  layout: HistoryLayout = "mjlab",
) -> torch.Tensor:
  """[N, H, F] oldest->newest -> [N, H*F].

  "mjlab": mjlab ObservationManager layout for a group with history_length=H: each term's H
  frames flattened (oldest first), then terms concatenated: [t1(h0..h9) | t2(h0..h9) | ...].
  "frame": [frame h0 | frame h1 | ...].
  """
  h, f = frames.shape[-2:]
  lead = frames.shape[:-2]
  if layout == "frame":
    return frames.reshape(lead + (h * f,))
  if term_dims is None:
    term_dims = _default_term_dims(f)
  chunks = torch.split(frames, list(term_dims), dim=-1)
  return torch.cat([c.reshape(lead + (-1,)) for c in chunks], dim=-1)


def unflatten_history(
  flat: torch.Tensor,
  history: int,
  term_dims: Sequence[int] | None = None,
  layout: HistoryLayout = "mjlab",
) -> torch.Tensor:
  """Inverse of flatten_history: [N, H*F] -> [N, H, F]."""
  lead = flat.shape[:-1]
  f = flat.shape[-1] // history
  if layout == "frame":
    return flat.reshape(lead + (history, f))
  if term_dims is None:
    term_dims = _default_term_dims(f)
  chunks = torch.split(flat, [history * d for d in term_dims], dim=-1)
  return torch.cat([c.reshape(lead + (history, -1)) for c in chunks], dim=-1)


def _default_term_dims(frame_dim: int) -> tuple[int, ...]:
  for has_box in (False, True):
    if frame_dim == obs_frame_dim(has_box):
      return obs_term_dims(has_box)
  raise ValueError(f"no default obs terms for frame dim {frame_dim}; pass term_dims")


class PolicyObsHistory:
  """Last `history` obs frames per env, oldest -> newest, for sim and imagination alike.

  Matches mjlab's CircularBuffer: after reset(ids, frame) every slot holds that first frame.
  push/reset are out-of-place (the buffer is rebound), so a history built from imagined
  frames stays on the autograd graph; call detach_() between sim steps if frames carry grads.
  get() flattens with `layout` ("mjlab" by default = the env's actor_hist group layout).
  """

  def __init__(
    self,
    num_envs: int,
    history: int = HISTORY_LEN,
    frame_dim: int = 93,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
    layout: HistoryLayout = "mjlab",
    term_dims: Sequence[int] | None = None,
  ) -> None:
    self.num_envs = num_envs
    self.history = history
    self.frame_dim = frame_dim
    self.layout: HistoryLayout = layout
    self.term_dims = tuple(term_dims) if term_dims is not None else (
      _default_term_dims(frame_dim) if layout == "mjlab" else (frame_dim,)
    )
    assert sum(self.term_dims) == frame_dim
    self.buf = torch.zeros(num_envs, history, frame_dim, device=device, dtype=dtype)

  @classmethod
  def from_frames(
    cls,
    frames: torch.Tensor,
    layout: HistoryLayout = "mjlab",
    term_dims: Sequence[int] | None = None,
  ) -> "PolicyObsHistory":
    """Wrap an existing [N, H, F] oldest->newest stack (e.g. a replay `sample_starts` rebuild)."""
    n, h, f = frames.shape
    obj = cls(n, h, f, device=frames.device, dtype=frames.dtype, layout=layout, term_dims=term_dims)
    obj.buf = frames
    return obj

  def reset(self, ids: torch.Tensor | Sequence[int] | None, frame: torch.Tensor) -> None:
    """Fill every slot of envs `ids` (None = all) with `frame` ([len(ids), F], or [N, F])."""
    if ids is None:
      ids = torch.arange(self.num_envs, device=self.buf.device)
    ids = torch.as_tensor(ids, device=self.buf.device)
    if ids.dtype == torch.bool:
      ids = ids.nonzero(as_tuple=False).squeeze(-1)
    if ids.numel() == 0:
      return
    if frame.shape[0] == self.num_envs and ids.numel() != self.num_envs:
      frame = frame[ids]
    fill = frame.to(self.buf.dtype).unsqueeze(1).expand(-1, self.history, -1)
    self.buf = self.buf.index_put((ids,), fill)

  def push(self, frame: torch.Tensor) -> None:
    """Append the newest frame [N, F]; the oldest drops out."""
    self.buf = torch.cat([self.buf[:, 1:], frame.unsqueeze(1).to(self.buf.dtype)], dim=1)

  def get(self) -> torch.Tensor:
    return flatten_history(self.buf, self.term_dims, self.layout)

  def frames(self) -> torch.Tensor:
    """Raw [N, H, F] stack, oldest -> newest."""
    return self.buf

  def detach_(self) -> "PolicyObsHistory":
    self.buf = self.buf.detach()
    return self

  def clone(self) -> "PolicyObsHistory":
    return PolicyObsHistory.from_frames(self.buf.clone(), self.layout, self.term_dims)
