"""Differentiable torch forward kinematics for mjlab's Unitree G1 (and other hinge-only trees).

`TorchFK` is built once from a CPU `mujoco.MjModel` and reproduces `mj_kinematics` (body/site
poses) and `mj_objectVelocity(mjOBJ_XBODY, flg_local=0)` (link-origin world velocities, i.e.
mjlab `body_link_lin_vel_w` / `body_link_ang_vel_w`). Warp FK has no backward pass, so the world
model, reward and features run on this instead.

State convention (brief §1): qpos [*,36] = [root pos (env-local), root quat wxyz, 29 joints];
qvel [*,35] = [root lin vel (world), root ang vel (WORLD, not MuJoCo's body-local), 29 joints].

The G1 constants below are plain Python so `features.py` can index bodies without importing
mujoco or mjlab; `TorchFK.g1()` asserts the compiled XML still matches them, and
tests/supermimic/test_kinematics.py cross-checks them against mjlab's configs.
"""

from __future__ import annotations

import importlib.util
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import torch

from mjlab_g1.rl.supermimic.geometry import (
  cross,
  quat_apply,
  quat_from_axis_angle,
  quat_mul,
  quat_normalize,
)

##
# mjlab G1 constants (mjlab 1.4.0, asset_zoo/robots/unitree_g1).
##

# Entity body order = MuJoCo body ids 1..30 of g1.xml (world excluded); the tracking npz uses it.
G1_BODY_NAMES: tuple[str, ...] = (
  "pelvis",
  "left_hip_pitch_link",
  "left_hip_roll_link",
  "left_hip_yaw_link",
  "left_knee_link",
  "left_ankle_pitch_link",
  "left_ankle_roll_link",
  "right_hip_pitch_link",
  "right_hip_roll_link",
  "right_hip_yaw_link",
  "right_knee_link",
  "right_ankle_pitch_link",
  "right_ankle_roll_link",
  "waist_yaw_link",
  "waist_roll_link",
  "torso_link",
  "left_shoulder_pitch_link",
  "left_shoulder_roll_link",
  "left_shoulder_yaw_link",
  "left_elbow_link",
  "left_wrist_roll_link",
  "left_wrist_pitch_link",
  "left_wrist_yaw_link",
  "right_shoulder_pitch_link",
  "right_shoulder_roll_link",
  "right_shoulder_yaw_link",
  "right_elbow_link",
  "right_wrist_roll_link",
  "right_wrist_pitch_link",
  "right_wrist_yaw_link",
)

G1_JOINT_NAMES: tuple[str, ...] = (
  "left_hip_pitch_joint",
  "left_hip_roll_joint",
  "left_hip_yaw_joint",
  "left_knee_joint",
  "left_ankle_pitch_joint",
  "left_ankle_roll_joint",
  "right_hip_pitch_joint",
  "right_hip_roll_joint",
  "right_hip_yaw_joint",
  "right_knee_joint",
  "right_ankle_pitch_joint",
  "right_ankle_roll_joint",
  "waist_yaw_joint",
  "waist_roll_joint",
  "waist_pitch_joint",
  "left_shoulder_pitch_joint",
  "left_shoulder_roll_joint",
  "left_shoulder_yaw_joint",
  "left_elbow_joint",
  "left_wrist_roll_joint",
  "left_wrist_pitch_joint",
  "left_wrist_yaw_joint",
  "right_shoulder_pitch_joint",
  "right_shoulder_roll_joint",
  "right_shoulder_yaw_joint",
  "right_elbow_joint",
  "right_wrist_roll_joint",
  "right_wrist_pitch_joint",
  "right_wrist_yaw_joint",
)

# mjlab tasks/tracking/config/g1/env_cfgs.py: motion_cmd.body_names (pelvis first) and anchor.
G1_TRACKED_BODY_NAMES: tuple[str, ...] = (
  "pelvis",
  "left_hip_roll_link",
  "left_knee_link",
  "left_ankle_roll_link",
  "right_hip_roll_link",
  "right_knee_link",
  "right_ankle_roll_link",
  "torso_link",
  "left_shoulder_roll_link",
  "left_elbow_link",
  "left_wrist_yaw_link",
  "right_shoulder_roll_link",
  "right_elbow_link",
  "right_wrist_yaw_link",
)
G1_ANCHOR_BODY_NAME = "torso_link"
G1_FOOT_BODY_NAMES: tuple[str, ...] = ("left_ankle_roll_link", "right_ankle_roll_link")
# Sites TorchFK.g1() evaluates, in this order (palms for the box task, IMU for the gyro).
G1_SITE_NAMES: tuple[str, ...] = ("left_palm", "right_palm", "imu_in_pelvis")

G1_SOFT_LIMIT_FACTOR = 0.9  # G1_ARTICULATION.soft_joint_pos_limit_factor

# KNEES_BENT_KEYFRAME (the mjlab G1 default pose); resolved like mjlab's resolve_expr:
# first pattern that re.match-es the joint name wins, unmatched joints are 0.
G1_KNEES_BENT_JOINT_POS: dict[str, float] = {
  ".*_hip_pitch_joint": -0.312,
  ".*_knee_joint": 0.669,
  ".*_ankle_pitch_joint": -0.363,
  ".*_elbow_joint": 0.6,
  "left_shoulder_roll_joint": 0.2,
  "left_shoulder_pitch_joint": 0.2,
  "right_shoulder_roll_joint": -0.2,
  "right_shoulder_pitch_joint": 0.2,
}
G1_KNEES_BENT_ROOT_POS: tuple[float, float, float] = (0.0, 0.0, 0.76)

# Indices into the 30-body list / G1_SITE_NAMES (what features.py and the reward use).
PELVIS_BODY_ID = G1_BODY_NAMES.index("pelvis")
ANCHOR_BODY_ID = G1_BODY_NAMES.index(G1_ANCHOR_BODY_NAME)
TRACKED_BODY_IDS: tuple[int, ...] = tuple(G1_BODY_NAMES.index(n) for n in G1_TRACKED_BODY_NAMES)
ANCHOR_TRACKED_IDX = G1_TRACKED_BODY_NAMES.index(G1_ANCHOR_BODY_NAME)
FOOT_BODY_IDS: tuple[int, ...] = tuple(G1_BODY_NAMES.index(n) for n in G1_FOOT_BODY_NAMES)
LEFT_PALM_SITE_ID = G1_SITE_NAMES.index("left_palm")
RIGHT_PALM_SITE_ID = G1_SITE_NAMES.index("right_palm")
IMU_SITE_ID = G1_SITE_NAMES.index("imu_in_pelvis")
NUM_BODIES = len(G1_BODY_NAMES)
NUM_JOINTS = len(G1_JOINT_NAMES)


def resolve_joint_values(
  pattern_map: Mapping[str, float], joint_names: Sequence[str], default: float = 0.0
) -> tuple[float, ...]:
  """mjlab `resolve_expr` semantics: first re.match wins, else default."""
  pats = [(re.compile(p), v) for p, v in pattern_map.items()]
  out = []
  for name in joint_names:
    for pat, val in pats:
      if pat.match(name):
        out.append(float(val))
        break
    else:
      out.append(float(default))
  return tuple(out)


G1_DEFAULT_JOINT_POS: tuple[float, ...] = resolve_joint_values(
  G1_KNEES_BENT_JOINT_POS, G1_JOINT_NAMES
)


def g1_xml_path() -> Path:
  """mjlab's g1.xml, located without importing mjlab (its import pulls in warp)."""
  spec = importlib.util.find_spec("mjlab")
  if spec is None or not spec.submodule_search_locations:
    raise ModuleNotFoundError("mjlab is not installed")
  root = Path(list(spec.submodule_search_locations)[0])
  return root / "asset_zoo" / "robots" / "unitree_g1" / "xmls" / "g1.xml"


##
# Forward kinematics.
##


@dataclass
class BodyKin:
  """World-frame (env-local) kinematics. Velocities are None when qvel was not given."""

  pos: torch.Tensor  # [*, B, 3] link origin
  quat: torch.Tensor  # [*, B, 4] wxyz
  lin_vel: torch.Tensor | None  # [*, B, 3] link-origin linear velocity, world
  ang_vel: torch.Tensor | None  # [*, B, 3] world
  site_pos: torch.Tensor  # [*, S, 3]
  site_quat: torch.Tensor  # [*, S, 4]


class TorchFK(torch.nn.Module):
  """Tree FK batched by depth: each level gathers its parents from the previous level only."""

  def __init__(
    self,
    *,
    all_body_names: Sequence[str],
    body_depth: Sequence[int],
    body_parent: Sequence[int],
    body_pos: torch.Tensor,
    body_quat: torch.Tensor,
    body_joint: Sequence[int],
    jnt_axis: torch.Tensor,
    jnt_pos: torch.Tensor,
    jnt_qpos0: torch.Tensor,
    joint_names: Sequence[str],
    joint_range: torch.Tensor,
    out_body_names: Sequence[str],
    site_names: Sequence[str],
    site_body: Sequence[int],
    site_pos: torch.Tensor,
    site_quat: torch.Tensor,
    soft_limit_factor: float,
    default_joint_pos: torch.Tensor,
  ) -> None:
    """Arrays are indexed by tree body index 0..nb-1 (0 = free-floating root, world excluded);
    `body_joint[b]` is the hinge index (0..nj-1) of body b (-1 for the root)."""
    super().__init__()
    self.all_body_names = tuple(all_body_names)
    self.body_names = tuple(out_body_names)
    self.site_names = tuple(site_names)
    self.joint_names = tuple(joint_names)
    nb = len(self.all_body_names)

    # Depth-ordered slots; level d holds the bodies at tree depth d, in body-index order.
    max_depth = max(body_depth)
    levels = [[b for b in range(nb) if body_depth[b] == d] for d in range(max_depth + 1)]
    assert levels[0] == [0], "expected a single floating root at body index 0"
    slot_of = {}
    offset = 0
    for lv in levels:
      for i, b in enumerate(lv):
        slot_of[b] = (offset + i, i)
      offset += len(lv)
    self._num_levels = len(levels)
    for d, lv in enumerate(levels[1:], start=1):
      idx = torch.tensor(lv, dtype=torch.long)
      parent_local = torch.tensor([slot_of[body_parent[b]][1] for b in lv], dtype=torch.long)
      jidx = torch.tensor([body_joint[b] for b in lv], dtype=torch.long)
      self.register_buffer(f"_l{d}_parent", parent_local, persistent=False)
      self.register_buffer(f"_l{d}_jidx", jidx, persistent=False)
      self.register_buffer(f"_l{d}_bpos", body_pos[idx].clone(), persistent=False)
      self.register_buffer(f"_l{d}_bquat", body_quat[idx].clone(), persistent=False)
      self.register_buffer(f"_l{d}_axis", jnt_axis[jidx].clone(), persistent=False)
      self.register_buffer(f"_l{d}_jpos", jnt_pos[jidx].clone(), persistent=False)
      self.register_buffer(f"_l{d}_qpos0", jnt_qpos0[jidx].clone(), persistent=False)
      # Skip the anchor terms when every joint sits at its body origin (true for the G1).
      setattr(self, f"_l{d}_has_jpos", bool(jnt_pos[jidx].abs().max() > 0))

    name_to_b = {n: i for i, n in enumerate(self.all_body_names)}
    out_slots = [slot_of[name_to_b[n]][0] for n in self.body_names]
    self.register_buffer("_out_slots", torch.tensor(out_slots, dtype=torch.long), persistent=False)
    site_slots = [slot_of[b][0] for b in site_body]
    site_slots_t = torch.tensor(site_slots, dtype=torch.long)
    self.register_buffer("_site_slots", site_slots_t, persistent=False)
    self.register_buffer("_site_pos", site_pos.clone(), persistent=False)
    self.register_buffer("_site_quat", site_quat.clone(), persistent=False)

    self.register_buffer("joint_range", joint_range.clone(), persistent=False)
    mean = 0.5 * (joint_range[:, 0] + joint_range[:, 1])
    half = 0.5 * (joint_range[:, 1] - joint_range[:, 0]) * soft_limit_factor
    self.soft_limit_factor = float(soft_limit_factor)
    soft = torch.stack([mean - half, mean + half], -1)
    self.register_buffer("soft_joint_limits", soft, persistent=False)
    self.register_buffer("default_joint_pos", default_joint_pos.clone(), persistent=False)

  # ----- construction -----

  @classmethod
  def from_xml(
    cls,
    xml_path: str | Path,
    body_names: Sequence[str] | None = None,
    site_names: Sequence[str] = (),
    soft_limit_factor: float = 1.0,
    default_joint_pos: Mapping[str, float] | Sequence[float] | None = None,
    dtype: torch.dtype = torch.float32,
  ) -> "TorchFK":
    """Build from an MJCF. `body_names` selects/orders the output bodies (default: all bodies
    except world, in MuJoCo id order = mjlab entity order)."""
    import mujoco  # local: features/tests that only need the constants stay mujoco-free

    m = mujoco.MjModel.from_xml_path(str(xml_path))
    return cls.from_model(m, body_names, site_names, soft_limit_factor, default_joint_pos, dtype)

  @classmethod
  def from_model(
    cls,
    m,  # mujoco.MjModel
    body_names: Sequence[str] | None = None,
    site_names: Sequence[str] = (),
    soft_limit_factor: float = 1.0,
    default_joint_pos: Mapping[str, float] | Sequence[float] | None = None,
    dtype: torch.dtype = torch.float32,
  ) -> "TorchFK":
    import mujoco

    nb = m.nbody - 1  # tree index b = MuJoCo body id b+1
    names = [m.body(i + 1).name for i in range(nb)]
    root_children = [i for i in range(1, m.nbody) if m.body_parentid[i] == 0]
    free, hinge = int(mujoco.mjtJoint.mjJNT_FREE), int(mujoco.mjtJoint.mjJNT_HINGE)
    if root_children != [1] or m.body_jntnum[1] != 1 or int(m.jnt_type[m.body_jntadr[1]]) != free:
      raise ValueError("TorchFK expects exactly one free-floating root body under world")
    if m.jnt_qposadr[m.body_jntadr[1]] != 0 or m.jnt_dofadr[m.body_jntadr[1]] != 0:
      raise ValueError("free joint must own qpos[0:7] / qvel[0:6]")

    depth = [0] * nb
    parent = [-1] * nb
    body_joint = [-1] * nb
    hinge_ids = [j for j in range(m.njnt) if int(m.jnt_type[j]) == hinge]
    if len(hinge_ids) != m.njnt - 1:
      raise ValueError("TorchFK supports one free root joint plus hinge joints only")
    joint_names = [m.joint(j).name for j in hinge_ids]
    for k, j in enumerate(hinge_ids):
      # Hinge k must own qpos[7+k] / qvel[6+k] so the 29-vectors line up with the joint list.
      if m.jnt_qposadr[j] != 7 + k or m.jnt_dofadr[j] != 6 + k:
        raise ValueError(f"joint {joint_names[k]} is not at qpos {7 + k}")
    hinge_of_body = {}
    for k, j in enumerate(hinge_ids):
      b = int(m.jnt_bodyid[j]) - 1
      if b in hinge_of_body:
        raise ValueError(f"body {names[b]} has more than one joint")
      hinge_of_body[b] = k
    for b in range(1, nb):
      p = int(m.body_parentid[b + 1]) - 1
      if p < 0 or b not in hinge_of_body:
        raise ValueError(f"body {names[b]} must hang off the tree by exactly one hinge")
      parent[b] = p
      depth[b] = depth[p] + 1  # MuJoCo ids are DFS-ordered: parents precede children
      body_joint[b] = hinge_of_body[b]

    f64 = torch.float64
    body_pos = torch.tensor(m.body_pos[1:], dtype=f64)
    body_quat = torch.tensor(m.body_quat[1:], dtype=f64)
    jnt_axis = torch.tensor(m.jnt_axis[hinge_ids], dtype=f64)
    jnt_pos = torch.tensor(m.jnt_pos[hinge_ids], dtype=f64)
    qpos0 = torch.tensor(m.qpos0[[m.jnt_qposadr[j] for j in hinge_ids]], dtype=f64)
    # Same float32 arithmetic as mjlab (warp jnt_range is float32) for bit-close soft limits.
    joint_range = torch.tensor(m.jnt_range[hinge_ids], dtype=torch.float32)
    if not all(m.jnt_limited[j] for j in hinge_ids):
      raise ValueError("unlimited hinge joints are not supported (soft limits would be inf)")

    out_names = list(names) if body_names is None else list(body_names)
    for n in out_names:
      if n not in names:
        raise KeyError(f"unknown body {n!r}")
    site_body, site_pos, site_quat = [], [], []
    for sname in site_names:
      sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, sname)
      if sid < 0:
        raise KeyError(f"unknown site {sname!r}")
      if m.site_bodyid[sid] == 0:
        raise ValueError(f"site {sname!r} is attached to world")
      site_body.append(int(m.site_bodyid[sid]) - 1)
      site_pos.append(m.site_pos[sid].tolist())
      site_quat.append(m.site_quat[sid].tolist())
    site_pos_t = torch.tensor(site_pos, dtype=f64).reshape(-1, 3)
    site_quat_t = torch.tensor(site_quat, dtype=f64).reshape(-1, 4)

    if default_joint_pos is None:
      q0 = torch.zeros(len(joint_names), dtype=f64)
    elif isinstance(default_joint_pos, Mapping):
      q0 = torch.tensor(resolve_joint_values(default_joint_pos, joint_names), dtype=f64)
    else:
      q0 = torch.as_tensor(default_joint_pos, dtype=f64)
      assert q0.shape == (len(joint_names),)

    fk = cls(
      all_body_names=names,
      body_depth=depth,
      body_parent=parent,
      body_pos=body_pos,
      body_quat=body_quat,
      body_joint=body_joint,
      jnt_axis=jnt_axis,
      jnt_pos=jnt_pos,
      jnt_qpos0=qpos0,
      joint_names=joint_names,
      joint_range=joint_range,
      out_body_names=out_names,
      site_names=site_names,
      site_body=site_body,
      site_pos=site_pos_t,
      site_quat=site_quat_t,
      soft_limit_factor=soft_limit_factor,
      default_joint_pos=q0,
    )
    # Soft limits were computed in float32 like mjlab; a later cast to `dtype` keeps the values.
    return fk.to(dtype)

  @classmethod
  def g1(
    cls,
    site_names: Sequence[str] = G1_SITE_NAMES,
    dtype: torch.dtype = torch.float32,
    device: str | torch.device = "cpu",
  ) -> "TorchFK":
    """mjlab G1: all 30 bodies, palms + IMU sites, soft factor 0.9, KNEES_BENT default pose."""
    fk = cls.from_xml(
      g1_xml_path(),
      body_names=None,
      site_names=site_names,
      soft_limit_factor=G1_SOFT_LIMIT_FACTOR,
      default_joint_pos=G1_KNEES_BENT_JOINT_POS,
      dtype=dtype,
    )
    if fk.body_names != G1_BODY_NAMES or fk.joint_names != G1_JOINT_NAMES:
      raise RuntimeError("mjlab g1.xml no longer matches the G1 constants in kinematics.py")
    return fk.to(device)

  # ----- lookup helpers -----

  def body_index(self, name: str) -> int:
    return self.body_names.index(name)

  def body_indices(self, names: Sequence[str]) -> torch.Tensor:
    ids = [self.body_names.index(n) for n in names]
    return torch.tensor(ids, dtype=torch.long, device=self._out_slots.device)

  def site_index(self, name: str) -> int:
    return self.site_names.index(name)

  # ----- forward -----

  def forward(self, qpos: torch.Tensor, qvel: torch.Tensor | None = None) -> BodyKin:
    root_p = qpos[..., 0:3]
    root_q = quat_normalize(qpos[..., 3:7])
    qj = qpos[..., 7:]
    pos_l = [root_p.unsqueeze(-2)]
    quat_l = [root_q.unsqueeze(-2)]
    with_vel = qvel is not None
    if with_vel:
      qdj = qvel[..., 6:]
      lin_l = [qvel[..., 0:3].unsqueeze(-2)]
      ang_l = [qvel[..., 3:6].unsqueeze(-2)]

    for d in range(1, self._num_levels):
      par = getattr(self, f"_l{d}_parent")
      jidx = getattr(self, f"_l{d}_jidx")
      axis = getattr(self, f"_l{d}_axis")
      pp = pos_l[-1].index_select(-2, par)
      pq = quat_l[-1].index_select(-2, par)
      q_pre = quat_mul(pq, getattr(self, f"_l{d}_bquat"))  # frame before the joint rotates
      p_pre = pp + quat_apply(pq, getattr(self, f"_l{d}_bpos"))
      angle = qj.index_select(-1, jidx) - getattr(self, f"_l{d}_qpos0")
      quat = quat_normalize(quat_mul(q_pre, quat_from_axis_angle(axis, angle)))
      has_jpos = getattr(self, f"_l{d}_has_jpos")
      if has_jpos:
        jpos = getattr(self, f"_l{d}_jpos")
        anchor = p_pre + quat_apply(q_pre, jpos)
        pos = anchor - quat_apply(quat, jpos)
      else:
        anchor = p_pre
        pos = p_pre
      pos_l.append(pos)
      quat_l.append(quat)
      if with_vel:
        pw = ang_l[-1].index_select(-2, par)
        pv = lin_l[-1].index_select(-2, par)
        axis_w = quat_apply(q_pre, axis)
        qd = qdj.index_select(-1, jidx).unsqueeze(-1)
        ang = pw + axis_w * qd
        lin = pv + cross(pw, pos - pp)
        if has_jpos:
          lin = lin + qd * cross(axis_w.expand_as(pos), pos - anchor)
        lin_l.append(lin)
        ang_l.append(ang)

    pos_all = torch.cat(pos_l, dim=-2)
    quat_all = torch.cat(quat_l, dim=-2)

    def out(x: torch.Tensor) -> torch.Tensor:
      return x.index_select(-2, self._out_slots)

    sb_pos = pos_all.index_select(-2, self._site_slots)
    sb_quat = quat_all.index_select(-2, self._site_slots)
    site_pos = sb_pos + quat_apply(sb_quat, self._site_pos)
    site_quat = quat_mul(sb_quat, self._site_quat)
    return BodyKin(
      pos=out(pos_all),
      quat=out(quat_all),
      lin_vel=out(torch.cat(lin_l, dim=-2)) if with_vel else None,
      ang_vel=out(torch.cat(ang_l, dim=-2)) if with_vel else None,
      site_pos=site_pos,
      site_quat=site_quat,
    )
