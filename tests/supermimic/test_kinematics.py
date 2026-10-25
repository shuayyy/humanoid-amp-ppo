"""TorchFK vs CPU MuJoCo, G1 constants vs mjlab, and geometry helpers (CPU, a few seconds)."""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
import pytest
import torch

from mjlab_g1.rl.supermimic import geometry as geo
from mjlab_g1.rl.supermimic.kinematics import (
  ANCHOR_BODY_ID,
  G1_ANCHOR_BODY_NAME,
  G1_BODY_NAMES,
  G1_DEFAULT_JOINT_POS,
  G1_JOINT_NAMES,
  G1_SITE_NAMES,
  G1_SOFT_LIMIT_FACTOR,
  G1_TRACKED_BODY_NAMES,
  TRACKED_BODY_IDS,
  TorchFK,
  g1_xml_path,
)

REPO = Path(__file__).resolve().parents[2]
WALK_NPZ = REPO / "dataset" / "tracking" / "locomotion1.npz"


@pytest.fixture(scope="module")
def mj():
  m = mujoco.MjModel.from_xml_path(str(g1_xml_path()))
  return m, mujoco.MjData(m)


@pytest.fixture(scope="module")
def fk64():
  return TorchFK.g1(dtype=torch.float64)


@pytest.fixture(scope="module")
def fk32():
  return TorchFK.g1(dtype=torch.float32)


def random_states(m, n: int, seed: int = 0, root_xy: float = 5.0):
  """Random G1 states in MuJoCo convention (qvel[3:6] body-local) plus our world-ω qvel."""
  rng = np.random.default_rng(seed)
  qpos = np.zeros((n, m.nq))
  qpos[:, 0:2] = rng.uniform(-root_xy, root_xy, (n, 2))  # env-local root, a few meters
  qpos[:, 2] = rng.uniform(0.2, 1.2, n)
  quat = rng.normal(size=(n, 4))
  qpos[:, 3:7] = quat / np.linalg.norm(quat, axis=1, keepdims=True)
  lo, hi = m.jnt_range[1:, 0], m.jnt_range[1:, 1]
  qpos[:, 7:] = rng.uniform(lo, hi, (n, m.nq - 7))
  qvel_mj = rng.normal(size=(n, m.nv))
  qvel_mj[:, 6:] *= 2.0
  return qpos, qvel_mj


def mujoco_reference(m, d, qpos, qvel_mj, site_names=G1_SITE_NAMES):
  n, nb = qpos.shape[0], m.nbody - 1
  out = {k: np.zeros((n, nb, 3)) for k in ("pos", "lin", "ang")}
  out["quat"] = np.zeros((n, nb, 4))
  out["site_pos"] = np.zeros((n, len(site_names), 3))
  out["site_quat"] = np.zeros((n, len(site_names), 4))
  qvel_w = qvel_mj.copy()
  sids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, s) for s in site_names]
  res = np.zeros(6)
  for i in range(n):
    d.qpos[:] = qpos[i]
    d.qvel[:] = qvel_mj[i]
    mujoco.mj_kinematics(m, d)
    mujoco.mj_comPos(m, d)
    mujoco.mj_comVel(m, d)
    out["pos"][i] = d.xpos[1:]
    out["quat"][i] = d.xquat[1:]
    for b in range(1, m.nbody):
      mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_XBODY, b, res, 0)
      out["ang"][i, b - 1] = res[:3]
      out["lin"][i, b - 1] = res[3:]
    qvel_w[i, 3:6] = d.xmat[1].reshape(3, 3) @ qvel_mj[i, 3:6]
    for k, s in enumerate(sids):
      out["site_pos"][i, k] = d.site_xpos[s]
      mujoco.mju_mat2Quat(out["site_quat"][i, k], d.site_xmat[s])
  return out, qvel_w


def quat_err(a: torch.Tensor, b: np.ndarray) -> float:
  b = torch.as_tensor(b, dtype=torch.float64)
  a = a.double()
  return torch.minimum((a - b).abs().amax(-1), (a + b).abs().amax(-1)).max().item()


def max_err(a: torch.Tensor, b: np.ndarray) -> float:
  return (a.double() - torch.as_tensor(b, dtype=torch.float64)).abs().max().item()


@pytest.mark.parametrize("dtype,tol", [(torch.float64, 1e-10), (torch.float32, 1e-5)])
def test_fk_matches_mujoco(mj, dtype, tol):
  m, d = mj
  qpos, qvel_mj = random_states(m, 48)
  ref, qvel_w = mujoco_reference(m, d, qpos, qvel_mj)
  fk = TorchFK.g1(dtype=dtype)
  kin = fk(torch.tensor(qpos, dtype=dtype), torch.tensor(qvel_w, dtype=dtype))
  assert kin.pos.shape == (48, 30, 3) and kin.site_pos.shape == (48, 3, 3)
  assert max_err(kin.pos, ref["pos"]) < tol
  assert quat_err(kin.quat, ref["quat"]) < tol
  assert max_err(kin.lin_vel, ref["lin"]) < tol
  assert max_err(kin.ang_vel, ref["ang"]) < tol
  assert max_err(kin.site_pos, ref["site_pos"]) < tol
  assert quat_err(kin.site_quat, ref["site_quat"]) < tol


def test_root_velocity_convention(mj):
  """mjlab root_link_vel_w = (qvel[0:3], R qvel_local[3:6]): our qvel[3:6] is that world ω."""
  m, d = mj
  qpos, qvel_mj = random_states(m, 4, seed=1)
  ref, qvel_w = mujoco_reference(m, d, qpos, qvel_mj)
  np.testing.assert_allclose(ref["lin"][:, 0], qvel_mj[:, 0:3], atol=1e-12)
  np.testing.assert_allclose(ref["ang"][:, 0], qvel_w[:, 3:6], atol=1e-12)


def test_fk_env_local_translation(fk64, mj):
  m, _ = mj
  qpos, qvel = random_states(m, 8, seed=2)
  qpos_t, qvel_t = torch.tensor(qpos), torch.tensor(qvel)
  offset = torch.tensor([3.0, -7.5, 0.0], dtype=torch.float64)
  a = fk64(qpos_t, qvel_t)
  shifted = qpos_t.clone()
  shifted[:, :3] += offset
  b = fk64(shifted, qvel_t)
  torch.testing.assert_close(b.pos, a.pos + offset, atol=1e-12, rtol=0)
  torch.testing.assert_close(b.site_pos, a.site_pos + offset, atol=1e-12, rtol=0)
  torch.testing.assert_close(b.quat, a.quat)
  torch.testing.assert_close(b.lin_vel, a.lin_vel)
  torch.testing.assert_close(b.ang_vel, a.ang_vel)


def test_fk_unnormalized_root_quat(fk64, mj):
  m, _ = mj
  qpos, qvel = random_states(m, 4, seed=3)
  scaled = qpos.copy()
  scaled[:, 3:7] *= 1.7  # MuJoCo normalizes the free-joint quaternion; so do we
  a = fk64(torch.tensor(qpos), torch.tensor(qvel))
  b = fk64(torch.tensor(scaled), torch.tensor(qvel))
  torch.testing.assert_close(a.pos, b.pos)
  torch.testing.assert_close(a.quat, b.quat)


def test_fk_batch_dims_subset_and_no_vel(mj):
  m, _ = mj
  qpos, qvel = random_states(m, 6, seed=4)
  full = TorchFK.g1(dtype=torch.float64)
  sub = TorchFK.from_xml(
    g1_xml_path(), body_names=("torso_link", "pelvis", "left_wrist_yaw_link"), dtype=torch.float64
  )
  qp = torch.tensor(qpos).reshape(2, 3, -1)
  qv = torch.tensor(qvel).reshape(2, 3, -1)
  kin_full = full(qp, qv)
  kin_sub = sub(qp, qv)
  assert kin_full.pos.shape == (2, 3, 30, 3) and kin_full.site_quat.shape == (2, 3, 3, 4)
  ids = [G1_BODY_NAMES.index(n) for n in sub.body_names]
  torch.testing.assert_close(kin_sub.pos, kin_full.pos[..., ids, :])
  torch.testing.assert_close(kin_sub.lin_vel, kin_full.lin_vel[..., ids, :])
  assert kin_sub.site_pos.shape == (2, 3, 0, 3)
  no_vel = full(qp)
  assert no_vel.lin_vel is None and no_vel.ang_vel is None
  torch.testing.assert_close(no_vel.pos, kin_full.pos)


def test_fk_matches_motion_npz(fk32):
  """The tracking npz (mjlab sim FK) uses the same 30-body order as TorchFK."""
  if not WALK_NPZ.exists():
    pytest.skip("dataset/tracking/locomotion1.npz not present")
  data = np.load(WALK_NPZ)
  bp = torch.tensor(data["body_pos_w"])
  bq = torch.tensor(data["body_quat_w"])
  qpos = torch.cat([bp[:, 0], bq[:, 0], torch.tensor(data["joint_pos"])], -1)
  qvel = torch.cat(
    [torch.tensor(data["body_lin_vel_w"])[:, 0], torch.tensor(data["body_ang_vel_w"])[:, 0], torch.tensor(data["joint_vel"])],
    -1,
  )
  kin = fk32(qpos, qvel)
  assert max_err(kin.pos, data["body_pos_w"]) < 1e-5
  assert quat_err(kin.quat, data["body_quat_w"]) < 1e-5
  assert max_err(kin.lin_vel, data["body_lin_vel_w"]) < 1e-4
  assert max_err(kin.ang_vel, data["body_ang_vel_w"]) < 1e-4


def test_fk_gradcheck(fk64, mj):
  m, _ = mj
  qpos, qvel = random_states(m, 2, seed=5)
  qp = torch.tensor(qpos, requires_grad=True)
  qv = torch.tensor(qvel, requires_grad=True)

  def f(qp, qv):
    kin = fk64(qp, qv)
    return kin.pos, kin.quat, kin.lin_vel, kin.ang_vel, kin.site_pos

  assert torch.autograd.gradcheck(f, (qp, qv), eps=1e-6, atol=1e-6)


def test_g1_metadata(fk32):
  assert fk32.body_names == G1_BODY_NAMES and len(G1_BODY_NAMES) == 30
  assert fk32.joint_names == G1_JOINT_NAMES and len(G1_JOINT_NAMES) == 29
  assert fk32.site_names == G1_SITE_NAMES
  assert [G1_BODY_NAMES[i] for i in TRACKED_BODY_IDS] == list(G1_TRACKED_BODY_NAMES)
  assert G1_BODY_NAMES[ANCHOR_BODY_ID] == "torso_link"
  torch.testing.assert_close(fk32.default_joint_pos, torch.tensor(G1_DEFAULT_JOINT_POS))
  # Soft limits: mjlab Entity.initialize formula, in float32.
  m = mujoco.MjModel.from_xml_path(str(g1_xml_path()))
  rng = torch.tensor(m.jnt_range[1:], dtype=torch.float32)
  mean, span = (rng[:, 0] + rng[:, 1]) / 2, rng[:, 1] - rng[:, 0]
  expect = torch.stack([mean - 0.5 * span * 0.9, mean + 0.5 * span * 0.9], -1)
  torch.testing.assert_close(fk32.soft_joint_limits, expect, atol=0, rtol=0)
  torch.testing.assert_close(fk32.joint_range, rng)
  # The IMU site frame is the pelvis frame (gyro = R_pelvisᵀ ω).
  sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "imu_in_pelvis")
  assert m.site_bodyid[sid] == 1 and np.allclose(m.site_quat[sid], [1, 0, 0, 0])


def test_g1_constants_match_mjlab():
  """Hard-coded G1 constants vs mjlab's own configs (imports mjlab, ~5 s)."""
  from mjlab.asset_zoo.robots.unitree_g1 import g1_constants as g1c
  from mjlab.tasks.tracking.config.g1.env_cfgs import unitree_g1_flat_tracking_env_cfg
  from mjlab.utils.string import resolve_expr

  assert g1_xml_path() == Path(g1c.G1_XML)
  q0 = resolve_expr(g1c.KNEES_BENT_KEYFRAME.joint_pos, G1_JOINT_NAMES, 0.0)
  np.testing.assert_allclose(G1_DEFAULT_JOINT_POS, q0)
  assert g1c.G1_ARTICULATION.soft_joint_pos_limit_factor == G1_SOFT_LIMIT_FACTOR
  motion = unitree_g1_flat_tracking_env_cfg().commands["motion"]
  assert tuple(motion.body_names) == G1_TRACKED_BODY_NAMES
  assert motion.anchor_body_name == G1_ANCHOR_BODY_NAME


# ----- geometry -----


def _rand_quat(n: int, seed: int = 0, dtype=torch.float64) -> torch.Tensor:
  g = torch.Generator().manual_seed(seed)
  return geo.quat_normalize(torch.randn(n, 4, generator=g, dtype=dtype))


def test_quat_ops_match_matrices():
  a, b = _rand_quat(16, 0), _rand_quat(16, 1)
  v = torch.randn(16, 3, dtype=torch.float64)
  ra, rb = geo.quat_to_matrix(a), geo.quat_to_matrix(b)
  torch.testing.assert_close(geo.quat_to_matrix(geo.quat_mul(a, b)), ra @ rb)
  torch.testing.assert_close(geo.quat_apply(a, v), (ra @ v[..., None])[..., 0])
  torch.testing.assert_close(geo.quat_apply_inverse(a, v), (ra.transpose(-1, -2) @ v[..., None])[..., 0])
  torch.testing.assert_close(geo.quat_mul(a, geo.quat_conj(a)), torch.tensor([1.0, 0, 0, 0], dtype=torch.float64).expand(16, 4))
  # rot6d follows mjlab's motion_anchor_ori_b flattening: mat[..., :2].reshape(-1, 6).
  torch.testing.assert_close(geo.quat_to_rot6d(a), ra[..., :2].reshape(16, 6))


def test_quat_ops_match_mujoco():
  a, b = _rand_quat(4, 2).numpy(), _rand_quat(4, 3).numpy()
  for i in range(4):
    out = np.zeros(4)
    mujoco.mju_mulQuat(out, a[i], b[i])
    np.testing.assert_allclose(geo.quat_mul(torch.tensor(a[i]), torch.tensor(b[i])).numpy(), out, atol=1e-12)
    axis = np.array([0.3, -0.5, 0.8]) / np.linalg.norm([0.3, -0.5, 0.8])
    mujoco.mju_axisAngle2Quat(out, axis, 0.7)
    np.testing.assert_allclose(
      geo.quat_from_axis_angle(torch.tensor(axis), torch.tensor(0.7)).numpy(), out, atol=1e-12
    )


def test_exp_log_roundtrip_and_small_angles():
  g = torch.Generator().manual_seed(4)
  r = torch.randn(64, 3, generator=g, dtype=torch.float64)
  r = r / r.norm(dim=-1, keepdim=True) * torch.rand(64, 1, generator=g, dtype=torch.float64) * 3.1
  torch.testing.assert_close(geo.quat_log(geo.quat_exp(r)), r)
  # Gradients stay finite at (and near) zero rotation.
  for scale in (0.0, 1e-7, 1e-3):
    x = (torch.ones(2, 3, dtype=torch.float64) * scale).requires_grad_(True)
    y = geo.quat_log(geo.quat_exp(x))
    y.sum().backward()
    assert torch.isfinite(x.grad).all()
    torch.testing.assert_close(x.grad, torch.ones_like(x), atol=1e-6, rtol=0)
  q = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32, requires_grad=True)
  geo.quat_angle_sq(q).sum().backward()
  assert torch.isfinite(q.grad).all()
  q2 = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32, requires_grad=True)
  geo.quat_error_magnitude(q2, q2.detach()).sum().backward()
  assert torch.isfinite(q2.grad).all()
  # Shortest arc: q and -q have the same log.
  qs = _rand_quat(8, 5)
  torch.testing.assert_close(geo.quat_log(qs), geo.quat_log(-qs))
  torch.testing.assert_close(geo.quat_angle(geo.quat_exp(r)), r.norm(dim=-1))


def test_yaw_quat():
  yaw = torch.linspace(-3.0, 3.0, 7, dtype=torch.float64)
  tilt = geo.quat_from_axis_angle(torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64), torch.full_like(yaw, 0.3))
  q = geo.quat_mul(geo.quat_from_yaw(yaw), tilt)
  torch.testing.assert_close(geo.yaw_quat(q), geo.quat_from_yaw(yaw))
  torch.testing.assert_close(geo.yaw_angle(q), yaw)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_log_and_angle_grads_finite_at_180_degrees(dtype):
  """quat_log's unselected Taylor branch used to overflow at w = 0 and leak NaN gradients
  (world_model's orientation loss / integration residual go through quat_log)."""
  for w in (0.0, 1e-20, -1e-20, 1e-7):
    for fn in (geo.quat_log, geo.quat_angle_sq, geo.quat_angle):
      q = torch.tensor([[w, 0.6, 0.0, 0.8]], dtype=dtype, requires_grad=True)
      fn(q).sum().backward()
      assert torch.isfinite(q.grad).all(), (fn.__name__, w, q.grad)
  q = torch.tensor([[0.0, 0.6, 0.0, 0.8]], dtype=dtype)
  torch.testing.assert_close(geo.quat_angle_sq(q), torch.tensor([np.pi**2], dtype=dtype))


def test_log_exp_gradcheck():
  g = torch.Generator().manual_seed(7)
  r = torch.randn(6, 3, generator=g, dtype=torch.float64)
  r = r / r.norm(dim=-1, keepdim=True) * torch.tensor([1e-6, 1e-3, 0.3, 1.0, 2.0, 3.0], dtype=torch.float64)[:, None]
  assert torch.autograd.gradcheck(lambda x: geo.quat_log(geo.quat_exp(x)), (r.requires_grad_(True),))
  q = _rand_quat(6, 8).requires_grad_(True)
  assert torch.autograd.gradcheck(geo.quat_log, (q,))
  assert torch.autograd.gradcheck(geo.yaw_quat, (q,))


def test_yaw_matches_mjlab_and_is_safe_at_gimbal_lock():
  from mjlab.utils.lab_api.math import yaw_quat as mj_yaw_quat

  q = _rand_quat(64, 9, dtype=torch.float32)
  torch.testing.assert_close(geo.yaw_quat(q), mj_yaw_quat(q), atol=1e-6, rtol=0)
  # Pitch ±90° (heading undefined): same forward value as mjlab, finite (zero) gradient.
  for dtype in (torch.float32, torch.float64):
    for pitch in (np.pi / 2, -np.pi / 2):
      lock = geo.quat_mul(
        geo.quat_from_yaw(torch.tensor(0.7, dtype=dtype)),
        geo.quat_from_axis_angle(torch.tensor([0.0, 1.0, 0.0], dtype=dtype), torch.tensor(pitch, dtype=dtype)),
      )
      exact = torch.tensor([np.cos(np.pi / 4), 0.0, np.sin(pitch / 2), 0.0], dtype=dtype)
      for qq in (lock, exact):
        qq = qq.clone().requires_grad_(True)
        y = geo.yaw_quat(qq)
        torch.testing.assert_close(y.detach(), mj_yaw_quat(qq.detach()[None])[0], atol=1e-6, rtol=0)
        y.sum().backward()
        assert torch.isfinite(qq.grad).all() and qq.grad.abs().max() < 1e5
