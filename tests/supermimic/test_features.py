"""features.py / reference.py on CPU: dims, invariances, mujoco obs parity, goal, history."""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
import pytest
import torch

from mjlab_g1.rl.supermimic import features as F
from mjlab_g1.rl.supermimic import geometry as geo
from mjlab_g1.rl.supermimic.kinematics import (
  G1_DEFAULT_JOINT_POS,
  TorchFK,
  g1_xml_path,
)
from mjlab_g1.rl.supermimic.reference import MotionLibrary

REPO = Path(__file__).resolve().parents[2]
WALK_NPZ = REPO / "dataset" / "tracking" / "locomotion1.npz"
f64 = torch.float64


@pytest.fixture(scope="module")
def fk():
  return TorchFK.g1(dtype=f64)


@pytest.fixture(scope="module")
def walk_lib():
  if not WALK_NPZ.exists():
    pytest.skip("dataset/tracking/locomotion1.npz not present")
  return MotionLibrary(WALK_NPZ)


def random_state(n: int, has_box: bool, seed: int = 0, dtype=f64) -> torch.Tensor:
  """Random packed state, our convention (world ω), joints inside the hard limits."""
  g = torch.Generator().manual_seed(seed)
  m = mujoco.MjModel.from_xml_path(str(g1_xml_path()))
  lo = torch.tensor(m.jnt_range[1:, 0], dtype=dtype)
  hi = torch.tensor(m.jnt_range[1:, 1], dtype=dtype)
  qpos = torch.zeros(n, 36, dtype=dtype)
  qpos[:, :2] = torch.rand(n, 2, generator=g, dtype=dtype) * 6 - 3
  qpos[:, 2] = 0.5 + 0.4 * torch.rand(n, generator=g, dtype=dtype)
  qpos[:, 3:7] = geo.quat_normalize(torch.randn(n, 4, generator=g, dtype=dtype))
  qpos[:, 7:] = lo + (hi - lo) * torch.rand(n, 29, generator=g, dtype=dtype)
  qvel = torch.randn(n, 35, generator=g, dtype=dtype)
  if not has_box:
    return F.pack_state(qpos, qvel)
  bq = torch.zeros(n, 7, dtype=dtype)
  bq[:, :3] = qpos[:, :3] + torch.randn(n, 3, generator=g, dtype=dtype) * 0.3
  bq[:, 3:] = geo.quat_normalize(torch.randn(n, 4, generator=g, dtype=dtype))
  bv = torch.randn(n, 6, generator=g, dtype=dtype)
  return F.pack_state(qpos, qvel, bq, bv)


def write_box_npz(path: Path, src: Path, partial: bool = False) -> Path:
  """Walk clip + a synthetic box 0.4 m ahead / 0.3 m below the pelvis in its heading frame."""
  d = dict(np.load(src))
  pelvis_p = torch.tensor(d["body_pos_w"][:, 0], dtype=f64)
  pelvis_q = torch.tensor(d["body_quat_w"][:, 0], dtype=f64)
  yaw = geo.yaw_quat(pelvis_q)
  off = torch.tensor([0.4, 0.0, -0.3], dtype=f64).expand_as(pelvis_p)
  d["box_pos_w"] = (pelvis_p + geo.quat_apply(yaw, off)).float().numpy()
  d["box_quat_w"] = pelvis_q.float().numpy()
  if not partial:
    d["box_lin_vel_w"] = np.zeros_like(d["box_pos_w"])
    d["box_ang_vel_w"] = np.zeros_like(d["box_pos_w"])
  d["grasp_frame"] = np.array(57)
  np.savez(path, **d)
  return path


def ref_state(lib: MotionLibrary, k: torch.Tensor, has_box: bool = False) -> torch.Tensor:
  """Packed state equal to the reference at frames k (root = pelvis = body 0)."""
  r = lib.frame(k)
  qpos = torch.cat([r.body_pos_w[:, 0], r.body_quat_w[:, 0], r.joint_pos], -1)
  qvel = torch.cat([r.body_lin_vel_w[:, 0], r.body_ang_vel_w[:, 0], r.joint_vel], -1)
  if not has_box:
    return F.pack_state(qpos, qvel).double()
  return F.pack_state(
    qpos,
    qvel,
    torch.cat([r.box_pos_w, r.box_quat_w], -1),
    torch.cat([r.box_lin_vel_w, r.box_ang_vel_w], -1),
  ).double()


# ----- dims and packing -----


@pytest.mark.parametrize("has_box", [False, True])
def test_feature_shapes(fk, walk_lib, tmp_path, has_box):
  n = 5
  lib = MotionLibrary(write_box_npz(tmp_path / "box.npz", WALK_NPZ)) if has_box else walk_lib
  s = random_state(n, has_box)
  kin = F.forward_kin(fk, s)
  xi = torch.zeros(n, F.XI_DIM[has_box], dtype=f64)
  assert s.shape[-1] == (84 if has_box else 71)
  assert F.wm_features(s, kin, has_box).shape == (n, 191 if has_box else 169)
  frame = F.obs_frame(s, kin, torch.zeros(n, 29, dtype=f64), xi, has_box)
  assert frame.shape == (n, 102 if has_box else 93) == (n, F.obs_frame_dim(has_box))
  hist = F.PolicyObsHistory(n, 10, frame.shape[-1], dtype=f64)
  hist.reset(None, frame)
  assert hist.get().shape == (n, 1020 if has_box else 930)
  g = F.goal(s, kin, lib, torch.zeros(n, dtype=torch.long), has_box=has_box)
  assert g.shape == (n, 730 if has_box else 640) == (n, F.goal_dim(has_box))
  assert F.encode_xi(xi, xi[0] - 1, xi[0] + 1).shape == (n, 35 if has_box else 33)
  assert F.XI_DIM[has_box] == (35 if has_box else 33)


def test_pack_unpack_roundtrip():
  s = random_state(3, True)
  st = F.unpack_state(s)
  assert st.box_qpos is not None
  torch.testing.assert_close(F.pack_state(*st), s)
  torch.testing.assert_close(st.box_quat, s[:, 74:78])
  torch.testing.assert_close(st.root_ang_vel, s[:, 39:42])
  w = random_state(3, False)
  st = F.unpack_state(w)
  assert st.box_qpos is None and st.joint_vel.shape == (3, 29)
  with pytest.raises(ValueError):
    F.unpack_state(w, has_box=True)


# ----- world-model features -----


def yaw_transform(s: torch.Tensor, yaw: float, shift_xy: tuple[float, float], has_box: bool) -> torch.Tensor:
  """Rotate the whole scene about world z and translate it in xy."""
  st = F.unpack_state(s, has_box)
  qz = geo.quat_from_yaw(torch.full(s.shape[:-1], yaw, dtype=s.dtype))
  t = torch.tensor([*shift_xy, 0.0], dtype=s.dtype)
  qpos = torch.cat([geo.quat_apply(qz, st.root_pos) + t, geo.quat_mul(qz, st.root_quat), st.joint_pos], -1)
  qvel = torch.cat([geo.quat_apply(qz, st.root_lin_vel), geo.quat_apply(qz, st.root_ang_vel), st.joint_vel], -1)
  if not has_box:
    return F.pack_state(qpos, qvel)
  bq = torch.cat([geo.quat_apply(qz, st.box_pos) + t, geo.quat_mul(qz, st.box_quat)], -1)
  bv = torch.cat([geo.quat_apply(qz, st.box_lin_vel), geo.quat_apply(qz, st.box_ang_vel)], -1)
  return F.pack_state(qpos, qvel, bq, bv)


@pytest.mark.parametrize("has_box", [False, True])
def test_wm_features_heading_invariant(fk, has_box):
  s = random_state(16, has_box, seed=1)
  phi = F.wm_features(s, F.forward_kin(fk, s), has_box)
  s2 = yaw_transform(s, 1.234, (2.5, -4.0), has_box)
  phi2 = F.wm_features(s2, F.forward_kin(fk, s2), has_box)
  torch.testing.assert_close(phi2, phi, atol=1e-9, rtol=0)
  # Sanity: the features do see yaw-independent changes (pitch) and joint changes.
  s3 = s.clone()
  s3[:, 7] += 0.1
  assert (F.wm_features(s3, F.forward_kin(fk, s3), has_box) - phi).abs().max() > 1e-3


def test_wm_features_layout(fk):
  s = random_state(4, False, seed=2)
  kin = F.forward_kin(fk, s)
  phi = F.wm_features(s, kin, False)
  st = F.unpack_state(s)
  torch.testing.assert_close(phi[:, 0], st.root_pos[:, 2])
  torch.testing.assert_close(phi[:, 13:42], st.joint_pos - torch.tensor(G1_DEFAULT_JOINT_POS, dtype=f64))
  torch.testing.assert_close(phi[:, 42:71], st.joint_vel)
  torch.testing.assert_close(phi[:, 10:13], geo.quat_apply_inverse(st.root_quat, st.root_ang_vel))
  torch.testing.assert_close(phi[:, 71:74], torch.zeros(4, 3, dtype=f64), atol=1e-12, rtol=0)  # pelvis rel pos
  torch.testing.assert_close(phi[:, -2:], kin.pos[:, [6, 12], 2])  # ankle_roll heights


# ----- policy observation frame -----


def test_obs_frame_matches_mujoco_sensors(fk):
  """Gyro = MuJoCo imu_ang_vel sensor, gravity = R_pelvisᵀ[0,0,-1], biased joint pos."""
  m = mujoco.MjModel.from_xml_path(str(g1_xml_path()))
  d = mujoco.MjData(m)
  s = random_state(8, False, seed=3)
  g = torch.Generator().manual_seed(0)
  xi = torch.rand(8, 33, generator=g, dtype=f64) * 0.02 - 0.01
  a_prev = torch.randn(8, 29, generator=g, dtype=f64)
  frame = F.obs_frame(s, None, a_prev, xi, False)
  adr = m.sensor_adr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, "imu_ang_vel")]
  for i in range(8):
    qpos = s[i, :36].numpy()
    R = geo.quat_to_matrix(s[i, 3:7]).numpy()
    qvel_mj = s[i, 36:71].numpy().copy()
    qvel_mj[3:6] = R.T @ qvel_mj[3:6]  # world ω -> MuJoCo's body-local free-joint ω
    d.qpos[:] = qpos
    d.qvel[:] = qvel_mj
    mujoco.mj_forward(m, d)
    gyro = d.sensordata[adr : adr + 3]
    grav = d.xmat[1].reshape(3, 3).T @ np.array([0.0, 0.0, -1.0])
    np.testing.assert_allclose(frame[i, 58:61].numpy(), gyro, atol=1e-10)
    np.testing.assert_allclose(frame[i, 58:61].numpy(), qvel_mj[3:6], atol=1e-10)
    np.testing.assert_allclose(frame[i, 61:64].numpy(), grav, atol=1e-10)
  q0 = torch.tensor(G1_DEFAULT_JOINT_POS, dtype=f64)
  torch.testing.assert_close(frame[:, :29], s[:, 7:36] - q0 + xi[:, 1:30])
  torch.testing.assert_close(frame[:, 29:58], s[:, 42:71])
  torch.testing.assert_close(frame[:, 64:93], a_prev)


def test_obs_frame_hand_values():
  """Upright pelvis yawed 90°: gravity_b = (0,0,-1); world ω_x shows up as body −y."""
  s = torch.zeros(1, 71, dtype=f64)
  s[0, 3:7] = geo.quat_from_yaw(torch.tensor(np.pi / 2, dtype=f64))
  s[0, 7:36] = torch.tensor(G1_DEFAULT_JOINT_POS, dtype=f64)
  s[0, 39:42] = torch.tensor([0.7, 0.0, 0.2], dtype=f64)
  frame = F.obs_frame(s, None, torch.zeros(1, 29, dtype=f64), torch.zeros(1, 33, dtype=f64), False)
  torch.testing.assert_close(frame[0, 61:64], torch.tensor([0.0, 0.0, -1.0], dtype=f64), atol=1e-12, rtol=0)
  torch.testing.assert_close(frame[0, 58:61], torch.tensor([0.0, -0.7, 0.2], dtype=f64), atol=1e-12, rtol=0)
  torch.testing.assert_close(frame[0, :29], torch.zeros(29, dtype=f64))
  # Pitched forward 90° (nose down): gravity points along +x of the body.
  s[0, 3:7] = geo.quat_from_axis_angle(torch.tensor([0.0, 1.0, 0.0], dtype=f64), torch.tensor(np.pi / 2, dtype=f64))
  frame = F.obs_frame(s, None, torch.zeros(1, 29, dtype=f64), torch.zeros(1, 33, dtype=f64), False)
  torch.testing.assert_close(frame[0, 61:64], torch.tensor([1.0, 0.0, 0.0], dtype=f64), atol=1e-12, rtol=0)


def test_obs_frame_box_terms():
  s = random_state(4, True, seed=4)
  st = F.unpack_state(s)
  frame = F.obs_frame(s, None, torch.zeros(4, 29, dtype=f64), torch.zeros(4, 35, dtype=f64), True)
  R = geo.quat_to_matrix(st.root_quat)
  Rb = geo.quat_to_matrix(st.box_quat)
  expect_pos = (R.transpose(-1, -2) @ (st.box_pos - st.root_pos)[..., None])[..., 0]
  torch.testing.assert_close(frame[:, 93:96], expect_pos)
  torch.testing.assert_close(frame[:, 96:102], (R.transpose(-1, -2) @ Rb)[..., :2].reshape(4, 6))


def test_obs_noise():
  for has_box in (False, True):
    scale = F.obs_noise_scale(has_box)
    assert scale.shape == (F.obs_frame_dim(has_box),)
    eta = F.sample_obs_noise((4000,), has_box, generator=torch.Generator().manual_seed(0))
    assert (eta.abs() <= scale + 1e-7).all()
    torch.testing.assert_close(eta.abs().amax(0), scale, atol=0.01 * scale.max().item(), rtol=0.01)
    assert (eta[:, 64:93] == 0).all()  # last_action is noise-free
  assert F.obs_noise_scale(True)[93:96].eq(0.01).all() and F.obs_noise_scale(True)[96:].eq(0.02).all()


# ----- goal -----


def test_goal_at_reference(fk, walk_lib):
  lib = walk_lib
  k = torch.tensor([0, 37, lib.clip_last - 3])
  s = ref_state(lib, k)
  kin = F.forward_kin(fk, s)
  g = F.goal(s, kin, lib, k).reshape(3, 10, 64)
  # Robot on the reference: frame 0's anchor term is the identity rot6d.
  ident = torch.tensor([1.0, 0.0, 0.0, 1.0, 0.0, 0.0], dtype=f64)
  torch.testing.assert_close(g[:, 0, 58:64], ident.expand(3, 6), atol=1e-6, rtol=0)
  for j in range(10):
    kk = (k + j).clamp(max=lib.clip_last)
    torch.testing.assert_close(g[:, j, :29], lib.joint_pos[kk].double())
    torch.testing.assert_close(g[:, j, 29:58], lib.joint_vel[kk].double())
  # Clamped at the clip end: frames 3..9 of the last row all repeat the final frame.
  assert (g[2, 3:] == g[2, 3:4]).all()
  # offset shifts the window.
  g1 = F.goal(s, kin, lib, k, offset=1).reshape(3, 10, 64)
  torch.testing.assert_close(g1[:2, :9, :58], g[:2, 1:, :58])


def test_goal_anchor_term_matches_mjlab_formula(fk, walk_lib):
  """rot6d(R_robot_anchorᵀ R_ref_anchor) = mjlab motion_anchor_ori_b for frame k."""
  s = random_state(6, False, seed=5)
  kin = F.forward_kin(fk, s)
  k = torch.tensor([0, 10, 20, 30, 40, 50])
  g = F.goal(s, kin, walk_lib, k).reshape(6, 10, 64)
  robot = geo.quat_to_matrix(kin.quat[:, 15])  # torso_link
  ref = geo.quat_to_matrix(walk_lib.body_quat_w[k, 15].double())
  torch.testing.assert_close(g[:, 0, 58:64], (robot.transpose(-1, -2) @ ref)[..., :2].reshape(6, 6))


def test_goal_box_terms(fk, tmp_path):
  if not WALK_NPZ.exists():
    pytest.skip("no walk npz")
  lib = MotionLibrary(write_box_npz(tmp_path / "box.npz", WALK_NPZ))
  assert lib.has_box and lib.grasp_frame == 57
  k = torch.tensor([5, 80])
  s = ref_state(lib, k, has_box=True)
  g = F.goal(s, F.forward_kin(fk, s), lib, k, has_box=True).reshape(2, 10, 73)
  torch.testing.assert_close(g[:, 0, 64:67], torch.tensor([0.4, 0.0, -0.3], dtype=f64).expand(2, 3), atol=1e-6, rtol=0)
  ident = torch.tensor([1.0, 0.0, 0.0, 1.0, 0.0, 0.0], dtype=f64)
  torch.testing.assert_close(g[:, 0, 67:73], ident.expand(2, 6), atol=1e-6, rtol=0)
  with pytest.raises(ValueError):
    walk = MotionLibrary(WALK_NPZ)
    F.goal(s, F.forward_kin(fk, s), walk, k, has_box=True)


# ----- gradients -----


@pytest.mark.parametrize("has_box", [False, True])
def test_features_differentiable(fk, tmp_path, has_box):
  if not WALK_NPZ.exists():
    pytest.skip("no walk npz")
  lib = MotionLibrary(write_box_npz(tmp_path / "box.npz", WALK_NPZ))
  s = random_state(3, has_box, seed=6).requires_grad_(True)
  kin = F.forward_kin(fk, s)
  a = torch.randn(3, 29, dtype=f64, generator=torch.Generator().manual_seed(1)).requires_grad_(True)
  xi = torch.zeros(3, F.XI_DIM[has_box], dtype=f64)
  hist = F.PolicyObsHistory(3, 10, F.obs_frame_dim(has_box), dtype=f64)
  hist.reset(None, F.obs_frame(s, kin, a, xi, has_box).detach())
  hist.push(F.obs_frame(s, kin, a, xi, has_box))  # history stays on the graph
  out = (
    F.wm_features(s, kin, has_box).square().sum()
    + hist.get().square().sum()
    + F.goal(s, kin, lib, torch.tensor([0, 5, 9]), has_box=has_box).square().sum()
  )
  out.backward()
  assert torch.isfinite(s.grad).all() and s.grad.abs().sum() > 0
  assert a.grad is not None and a.grad.abs().sum() > 0


# ----- xi encoding -----


def test_encode_xi():
  lo = torch.tensor([0.3, -0.01, 0.0, 1.0])
  hi = torch.tensor([1.2, 0.01, 0.0, 1.0])
  xi = torch.stack([lo, hi, (lo + hi) / 2])
  enc = F.encode_xi(xi, lo, hi)
  torch.testing.assert_close(enc[:, :2], torch.tensor([[-1.0, -1.0], [1.0, 1.0], [0.0, 0.0]]))
  assert (enc[:, 2:] == 0).all()
  x = xi.clone().requires_grad_(True)
  F.encode_xi(x, lo, hi).sum().backward()
  assert torch.isfinite(x.grad).all()


# ----- observation history -----


def test_history_reset_push_order():
  h = F.PolicyObsHistory(3, history=4, frame_dim=93, layout="frame")
  first = torch.arange(3, dtype=torch.float32)[:, None].expand(3, 93)
  h.reset(None, first)
  assert (h.frames() == first[:, None]).all()  # every slot filled at reset
  for t in range(1, 3):
    h.push(first + 10 * t)
  fr = h.frames()[:, :, 0]
  torch.testing.assert_close(fr[1], torch.tensor([1.0, 1.0, 11.0, 21.0]))  # oldest -> newest
  h.reset(torch.tensor([2]), torch.full((1, 93), -5.0))
  assert (h.frames()[2] == -5).all() and (h.frames()[0, -1] == 20).all()
  h.reset(torch.tensor([True, False, False]), torch.full((3, 93), 7.0))  # bool mask + full rows
  assert (h.frames()[0] == 7).all() and (h.frames()[1, -1] == 21).all()
  torch.testing.assert_close(h.get(), h.frames().reshape(3, -1))


@pytest.mark.parametrize("has_box", [False, True])
def test_history_matches_mjlab_layout(has_box):
  """get() equals mjlab's ObservationManager output: per-term CircularBuffers, flattened, cat."""
  from mjlab.utils.buffers.circular_buffer import CircularBuffer

  n, hl = 4, 10
  dims = F.obs_term_dims(has_box)
  fdim = sum(dims)
  ours = F.PolicyObsHistory(n, hl, fdim)
  bufs = [CircularBuffer(hl, n, "cpu") for _ in dims]
  g = torch.Generator().manual_seed(0)

  def mjlab_obs() -> torch.Tensor:
    return torch.cat([b.buffer.reshape(n, -1) for b in bufs], dim=-1)

  def append(frame: torch.Tensor) -> None:
    for b, chunk in zip(bufs, torch.split(frame, list(dims), dim=-1), strict=True):
      b.append(chunk)

  frame = torch.randn(n, fdim, generator=g)
  append(frame)
  ours.reset(None, frame)
  torch.testing.assert_close(ours.get(), mjlab_obs())
  for t in range(14):
    frame = torch.randn(n, fdim, generator=g)
    if t == 6:  # env 1 resets: mjlab zeroes its rows, the next append backfills them
      for b in bufs:
        b.reset(batch_ids=[1])
      append(frame)
      ours.push(frame)
      ours.reset(torch.tensor([1]), frame[1:2])
    else:
      append(frame)
      ours.push(frame)
    torch.testing.assert_close(ours.get(), mjlab_obs())
  torch.testing.assert_close(F.unflatten_history(ours.get(), hl), ours.frames())
  torch.testing.assert_close(
    F.unflatten_history(F.flatten_history(ours.frames(), layout="frame"), hl, layout="frame"), ours.frames()
  )


def test_history_from_frames():
  frames = torch.randn(2, 10, 102)
  h = F.PolicyObsHistory.from_frames(frames)
  assert h.term_dims == F.obs_term_dims(True)
  torch.testing.assert_close(h.get()[:, :290], frames[:, :, :29].reshape(2, -1))


# ----- motion library -----


def test_motion_library_walk(walk_lib):
  lib = walk_lib
  assert lib.num_frames == 199 and lib.clip_last == 198 and lib.num_bodies == 30
  assert not lib.has_box and lib.grasp_frame is None and lib.box_pos_w is None
  r = lib.frame(torch.tensor([[0, 198, 250]]))
  assert r.joint_pos.shape == (1, 3, 29) and r.body_quat_w.shape == (1, 3, 30, 4)
  torch.testing.assert_close(r.joint_pos[0, 2], lib.joint_pos[198])
  assert lib.alive(torch.tensor([0, 198, 199])).tolist() == [True, True, False]
  torch.testing.assert_close(lib.t_left(torch.tensor([0, 150, 198, 210])), torch.tensor([1.0, 48 / 64, 0.0, 0.0]))
  sub = lib.frame(5, body_ids=torch.tensor([15, 0]))
  torch.testing.assert_close(sub.body_pos_w, lib.body_pos_w[5, [15, 0]])


def test_motion_library_box_keys(tmp_path):
  if not WALK_NPZ.exists():
    pytest.skip("no walk npz")
  lib = MotionLibrary(write_box_npz(tmp_path / "box.npz", WALK_NPZ))
  r = lib.frame(torch.tensor([3, 500]))
  assert r.box_pos_w.shape == (2, 3) and r.box_quat_w.shape == (2, 4)
  torch.testing.assert_close(r.box_pos_w[1], lib.box_pos_w[-1])
  with pytest.raises(KeyError):
    MotionLibrary(write_box_npz(tmp_path / "partial.npz", WALK_NPZ, partial=True))


# ----- adversarial-review additions -----

BOX_NPZ = REPO / "dataset" / "tracking" / "dual_arm1_box.npz"


@pytest.mark.parametrize("has_box", [False, True])
def test_features_gradcheck(fk, tmp_path, has_box):
  """float64 gradcheck of φ, the obs frame and the goal w.r.t. the raw state (critique T2)."""
  if not WALK_NPZ.exists():
    pytest.skip("no walk npz")
  lib = MotionLibrary(write_box_npz(tmp_path / "box.npz", WALK_NPZ))
  s = random_state(2, has_box, seed=11).requires_grad_(True)
  a = torch.randn(2, 29, dtype=f64, generator=torch.Generator().manual_seed(2)).requires_grad_(True)
  xi = torch.rand(2, F.XI_DIM[has_box], dtype=f64, generator=torch.Generator().manual_seed(3)) * 0.02
  k = torch.tensor([3, 150])

  def fn(s, a):
    kin = F.forward_kin(fk, s)
    return (
      F.wm_features(s, kin, has_box),
      F.obs_frame(s, kin, a, xi, has_box),
      F.goal(s, kin, lib, k, has_box=has_box),
    )

  assert torch.autograd.gradcheck(fn, (s, a), eps=1e-6, atol=1e-5)


def _flip_quats(s: torch.Tensor, has_box: bool) -> torch.Tensor:
  out = s.clone()
  out[:, 3:7] = -out[:, 3:7]
  if has_box:
    out[:, 74:78] = -out[:, 74:78]
  return out


@pytest.mark.parametrize("has_box", [False, True])
def test_quaternion_sign_invariance(fk, tmp_path, has_box):
  """q and −q are the same rotation: φ, obs and goal must not depend on the sign."""
  if not WALK_NPZ.exists():
    pytest.skip("no walk npz")
  lib = MotionLibrary(write_box_npz(tmp_path / "box.npz", WALK_NPZ))
  s = random_state(8, has_box, seed=12)
  s2 = _flip_quats(s, has_box)
  xi = torch.zeros(8, F.XI_DIM[has_box], dtype=f64)
  a = torch.zeros(8, 29, dtype=f64)
  k = torch.arange(8) * 20
  kin, kin2 = F.forward_kin(fk, s), F.forward_kin(fk, s2)
  torch.testing.assert_close(kin2.pos, kin.pos)
  torch.testing.assert_close(F.wm_features(s2, kin2, has_box), F.wm_features(s, kin, has_box))
  torch.testing.assert_close(F.obs_frame(s2, kin2, a, xi, has_box), F.obs_frame(s, kin, a, xi, has_box))
  torch.testing.assert_close(
    F.goal(s2, kin2, lib, k, has_box=has_box), F.goal(s, kin, lib, k, has_box=has_box)
  )


def test_goal_rejects_tracked_body_only_library(fk, tmp_path):
  """ANCHOR_BODY_ID indexes all 30 bodies; a 14-body npz must fail loudly, not mis-index."""
  if not WALK_NPZ.exists():
    pytest.skip("no walk npz")
  d = dict(np.load(WALK_NPZ))
  for key in ("body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w"):
    d[key] = d[key][:, :14]
  np.savez(tmp_path / "b14.npz", **d)
  lib = MotionLibrary(tmp_path / "b14.npz")
  s = random_state(2, False)
  with pytest.raises(ValueError, match="bodies"):
    F.goal(s, F.forward_kin(fk, s), lib, torch.tensor([0, 1]))


def test_lift_features_reject_fk_without_palms():
  fk_imu = TorchFK.from_xml(g1_xml_path(), site_names=("imu_in_pelvis",), dtype=f64)
  s = random_state(2, True)
  with pytest.raises(ValueError, match="sites"):
    F.wm_features(s, F.forward_kin(fk_imu, s), True)
  F.wm_features(s[:, :71], F.forward_kin(fk_imu, s[:, :71]), False)  # walk needs no palms


def test_real_box_reference(fk):
  """The generated lift npz (scripts/make_box_reference.py) loads and gives sane lift features."""
  if not BOX_NPZ.exists():
    pytest.skip("dataset/tracking/dual_arm1_box.npz not present")
  lib = MotionLibrary(BOX_NPZ)
  assert lib.has_box and lib.num_bodies == 30 and lib.num_frames == 520
  assert lib.grasp_frame is not None and 0 < lib.grasp_frame < lib.num_frames // 2
  torch.testing.assert_close(lib.box_quat_w.norm(dim=-1), torch.ones(lib.num_frames), atol=1e-5, rtol=0)
  # Box rests on the floor (at its half height) before the grasp and rises afterwards.
  half_z = float(np.load(BOX_NPZ)["box_half_extents"][2])
  assert torch.allclose(lib.box_pos_w[: lib.grasp_frame, 2], torch.tensor(half_z), atol=1e-4)
  assert lib.box_pos_w[-1, 2] > 0.5
  k = torch.arange(lib.num_frames)
  s = ref_state(lib, k, has_box=True)
  kin = F.forward_kin(fk, s)
  # The npz body arrays are the FK of its own root/joint trajectory (same 30-body order).
  assert (kin.pos - lib.body_pos_w.double()).abs().max() < 1e-5
  assert (kin.lin_vel - lib.body_lin_vel_w.double()).abs().max() < 1e-4
  phi = F.wm_features(s, kin, True)
  g = F.goal(s, kin, lib, k, has_box=True).reshape(-1, 10, 73)
  frame = F.obs_frame(s, kin, torch.zeros(lib.num_frames, 29, dtype=f64), torch.zeros(lib.num_frames, 35, dtype=f64), True)
  assert torch.isfinite(phi).all() and torch.isfinite(g).all() and torch.isfinite(frame).all()
  # On the reference, goal frame 0 = the box seen from the robot: heading-frame offset and
  # the same relative orientation the obs frame reports in the pelvis frame.
  st = F.unpack_state(s, True)
  yaw = geo.yaw_quat(st.root_quat)
  torch.testing.assert_close(g[:, 0, 64:67], geo.quat_apply_inverse(yaw, st.box_pos - st.root_pos))
  torch.testing.assert_close(g[:, 0, 67:73], frame[:, 96:102])
  ident = torch.tensor([1.0, 0.0, 0.0, 1.0, 0.0, 0.0], dtype=f64)
  torch.testing.assert_close(g[:, 0, 58:64], ident.expand(lib.num_frames, 6), atol=1e-5, rtol=0)
  # Palms hold the box across its y faces after the grasp (left at +y, right at -y).
  palms_y = phi[lib.grasp_frame :, -6:].reshape(-1, 2, 3)[..., 1]
  assert (palms_y[:, 0] > 0).all() and (palms_y[:, 1] < 0).all()
