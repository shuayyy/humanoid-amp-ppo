"""TrackingReward / term_errors vs mjlab's own reward, re-anchoring and termination functions.

mjlab's functions run on fake command/env namespaces holding the same tensors, so the analytic
reward is compared to the env's math itself rather than to a re-derivation. CPU, float64.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch

from mjlab_g1.rl.supermimic.geometry import quat_apply, quat_mul, quat_normalize
from mjlab_g1.rl.supermimic.reward import (
  BOX_TERMS,
  G1_BODY_NAMES,
  G1_TRACKED_BODY_NAMES,
  TRACKING_TERMS,
  TrackingReward,
  TrackingRewardCfg,
  quat_error_magnitude,
  soft_joint_limits_from_xml,
)

mj_math = pytest.importorskip("mjlab.utils.lab_api.math")
trk_rewards = pytest.importorskip("mjlab.tasks.tracking.mdp.rewards")
trk_terms = pytest.importorskip("mjlab.tasks.tracking.mdp.terminations")
env_rewards = pytest.importorskip("mjlab.envs.mdp.rewards")
from mjlab.tasks.tracking.mdp.commands import MotionCommand  # noqa: E402

DT = 0.02
N = 64
DTYPE = torch.float64
TRK = [G1_BODY_NAMES.index(n) for n in G1_TRACKED_BODY_NAMES]
ANCHOR = G1_BODY_NAMES.index("torso_link")


def _rand_quat(*shape: int, gen: torch.Generator) -> torch.Tensor:
  return quat_normalize(torch.randn(*shape, 4, generator=gen, dtype=DTYPE))


def _small_rot(*shape: int, scale: float, gen: torch.Generator) -> torch.Tensor:
  v = torch.randn(*shape, 3, generator=gen, dtype=DTYPE) * scale
  return quat_normalize(torch.cat((torch.ones(*shape, 1, dtype=DTYPE), 0.5 * v), dim=-1))


def _ref(gen: torch.Generator, box: bool = False) -> SimpleNamespace:
  ref = SimpleNamespace(
    body_pos_w=torch.randn(N, 30, 3, generator=gen, dtype=DTYPE) * 0.5 + torch.tensor([0, 0, 0.8], dtype=DTYPE),
    body_quat_w=_rand_quat(N, 30, gen=gen),
    body_lin_vel_w=torch.randn(N, 30, 3, generator=gen, dtype=DTYPE),
    body_ang_vel_w=torch.randn(N, 30, 3, generator=gen, dtype=DTYPE) * 2,
  )
  if box:
    ref.box_pos_w = torch.randn(N, 3, generator=gen, dtype=DTYPE) * 0.3
    ref.box_quat_w = _rand_quat(N, gen=gen)
  return ref


def _kin_near(ref: SimpleNamespace, gen: torch.Generator, scale: float = 0.1) -> SimpleNamespace:
  """Robot kinematics near the reference so every exp term is in its informative range."""
  return SimpleNamespace(
    pos=ref.body_pos_w + scale * torch.randn(N, 30, 3, generator=gen, dtype=DTYPE),
    quat=quat_mul(_small_rot(N, 30, scale=3 * scale, gen=gen), ref.body_quat_w),
    lin_vel=ref.body_lin_vel_w + 5 * scale * torch.randn(N, 30, 3, generator=gen, dtype=DTYPE),
    ang_vel=ref.body_ang_vel_w + 10 * scale * torch.randn(N, 30, 3, generator=gen, dtype=DTYPE),
  )


def _state(limits: torch.Tensor, gen: torch.Generator, box: bool, ref: SimpleNamespace | None = None) -> torch.Tensor:
  lo, hi = limits[:, 0].to(DTYPE), limits[:, 1].to(DTYPE)
  mid, half = (lo + hi) / 2, (hi - lo) / 2
  q = mid + half * 1.15 * (2 * torch.rand(N, 29, generator=gen, dtype=DTYPE) - 1)  # some violate
  s = torch.zeros(N, 84 if box else 71, dtype=DTYPE)
  s[:, 3] = 1.0
  s[:, 7:36] = q
  if box:
    s[:, 71:74] = ref.box_pos_w + 0.1 * torch.randn(N, 3, generator=gen, dtype=DTYPE)
    s[:, 74:78] = quat_mul(_small_rot(N, scale=0.3, gen=gen), ref.box_quat_w)
  return s


def _mjlab_terms(ref, kin_next, kin_prev, s_next, a, a_prev, limits):
  """Run mjlab's reward functions exactly as the env would at this step."""
  cmd = SimpleNamespace(
    cfg=SimpleNamespace(body_names=G1_TRACKED_BODY_NAMES),
    body_pos_w=ref.body_pos_w[:, TRK],
    body_quat_w=ref.body_quat_w[:, TRK],
    body_lin_vel_w=ref.body_lin_vel_w[:, TRK],
    body_ang_vel_w=ref.body_ang_vel_w[:, TRK],
    anchor_pos_w=ref.body_pos_w[:, ANCHOR],
    anchor_quat_w=ref.body_quat_w[:, ANCHOR],
    # update_relative_body_poses runs at the end of the PREVIOUS step, on the previous state.
    robot_anchor_pos_w=kin_prev.pos[:, ANCHOR],
    robot_anchor_quat_w=kin_prev.quat[:, ANCHOR],
  )
  MotionCommand.update_relative_body_poses(cmd)
  cmd.robot_anchor_pos_w = kin_next.pos[:, ANCHOR]
  cmd.robot_anchor_quat_w = kin_next.quat[:, ANCHOR]
  cmd.robot_body_pos_w = kin_next.pos[:, TRK]
  cmd.robot_body_quat_w = kin_next.quat[:, TRK]
  cmd.robot_body_lin_vel_w = kin_next.lin_vel[:, TRK]
  cmd.robot_body_ang_vel_w = kin_next.ang_vel[:, TRK]
  robot = SimpleNamespace(
    data=SimpleNamespace(
      joint_pos=s_next[:, 7:36],
      soft_joint_pos_limits=limits.to(DTYPE)[None].expand(N, 29, 2),
      gravity_vec_w=torch.tensor([0.0, 0.0, -1.0], dtype=DTYPE).expand(N, 3),
    )
  )
  env = SimpleNamespace(
    command_manager=SimpleNamespace(get_term=lambda name: cmd),
    action_manager=SimpleNamespace(action=a, prev_action=a_prev),
    scene={"robot": robot},
  )
  asset = SimpleNamespace(name="robot", joint_ids=slice(None))
  out = {
    "motion_global_root_pos": trk_rewards.motion_global_anchor_position_error_exp(env, "motion", 0.3),
    "motion_global_root_ori": trk_rewards.motion_global_anchor_orientation_error_exp(env, "motion", 0.4),
    "motion_body_pos": trk_rewards.motion_relative_body_position_error_exp(env, "motion", 0.3),
    "motion_body_ori": trk_rewards.motion_relative_body_orientation_error_exp(env, "motion", 0.4),
    "motion_body_lin_vel": trk_rewards.motion_global_body_linear_velocity_error_exp(env, "motion", 1.0),
    "motion_body_ang_vel": trk_rewards.motion_global_body_angular_velocity_error_exp(env, "motion", 3.14),
    "action_rate_l2": env_rewards.action_rate_l2(env),
    "joint_limit": env_rewards.joint_pos_limits(env, asset),
  }
  dones = {
    "anchor_pos": trk_terms.bad_anchor_pos_z_only(env, "motion", 0.25),
    "anchor_ori": trk_terms.bad_anchor_ori(env, asset, "motion", 0.8),
  }
  return out, dones


@pytest.fixture(scope="module")
def limits() -> torch.Tensor:
  return soft_joint_limits_from_xml()


@pytest.mark.parametrize("box", [False, True])
def test_matches_mjlab_reward_functions(limits: torch.Tensor, box: bool) -> None:
  gen = torch.Generator().manual_seed(0)
  ref = _ref(gen, box)
  kin_next, kin_prev = _kin_near(ref, gen), _kin_near(ref, gen)
  s_next = _state(limits, gen, box, ref)
  a = torch.randn(N, 29, generator=gen, dtype=DTYPE)
  a_prev = a + 0.3 * torch.randn(N, 29, generator=gen, dtype=DTYPE)
  cfg = TrackingRewardCfg.lift() if box else TrackingRewardCfg.walk()
  rew = TrackingReward(cfg)
  r, terms = rew(s_next, kin_next, ref, a, a_prev, kin_prev=kin_prev)

  expect, dones = _mjlab_terms(ref, kin_next, kin_prev, s_next, a, a_prev, limits)
  weights = {t.name: t.weight for t in cfg.terms}
  for name, val in expect.items():
    torch.testing.assert_close(terms[name], val * weights[name] * DT, atol=1e-9, rtol=1e-7, msg=name)
  # every exp term in its informative range, joint-limit violations actually present
  assert (expect["motion_body_pos"].mean() > 0.1) and (expect["joint_limit"] > 0).any()
  if box:
    e_box = (ref.box_pos_w - s_next[:, 71:74]).square().sum(-1)
    th = mj_math.quat_error_magnitude(ref.box_quat_w, s_next[:, 74:78])
    torch.testing.assert_close(terms["box_pos"], (-e_box / 0.1**2).exp() * DT)
    torch.testing.assert_close(terms["box_ori"], (-th.square() / 0.4**2).exp() * 0.5 * DT)
  torch.testing.assert_close(r, sum(terms.values()))
  assert set(terms) == {t.name for t in cfg.terms}

  # Terminations: same continuous errors and the same boolean as mjlab's functions.
  errs = rew.term_errors(kin_next, ref, s_next)
  assert len(errs) == (3 if box else 2)
  torch.testing.assert_close(errs[0], (ref.body_pos_w[:, ANCHOR, 2] - kin_next.pos[:, ANCHOR, 2]).abs())
  g = torch.tensor([0.0, 0.0, -1.0], dtype=DTYPE).expand(N, 3)
  tilt = (
    mj_math.quat_apply_inverse(ref.body_quat_w[:, ANCHOR], g)[:, 2]
    - mj_math.quat_apply_inverse(kin_next.quat[:, ANCHOR], g)[:, 2]
  ).abs()
  torch.testing.assert_close(errs[1], tilt)
  if not box:
    assert torch.equal(rew.terminated(kin_next, ref), dones["anchor_pos"] | dones["anchor_ori"])


def test_quat_helpers_match_mjlab() -> None:
  gen = torch.Generator().manual_seed(1)
  q1, q2 = _rand_quat(256, gen=gen), _rand_quat(256, gen=gen)
  q_near = quat_mul(_small_rot(256, scale=1e-4, gen=gen), q1)
  for a, b in ((q1, q2), (q1, q_near), (q1, q1), (q1, -q1), (2 * q1, q2)):
    torch.testing.assert_close(quat_error_magnitude(a, b), mj_math.quat_error_magnitude(a, b))


def test_reward_maxima_at_identical_states(limits: torch.Tensor) -> None:
  gen = torch.Generator().manual_seed(2)
  for cfg in (TrackingRewardCfg.walk(), TrackingRewardCfg.lift()):
    ref = _ref(gen, cfg.has_box)
    kin = SimpleNamespace(pos=ref.body_pos_w, quat=ref.body_quat_w, lin_vel=ref.body_lin_vel_w, ang_vel=ref.body_ang_vel_w)
    s = torch.zeros(N, 84 if cfg.has_box else 71, dtype=DTYPE)
    s[:, 3] = 1.0
    s[:, 7:36] = limits.to(DTYPE).mean(-1)
    if cfg.has_box:
      s[:, 71:74], s[:, 74:78] = ref.box_pos_w, ref.box_quat_w
    a = torch.randn(N, 29, generator=gen, dtype=DTYPE)
    rew = TrackingReward(cfg)
    r, terms = rew(s, kin, ref, a, a.clone(), kin_prev=kin)
    pos_sum = sum(t.weight for t in cfg.terms if t.weight > 0) * DT
    assert pos_sum == pytest.approx(0.13 if cfg.has_box else 0.1)
    torch.testing.assert_close(r, torch.full((N,), pos_sum, dtype=DTYPE))
    assert rew.max_step_reward == pytest.approx(pos_sum)
    for t in cfg.terms:
      torch.testing.assert_close(terms[t.name], torch.full((N,), max(t.weight, 0.0) * DT, dtype=DTYPE))


def _yaw_xy_transform(kin: SimpleNamespace, yaw: torch.Tensor, xy: torch.Tensor) -> SimpleNamespace:
  qz = torch.stack((torch.cos(yaw / 2), 0 * yaw, 0 * yaw, torch.sin(yaw / 2)), dim=-1)[:, None]
  t = torch.cat((xy, torch.zeros(N, 1, dtype=DTYPE)), dim=-1)[:, None]
  return SimpleNamespace(
    pos=quat_apply(qz, kin.pos) + t,
    quat=quat_mul(qz.expand_as(kin.quat), kin.quat),
    lin_vel=quat_apply(qz, kin.lin_vel),
    ang_vel=quat_apply(qz, kin.ang_vel),
  )


def test_relative_terms_invariant_to_robot_xy_yaw(limits: torch.Tensor) -> None:
  gen = torch.Generator().manual_seed(3)
  ref = _ref(gen)
  s = _state(limits, gen, box=False)
  a = torch.randn(N, 29, generator=gen, dtype=DTYPE)
  rew = TrackingReward(TrackingRewardCfg.walk())
  yaw = (2 * torch.rand(N, generator=gen, dtype=DTYPE) - 1) * math.pi
  xy = torch.randn(N, 2, generator=gen, dtype=DTYPE) * 3

  # (1) the reference itself, shifted in xy and yaw: relative terms stay at their maxima
  kin_ref = SimpleNamespace(pos=ref.body_pos_w, quat=ref.body_quat_w, lin_vel=ref.body_lin_vel_w, ang_vel=ref.body_ang_vel_w)
  moved = _yaw_xy_transform(kin_ref, yaw, xy)
  _, terms = rew(s, moved, ref, a, a, kin_prev=moved)
  for name in ("motion_body_pos", "motion_body_ori"):
    torch.testing.assert_close(terms[name], torch.full((N,), DT, dtype=DTYPE))
  assert (terms["motion_global_root_pos"] < 0.5 * DT * 0.99).all()  # global terms do see it

  # (2) a perturbed robot: relative terms unchanged by an extra rigid xy/yaw move
  kin_next, kin_prev = _kin_near(ref, gen), _kin_near(ref, gen)
  _, base = rew(s, kin_next, ref, a, a, kin_prev=kin_prev)
  _, shifted = rew(s, _yaw_xy_transform(kin_next, yaw, xy), ref, a, a, kin_prev=_yaw_xy_transform(kin_prev, yaw, xy))
  for name in ("motion_body_pos", "motion_body_ori"):
    torch.testing.assert_close(shifted[name], base[name], atol=1e-12, rtol=1e-9)
  assert not torch.allclose(shifted["motion_body_lin_vel"], base["motion_body_lin_vel"])


def test_reanchor_prev_vs_next(limits: torch.Tensor) -> None:
  gen = torch.Generator().manual_seed(4)
  ref = _ref(gen)
  kin_next, kin_prev = _kin_near(ref, gen), _kin_near(ref, gen)
  s = _state(limits, gen, box=False)
  a = torch.zeros(N, 29, dtype=DTYPE)
  prev = TrackingReward(TrackingRewardCfg.walk())
  nxt = TrackingReward(TrackingRewardCfg.walk(reanchor="next"))
  _, t_prev = prev(s, kin_next, ref, a, a, kin_prev=kin_prev)
  _, t_next = nxt(s, kin_next, ref, a, a, kin_prev=kin_prev)
  assert not torch.allclose(t_prev["motion_body_pos"], t_next["motion_body_pos"])
  with pytest.warns(UserWarning, match="kin_prev"):
    _, t_fallback = prev(s, kin_next, ref, a, a)
  torch.testing.assert_close(t_fallback["motion_body_pos"], t_next["motion_body_pos"])


def test_joint_limit_by_hand() -> None:
  lim = torch.tensor([[-1.0, 1.0]] * 29)
  rew = TrackingReward(TrackingRewardCfg.walk(soft_joint_pos_limits=lim))
  s = torch.zeros(2, 71)
  s[:, 3] = 1.0
  s[0, 7], s[0, 8], s[0, 9] = 1.5, -1.25, 0.9  # violations 0.5 + 0.25
  ref = SimpleNamespace(body_pos_w=torch.zeros(2, 30, 3), body_quat_w=torch.tensor([1.0, 0, 0, 0]).expand(2, 30, 4),
                        body_lin_vel_w=torch.zeros(2, 30, 3), body_ang_vel_w=torch.zeros(2, 30, 3))
  kin = SimpleNamespace(pos=ref.body_pos_w, quat=ref.body_quat_w, lin_vel=ref.body_lin_vel_w, ang_vel=ref.body_ang_vel_w)
  _, terms = rew(s, kin, ref, torch.zeros(2, 29), torch.zeros(2, 29), kin_prev=kin)
  torch.testing.assert_close(terms["joint_limit"], torch.tensor([-2.0 * 0.75 * DT, 0.0]))


def test_gradients_finite_at_identical_states(limits: torch.Tensor) -> None:
  gen = torch.Generator().manual_seed(5)
  ref = _ref(gen, box=True)
  pos = ref.body_pos_w.clone().requires_grad_(True)
  quat = ref.body_quat_w.clone().requires_grad_(True)
  lin = ref.body_lin_vel_w.clone().requires_grad_(True)
  s = torch.zeros(N, 84, dtype=DTYPE)
  s[:, 3] = 1.0
  s[:, 7:36] = limits.to(DTYPE).mean(-1)
  s[:, 71:74], s[:, 74:78] = ref.box_pos_w, ref.box_quat_w
  s.requires_grad_(True)
  a = torch.zeros(N, 29, dtype=DTYPE, requires_grad=True)
  kin = SimpleNamespace(pos=pos, quat=quat, lin_vel=lin, ang_vel=ref.body_ang_vel_w)
  rew = TrackingReward(TrackingRewardCfg.lift())
  r, _ = rew(s, kin, ref, a, a.detach(), kin_prev=kin)
  errs = rew.term_errors(kin, ref, s)
  (r.sum() + sum(e.sum() for e in errs)).backward()
  for x in (pos, quat, lin, s, a):
    assert torch.isfinite(x.grad).all()


def test_box_terms_need_box_state() -> None:
  with pytest.raises(ValueError, match="has_box"):
    TrackingReward(TrackingRewardCfg(terms=TRACKING_TERMS + BOX_TERMS))
  rew = TrackingReward(TrackingRewardCfg.lift())
  ref = _ref(torch.Generator().manual_seed(6), box=True)
  kin = SimpleNamespace(pos=ref.body_pos_w, quat=ref.body_quat_w)
  with pytest.raises(ValueError, match="s_next"):
    rew.term_errors(kin, ref)


def test_soft_limits_match_torchfk() -> None:
  kin_mod = pytest.importorskip("mjlab_g1.rl.supermimic.kinematics")
  fk = kin_mod.TorchFK.g1()
  torch.testing.assert_close(soft_joint_limits_from_xml(), fk.soft_joint_limits.float())


def test_from_env_reads_weights_and_skips_zero_terms() -> None:
  def cfg(func, weight, **params):
    return SimpleNamespace(func=func, weight=weight, params=params)

  def box_position_error_exp(env, std):  # stand-in for the env team's box term
    raise AssertionError

  rterms = {
    "motion_global_root_pos": cfg(trk_rewards.motion_global_anchor_position_error_exp, 0.5, command_name="motion", std=0.3),
    "motion_body_ang_vel": cfg(trk_rewards.motion_global_body_angular_velocity_error_exp, 1.0, command_name="motion", std=3.14),
    "action_rate_l2": cfg(env_rewards.action_rate_l2, -0.1),
    "joint_limit": cfg(env_rewards.joint_pos_limits, -2.0, asset_cfg=None),
    "self_collisions": cfg(trk_rewards.self_collision_cost, 0.0, sensor_name="x"),
    "box_pos": cfg(box_position_error_exp, 1.0, std=0.1),
  }
  tterms = {
    "anchor_pos": cfg(trk_terms.bad_anchor_pos_z_only, 0.0, command_name="motion", threshold=0.2),
    "anchor_ori": cfg(trk_terms.bad_anchor_ori, 0.0, command_name="motion", threshold=0.7),
    "box_pos": cfg(lambda env: None, 0.0, threshold=0.35),
  }
  lim = torch.tensor([[-1.0, 1.0]] * 29)
  env = SimpleNamespace(
    reward_manager=SimpleNamespace(active_terms=list(rterms), get_term_cfg=rterms.__getitem__),
    termination_manager=SimpleNamespace(active_terms=list(tterms), get_term_cfg=tterms.__getitem__),
    command_manager=SimpleNamespace(get_term=lambda n: SimpleNamespace(cfg=SimpleNamespace(body_names=G1_TRACKED_BODY_NAMES, anchor_body_name="torso_link"))),
    scene={"robot": SimpleNamespace(body_names=G1_BODY_NAMES, data=SimpleNamespace(soft_joint_pos_limits=lim[None].expand(3, 29, 2)))},
    step_dt=0.02,
    cfg=SimpleNamespace(scale_rewards_by_dt=True),
    has_box=True,
  )
  c = TrackingRewardCfg.from_env(env)
  assert [(t.name, t.kind, t.weight, t.std) for t in c.terms] == [
    ("motion_global_root_pos", "anchor_pos", 0.5, 0.3),
    ("motion_body_ang_vel", "body_ang_vel", 1.0, 3.14),
    ("action_rate_l2", "action_rate", -0.1, 1.0),
    ("joint_limit", "joint_limit", -2.0, 1.0),
    ("box_pos", "box_pos", 1.0, 0.1),
  ]
  assert c.term_thresholds == (0.2, 0.7, 0.35) and c.has_box and c.dt == 0.02
  torch.testing.assert_close(torch.as_tensor(c.soft_joint_pos_limits), lim)
  TrackingReward(c)  # builds

  rterms["mystery"] = cfg(lambda env: None, 1.0)
  env.reward_manager.active_terms = list(rterms)
  with pytest.raises(ValueError, match="mystery"):
    TrackingRewardCfg.from_env(env)


def test_from_env_termination_mapping_is_strict() -> None:
  """Missing env criteria never fire (inf); an unmappable failure term raises."""
  def cfg(func, **params):
    return SimpleNamespace(func=func, weight=0.0, params=params, time_out=False)

  rterms = {"action_rate_l2": SimpleNamespace(func=env_rewards.action_rate_l2, weight=-0.1, params={})}
  tterms = {
    "time_out": SimpleNamespace(func=lambda env: None, weight=0.0, params={}, time_out=True),
    "anchor_pos": cfg(trk_terms.bad_anchor_pos_z_only, command_name="motion", threshold=0.25),
  }
  env = SimpleNamespace(
    reward_manager=SimpleNamespace(active_terms=list(rterms), get_term_cfg=rterms.__getitem__),
    termination_manager=SimpleNamespace(active_terms=list(tterms), get_term_cfg=tterms.__getitem__),
    command_manager=SimpleNamespace(get_term=lambda n: SimpleNamespace(cfg=SimpleNamespace(body_names=G1_TRACKED_BODY_NAMES, anchor_body_name="torso_link"))),
    scene={"robot": SimpleNamespace(body_names=G1_BODY_NAMES, data=SimpleNamespace(soft_joint_pos_limits=torch.tensor([[-1.0, 1.0]] * 29)[None]))},
    step_dt=0.02, cfg=SimpleNamespace(scale_rewards_by_dt=True), has_box=False,
  )  # fmt: skip
  c = TrackingRewardCfg.from_env(env)
  assert c.term_thresholds == (0.25, math.inf, math.inf)
  rew = TrackingReward(c)
  ref = _ref(torch.Generator().manual_seed(8))
  tilted = SimpleNamespace(pos=ref.body_pos_w.clone(), quat=quat_mul(_small_rot(N, 30, scale=4.0, gen=torch.Generator().manual_seed(9)), ref.body_quat_w))
  assert (rew.term_errors(tilted, ref)[1] > 0.8).any() and not rew.terminated(tilted, ref).any()

  tterms["ee_body_pos"] = cfg(trk_terms.bad_motion_body_pos, command_name="motion", threshold=0.25)
  env.termination_manager.active_terms = list(tterms)
  with pytest.raises(ValueError, match="ee_body_pos"):
    TrackingRewardCfg.from_env(env)


@pytest.mark.parametrize("box", [False, True])
def test_reward_gradcheck(limits: torch.Tensor, box: bool) -> None:
  """float64 gradcheck of the whole analytic reward + termination errors (the actor's gradient
  path), at generic states and at robot == reference (the safe-gradient branches)."""
  n = 2
  gen = torch.Generator().manual_seed(10)
  ref = _ref(gen, box)
  for name in vars(ref):
    setattr(ref, name, getattr(ref, name)[:n])
  rew = TrackingReward(TrackingRewardCfg.lift() if box else TrackingRewardCfg.walk())
  lim = limits.to(DTYPE)
  for exact in (False, True):
    noise = 0.0 if exact else 0.05
    pos = ref.body_pos_w + noise * torch.randn(n, 30, 3, generator=gen, dtype=DTYPE)
    quat = ref.body_quat_w if exact else quat_mul(_small_rot(n, 30, scale=0.2, gen=gen), ref.body_quat_w)
    lin = ref.body_lin_vel_w + noise * torch.randn(n, 30, 3, generator=gen, dtype=DTYPE)
    ang = ref.body_ang_vel_w + noise * torch.randn(n, 30, 3, generator=gen, dtype=DTYPE)
    s = torch.zeros(n, 84 if box else 71, dtype=DTYPE)
    s[:, 3] = 1.0
    # mid-range joints, plus two strictly outside the soft limits (the |x| kink stays away)
    s[:, 7:36] = lim.mean(-1)
    s[:, 7] = lim[0, 1] + 0.1
    s[:, 8] = lim[1, 0] - 0.1
    if box:
      s[:, 71:74] = ref.box_pos_w[:n] + (0.0 if exact else 0.05)
      s[:, 74:78] = ref.box_quat_w[:n] if exact else quat_mul(_small_rot(n, scale=0.2, gen=gen), ref.box_quat_w[:n])
    a = torch.randn(n, 29, generator=gen, dtype=DTYPE)
    a_prev = torch.randn(n, 29, generator=gen, dtype=DTYPE)
    prev_pos = pos + 0.02
    prev_quat = quat_mul(_small_rot(n, 30, scale=0.05, gen=gen), quat)
    inputs = [x.clone().requires_grad_(True) for x in (pos, quat, lin, ang, s, a, prev_pos, prev_quat)]

    def f(pos, quat, lin, ang, s, a, ppos, pquat):
      kin = SimpleNamespace(pos=pos, quat=quat, lin_vel=lin, ang_vel=ang)
      kp = SimpleNamespace(pos=ppos, quat=pquat)
      r, terms = rew(s, kin, ref, a, a_prev, kin_prev=kp)
      # term errors are |.| (kink at 0): checked away from robot == reference only
      errs = () if exact else rew.term_errors(kin, ref, s)
      return (r, *terms.values(), *errs)

    assert torch.autograd.gradcheck(f, inputs, eps=1e-6, atol=1e-6, rtol=1e-4)


def test_parity_on_real_env_transitions() -> None:
  """The real SMTrackEnv (walk + lift, CPU): analytic per-term rewards and terminations on
  (s_t, a_t, s_{t+1}, ref_idx_t) equal the env's own (brief §4 parity, exact kinematics)."""
  env_mod = pytest.importorskip("mjlab_g1.envs.smtrack_env")
  from mjlab_g1.tasks.smtrack import make_env_cfg

  torch.set_num_threads(min(4, torch.get_num_threads()))
  n, steps = 4, 40
  for task in ("G1-SMTrack-Walk", "G1-SMTrack-Lift"):
    cfg = make_env_cfg(task)
    cfg.scene.num_envs = n
    cfg.seed = 0
    env = env_mod.SMTrackEnv(cfg, device="cpu")
    try:
      env.reset()
      lib, fk = env.motion_lib, env.torch_fk
      rew = TrackingReward(TrackingRewardCfg.from_env(env))
      assert rew.names == env.reward_term_names
      am = env.action_manager
      scale = am.get_term(am.active_terms[0]).scale
      q0 = env.scene["robot"].data.default_joint_pos[0]
      gen = torch.Generator().manual_seed(0)
      a_prev = torch.zeros(n, 29)
      n_term, worst = 0, 0.0
      for t in range(steps):
        s, k = env.get_state().clone(), env.get_ref_index().clone()
        # open-loop reference joints + large noise: contacts, limit hits and falls
        a = (lib.joint_pos[lib.clamp(k + 1)] - q0) / scale + 1.0 * torch.randn(n, 29, generator=gen)
        _, r_env, term, trunc, _ = env.step(a)
        s1 = env.pre_reset_state.clone()
        kin1 = fk(s1[:, :36], s1[:, 36:71])
        ref = lib.frame(k)  # reward_ref_offset = 0: pre-advance frame
        r, terms = rew(s1, kin1, ref, a, a_prev, kin_prev=fk(s[:, :36], s[:, 36:71]))
        r_terms = env.get_reward_terms()
        for i, name in enumerate(env.reward_term_names):
          torch.testing.assert_close(terms[name], r_terms[:, i], atol=1e-5, rtol=0, msg=f"{task} {name} t={t}")
        worst = max(worst, (r - r_env).abs().max().item())
        assert torch.equal(rew.terminated(kin1, ref, s1), env.reset_terminated), f"{task} t={t}"
        n_term += int(env.reset_terminated.sum())
        a_prev = torch.where((term | trunc)[:, None], torch.zeros_like(a), a)
      assert worst < 1e-5, (task, worst)
      assert n_term > 0, f"{task}: no terminations, the termination check was vacuous"
    finally:
      env.close()
