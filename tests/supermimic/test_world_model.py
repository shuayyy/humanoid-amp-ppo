"""WorldModel: integrator identity / inverse, gradient safety, Eq. 1 masking, normalizers, toy fit
(design_learner.md §3, §8)."""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

from mjlab_g1.rl.supermimic.replay import ReplaySpec, SequenceReplay, Window
from mjlab_g1.rl.supermimic.world_model import (
  StateSpec,
  WorldModel,
  motion_action_scale,
  quat_exp,
  quat_log,
  quat_mul,
)

DT = 0.02
REPO = Path(__file__).resolve().parents[2]


def _identity_phi(s: torch.Tensor, kin) -> torch.Tensor:
  return s


def _model(spec: StateSpec, xi_dim: int = 3, **kw) -> WorldModel:
  kw.setdefault("hidden_dims", (32, 32))
  kw.setdefault("term_hidden_dims", (16,))
  return WorldModel(spec, _identity_phi, spec.state_dim, xi_dim, **kw)


def _rand_state(spec: StateSpec, B: int, g: torch.Generator, still: bool = False) -> torch.Tensor:
  nj = spec.num_joints
  vel = 0.0 if still else 1.0
  parts = {
    "p": torch.randn(B, 3, generator=g),
    "quat": torch.nn.functional.normalize(torch.randn(B, 4, generator=g), dim=-1),
    "q": 0.5 * torch.randn(B, nj, generator=g),
    "v": vel * torch.randn(B, 3, generator=g),
    "w": vel * 2.0 * torch.randn(B, 3, generator=g),
    "qd": vel * torch.randn(B, nj, generator=g),
    "bp": torch.randn(B, 3, generator=g),
    "bquat": torch.nn.functional.normalize(torch.randn(B, 4, generator=g), dim=-1),
    "bv": vel * torch.randn(B, 3, generator=g),
    "bw": vel * 2.0 * torch.randn(B, 3, generator=g),
  }
  return spec.join(parts, spec.state_layout())


# Independent float64 reference for "rotate by a constant WORLD angular velocity".
def _quat_mul64(a, b):
  aw, ax, ay, az = a.unbind(-1)
  bw, bx, by, bz = b.unbind(-1)
  return torch.stack(
    (
      aw * bw - ax * bx - ay * by - az * bz,
      aw * bx + ax * bw + ay * bz - az * by,
      aw * by - ax * bz + ay * bw + az * bx,
      aw * bz + ax * by - ay * bx + az * bw,
    ),
    -1,
  )


def _world_rotated(quat0: torch.Tensor, w_world: torch.Tensor, t: float) -> torch.Tensor:
  w = w_world.double()
  ang = w.norm(dim=-1, keepdim=True) * t
  axis = w / w.norm(dim=-1, keepdim=True)
  dq = torch.cat((torch.cos(ang / 2), axis * torch.sin(ang / 2)), -1)
  return _quat_mul64(dq, quat0.double())


def _quat_angle(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
  """Rotation angle between a and b (sign-invariant); atan2 is well conditioned near zero."""
  a = a.double()
  d = _quat_mul64(torch.cat((a[..., :1], -a[..., 1:]), -1), b.double())
  return 2.0 * torch.atan2(d[..., 1:].norm(dim=-1), d[..., 0].abs())


def test_spec_dims():
  assert (StateSpec.g1(False).state_dim, StateSpec.g1(False).out_dim) == (71, 70)
  assert (StateSpec.g1(True).state_dim, StateSpec.g1(True).out_dim) == (84, 82)
  toy = StateSpec(num_joints=3, robot_root=False)
  assert (toy.state_dim, toy.out_dim, toy.err_dim) == (6, 6, 6)
  # Brief §1 layout: qpos 36 = root pos 3, quat 4, joints 29; qvel 35.
  assert (StateSpec.g1().qpos_dim, StateSpec.g1().qvel_dim) == (36, 35)
  wm = WorldModel(
    StateSpec.g1(True), _identity_phi, 191, 35, hidden_dims=(8,), term_hidden_dims=(8,)
  )
  assert wm.dynamics[0].in_features == 191 + 29 + 35


def test_wm_integration_identity():
  spec = StateSpec.g1(has_box=True)
  g = torch.Generator().manual_seed(0)
  wm = _model(spec, xi_dim=35)
  B, n_steps = 64, 100
  s0 = _rand_state(spec, B, g)
  a = torch.randn(B, 29, generator=g)
  xi = torch.rand(B, 35, generator=g) * 2 - 1
  st0 = spec.split(s0, spec.state_layout())
  with torch.no_grad():
    # Zero-init output -> residual exactly zero -> constant-velocity semi-implicit Euler.
    assert torch.count_nonzero(wm.residual(s0, a, xi)) == 0
    s = s0
    for n in range(1, n_steps + 1):
      s = wm.step(s, a, xi)
      st = spec.split(s, spec.state_layout())
      for qk in ("quat", "bquat"):
        assert float((st[qk].norm(dim=-1) - 1).abs().max()) < 1e-6
      if n == 1:
        # One step equals the constant-velocity integrator exactly (up to float32 rounding).
        torch.testing.assert_close(st["p"], st0["p"] + DT * st0["v"], atol=1e-6, rtol=0)
        torch.testing.assert_close(st["q"], st0["q"] + DT * st0["qd"], atol=1e-6, rtol=0)
        torch.testing.assert_close(st["bp"], st0["bp"] + DT * st0["bv"], atol=1e-6, rtol=0)
        expect = quat_mul(st0["quat"], quat_exp(DT * _rot_inv(st0["quat"], st0["w"])))
        assert float(_quat_angle(st["quat"], expect).max()) < 1e-5
  t = n_steps * DT
  for kp, kv in (("p", "v"), ("q", "qd"), ("bp", "bv")):
    torch.testing.assert_close(st[kv], st0[kv], atol=1e-5, rtol=0)
    torch.testing.assert_close(st[kp], st0[kp] + t * st0[kv], atol=2e-4, rtol=0)
  for kq, kw in (("quat", "w"), ("bquat", "bw")):
    torch.testing.assert_close(st[kw], st0[kw], atol=1e-5, rtol=0)  # world ω constant
    ang = _quat_angle(st[kq], _world_rotated(st0[kq], st0[kw], t))
    assert float(ang.max()) < 1e-4


def _rot_inv(q, v):
  from mjlab_g1.rl.supermimic.world_model import quat_rotate_inv

  return quat_rotate_inv(q, v)


def test_quat_exp_log_inverse():
  g = torch.Generator().manual_seed(1)
  v = torch.randn(256, 3, generator=g)
  v = v / v.norm(dim=-1, keepdim=True) * torch.rand(256, 1, generator=g) * 3.0  # angles < π
  v[:4] = torch.tensor([[0.0, 0.0, 0.0], [1e-6, 0, 0], [0, -3e-5, 0], [1e-4, 1e-4, -1e-4]])
  torch.testing.assert_close(quat_log(quat_exp(v)), v, atol=1e-5, rtol=1e-5)
  torch.testing.assert_close(quat_log(-quat_exp(v)), v, atol=1e-5, rtol=1e-5)


def test_residual_targets_roundtrip():
  for spec in (StateSpec.g1(False), StateSpec.g1(True), StateSpec(3, robot_root=False)):
    g = torch.Generator().manual_seed(2)
    wm = _model(spec)
    B = 128
    s = _rand_state(spec, B, g)
    layout = spec.state_layout()
    # A plausible next state: small rotations, arbitrary everything else.
    nxt = spec.split(_rand_state(spec, B, g), layout)
    cur = spec.split(s, layout)
    if spec.robot_root:
      nxt["quat"] = quat_mul(cur["quat"], quat_exp(0.4 * torch.randn(B, 3, generator=g)))
    if spec.has_box:
      nxt["bquat"] = quat_mul(cur["bquat"], quat_exp(0.4 * torch.randn(B, 3, generator=g)))
    s1 = spec.join(nxt, layout)
    y = wm.residual_targets(s, s1)
    s1r = wm.integrate(s, y)
    e = wm.state_error(s1r, s1)
    assert float(e.abs().max()) < 2e-5, spec
    # And the other direction: residual_targets(s, integrate(s, y)) == y for in-range y.
    y2 = 0.05 * torch.randn(B, spec.out_dim, generator=g)
    y2r = wm.residual_targets(s, wm.integrate(s, y2))
    torch.testing.assert_close(y2r, y2, atol=2e-5, rtol=1e-4)
    # The error vector of a state against itself is zero.
    assert float(wm.state_error(s1, s1).abs().max()) < 1e-6


def test_grads_finite_at_rest():
  """Zero-init output at a robot at rest makes every exp/log angle exactly zero."""
  spec = StateSpec.g1(has_box=True)
  g = torch.Generator().manual_seed(3)
  wm = _model(spec, xi_dim=35)
  s = _rand_state(spec, 8, g, still=True).requires_grad_(True)
  a = torch.randn(8, 29, generator=g, requires_grad=True)
  xi = torch.zeros(8, 35)
  s1 = wm.step(s, a, xi)
  (s1.square().sum() + wm.term_prob(s1).sum()).backward()
  assert bool(torch.isfinite(s.grad).all()) and bool(torch.isfinite(a.grad).all())
  for name, p in wm.named_parameters():
    assert p.grad is None or bool(torch.isfinite(p.grad).all()), name
  # A rollout through window_loss with targets equal to the start state (zero error).
  wm.zero_grad()
  s_d = s.detach()
  w = _hand_window(s_d, torch.stack([s_d] * 3, 1), a.detach()[:, None].expand(-1, 3, -1), xi)
  loss, _ = wm.window_loss(w)
  loss.backward()
  for name, p in wm.named_parameters():
    assert p.grad is None or bool(torch.isfinite(p.grad).all()), name


def _hand_window(s0, target, a, xi, mask=None, term=None, ref_idx=None) -> Window:
  B, k = target.shape[:2]
  z = torch.zeros(B, k)
  return Window(
    s0=s0,
    target=target,
    a=a,
    a_prev=torch.zeros_like(a),
    xi=xi,
    r=z,
    r_terms=torch.zeros(B, k, 0),
    term=torch.zeros(B, k, dtype=torch.bool) if term is None else term,
    trunc=torch.zeros(B, k, dtype=torch.bool),
    mask=torch.ones(B, k) if mask is None else mask,
    ref_idx=torch.zeros(B, k, dtype=torch.long) if ref_idx is None else ref_idx,
    ep_step=torch.zeros(B, k, dtype=torch.long),
    env_ids=torch.arange(B),
    slots=torch.zeros(B, k, dtype=torch.long),
  )


def test_window_loss_mask_metrics_and_ref_pairing():
  spec = StateSpec.g1(has_box=False)
  g = torch.Generator().manual_seed(4)
  calls: list[torch.Tensor] = []

  def extra_fn(s, kin, ref):
    calls.append(ref["k"].clone())
    return torch.stack((s[..., 2], ref["k"].float() * 0.01), -1)

  # ref_fn maps the stored ref_idx to the object the termination features are scored against.
  ref_fn = lambda idx: {"k": idx + 100}  # noqa: E731
  wm = _model(spec, term_extra_fn=extra_fn, term_extra_dim=2, ref_fn=ref_fn)
  # Non-zero dynamics so the rollout actually moves.
  torch.nn.init.normal_(wm.dynamics[-1].weight, std=0.1)
  B, k = 32, 5
  s0 = _rand_state(spec, B, g)
  target = torch.stack([_rand_state(spec, B, g) for _ in range(k)], 1)
  a = torch.randn(B, k, 29, generator=g)
  xi = torch.rand(B, 3, generator=g)
  mask = torch.ones(B, k)
  mask[: B // 2, 3:] = 0
  mask[: B // 4, 1:] = 0
  term = torch.zeros(B, k, dtype=torch.bool)
  term[B // 2 :, 4] = True
  ref_idx = torch.arange(k).expand(B, k) + 7
  stats_before = {n: b.clone() for n, b in wm.named_buffers()}
  loss, metrics = wm.window_loss(_hand_window(s0, target, a, xi, mask, term, ref_idx))
  assert torch.isfinite(loss)
  for j in range(1, k + 1):
    assert f"err_norm_step{j}" in metrics
  for key in ("loss", "loss_state", "loss_bce", "term_acc", "term_rate", "mask_frac"):
    assert key in metrics and not metrics[key].requires_grad
  torch.testing.assert_close(metrics["loss"], metrics["loss_state"] + metrics["loss_bce"])
  # The termination features of step j are scored against ref_fn(ref_idx[:, j]).
  assert len(calls) == k and all(torch.equal(c, ref_idx[:, j] + 100) for j, c in enumerate(calls))
  # Imagination-style call term_prob(s', ref) gives the same head output as the loss path.
  s_probe = target[:, 0]
  d_ref = wm.term_prob(s_probe, ref_fn(ref_idx[:, 0]))
  d_extra = wm.term_prob(s_probe, extra_feats=extra_fn(s_probe, None, ref_fn(ref_idx[:, 0])))
  assert d_ref.shape == (B,) and bool(((d_ref > 0) & (d_ref < 1)).all())
  torch.testing.assert_close(d_ref, d_extra)
  calls.clear()
  # window_loss never touches the normalizers (they are frozen inside an update).
  for n, b in wm.named_buffers():
    assert torch.equal(b, stats_before[n]), n

  # Changing targets / labels at masked steps changes nothing.
  target2, term2 = target.clone(), term.clone()
  target2[mask == 0] += 100.0
  term2[mask == 0] = ~term2[mask == 0]
  loss2, metrics2 = wm.window_loss(_hand_window(s0, target2, a, xi, mask, term2, ref_idx))
  torch.testing.assert_close(loss2, loss)
  # Changing an unmasked target does.
  target3 = target.clone()
  target3[B - 1, k - 1] += 1.0
  loss3, _ = wm.window_loss(_hand_window(s0, target3, a, xi, mask, term, ref_idx))
  assert float((loss3 - loss).abs()) > 1e-6

  # Loss = mean_B Σ_j m_j [mse_j + bce_j]: recompute step 1 by hand.
  with torch.no_grad():
    s1 = wm.step(s0, a[:, 0], wm.encode_xi(xi))
    mse1 = (wm.state_error(s1, target[:, 0]) / wm.nom_std).square().mean(-1)
    m1 = mask[:, 0] > 0
    torch.testing.assert_close(metrics["err_norm_step1"], mse1[m1].mean())
    torch.testing.assert_close(metrics["err_rms_step1"], mse1[m1].mean().sqrt())

    # Termination diagnostics pooled over unmasked steps: Brier score and the mean d̂ on
    # positives / negatives, so a head that only learned the ~1% prior is visible.
    probs, sk = [], s0
    for j in range(k):
      sk = wm.step(sk, a[:, j], wm.encode_xi(xi))
      probs.append(wm.term_prob(sk, ref_fn(ref_idx[:, j])))
  valid = mask > 0
  P, Y = torch.stack(probs, 1)[valid], term.float()[valid]
  torch.testing.assert_close(metrics["term_brier"], (P - Y).square().mean())
  torch.testing.assert_close(metrics["term_p_pos"], P[Y > 0].mean())
  torch.testing.assert_close(metrics["term_p_neg"], P[Y == 0].mean())
  torch.testing.assert_close(metrics["term_rate"], Y.mean())


def test_update_normalizers():
  spec = StateSpec.g1(has_box=True)
  g = torch.Generator().manual_seed(5)
  wm = _model(spec)
  s = _rand_state(spec, 300, g)
  s1 = wm.integrate(s, 0.1 * torch.randn(300, spec.out_dim, generator=g))
  wm.update_normalizers(s[:100], s1[:100])
  # The algorithm passes the replay transition dict.
  wm.update_normalizers({"s": s[100:], "s_next": s1[100:], "a_exec": None})
  y = wm.residual_targets(s, s1)
  e = wm.state_error(s1, s)
  tol = dict(atol=1e-5, rtol=1e-4)
  torch.testing.assert_close(wm.out_std, y.std(0, unbiased=False).clamp_min(wm.out_floor), **tol)
  torch.testing.assert_close(wm.nom_std, e.std(0, unbiased=False).clamp_min(wm.nom_floor), **tol)
  both = torch.cat((s, s1))
  torch.testing.assert_close(wm.phi_norm.mean, both.mean(0), atol=1e-5, rtol=1e-4)
  torch.testing.assert_close(wm.phi_norm.std, both.std(0, unbiased=False), atol=1e-5, rtol=1e-4)
  assert float(wm.phi_norm.count) == 600
  # Non-finite rows are dropped instead of poisoning the cumulative stats.
  bad = s[:4].clone()
  bad[0, 0] = float("nan")
  wm.update_normalizers(bad, s1[:4])
  assert float(wm.phi_norm.count) == 606 and bool(torch.isfinite(wm.out_std).all())
  # Frozen normalizers ignore new data (critique M2 option for the algorithm).
  before = {n: b.clone() for n, b in wm.named_buffers()}
  wm.normalizers_frozen = True
  wm.update_normalizers(s, 3.0 * s1)
  assert all(torch.equal(b, before[n]) for n, b in wm.named_buffers())


def _quat_rows(B: int, g: torch.Generator, angles: torch.Tensor) -> torch.Tensor:
  """Rotation vectors [B,3] with the given angles (random axes), float64."""
  ax = torch.nn.functional.normalize(torch.randn(B, 3, generator=g, dtype=torch.float64), dim=-1)
  return ax * angles.to(torch.float64)[:, None]


def test_gradcheck_float64():
  """Analytic vs finite-difference Jacobians of the exp/log maps (incl. both sides of the Taylor
  threshold and exactly zero) and of integrate / residual_targets / state_error."""
  g = torch.Generator().manual_seed(11)
  # Threshold is 1e-8 on |v|² (exp) and on |u|² = sin²(θ/2) (log): 0 and 1e-5 are Taylor for
  # both, 1.5e-4 is exact for exp / Taylor for log, 2.2e-4 is exact for both, then generic.
  angles = torch.tensor([0.0, 1e-5, 1.5e-4, 2.2e-4, 0.7, 1.5])
  v = _quat_rows(angles.numel(), g, angles).requires_grad_(True)
  assert torch.autograd.gradcheck(quat_exp, (v,), eps=1e-7, atol=1e-6)
  q = quat_exp(v.detach()).requires_grad_(True)
  assert torch.autograd.gradcheck(quat_log, (q,), eps=1e-7, atol=1e-6)
  qn = (-q.detach()).clone().requires_grad_(True)  # w < 0: the sign-flip branch
  assert torch.autograd.gradcheck(quat_log, (qn,), eps=1e-7, atol=1e-6)

  spec = StateSpec.g1(has_box=True)
  wm = _model(spec, xi_dim=35).double()
  B = 3
  s = _rand_state(spec, B, g).double()
  st = spec.split(s, spec.state_layout())
  st["w"][0] = 0.0  # a body at rest: exp(dt ω_b + δθ) at exactly zero below
  st["bw"][0] = 0.0
  s = spec.join(st, spec.state_layout()).requires_grad_(True)
  y = (0.05 * torch.randn(B, spec.out_dim, generator=g, dtype=torch.float64))
  y[0] = 0.0
  y.requires_grad_(True)
  assert torch.autograd.gradcheck(wm.integrate, (s, y), eps=1e-7, atol=1e-6)
  s1 = wm.integrate(s.detach(), y.detach()).requires_grad_(True)
  assert torch.autograd.gradcheck(wm.residual_targets, (s, s1), eps=1e-7, atol=1e-6)
  s_hat = wm.integrate(s1.detach(), 0.01 * y.detach()).requires_grad_(True)
  assert torch.autograd.gradcheck(wm.state_error, (s_hat, s1), eps=1e-7, atol=1e-6)
  # Zero error (prediction == target) is the exact-zero log case.
  same = s1.detach().clone().requires_grad_(True)
  assert torch.autograd.gradcheck(lambda x: wm.state_error(x, s1.detach()), (same,), eps=1e-7, atol=1e-6)


def test_window_loss_nonfinite_inputs():
  """A physics blow-up stores NaN states. They must be masked AND kept out of the gradient
  (where()'s backward would otherwise multiply 0 by a NaN derivative)."""
  spec = StateSpec.g1(has_box=True)
  g = torch.Generator().manual_seed(12)
  wm = _model(spec, xi_dim=35)
  torch.nn.init.normal_(wm.dynamics[-1].weight, std=0.1)
  B, k = 8, 4
  s0 = _rand_state(spec, B, g)
  target = torch.stack([_rand_state(spec, B, g) for _ in range(k)], 1)
  a = torch.randn(B, k, 29, generator=g)
  xi = torch.rand(B, 35, generator=g)
  mask = torch.ones(B, k)
  mask[0, 2:] = 0  # window 0 crosses a reset after step 1

  bad_t, bad_a, bad_s0, bad_xi = target.clone(), a.clone(), s0.clone(), xi.clone()
  bad_t[0, 3] = float("nan")  # already masked
  bad_t[1, 2, 5] = float("inf")  # blow-up inside an episode: steps 2.. masked
  bad_t[1, 3] = float("nan")
  bad_a[2, 1, 0] = float("nan")  # NaN action at step 1: steps 1.. masked
  bad_s0[3, 4] = float("nan")  # whole window invalid
  bad_xi[4, 0] = float("nan")  # whole window invalid
  loss, metrics = wm.window_loss(_hand_window(bad_s0, bad_t, bad_a, bad_xi, mask))
  wm.zero_grad()
  loss.backward()
  assert torch.isfinite(loss)
  for name, p in wm.named_parameters():
    assert p.grad is not None and bool(torch.isfinite(p.grad).all()), name

  # Same loss as the clean data with those steps masked out by hand.
  clean_mask = mask.clone()
  clean_mask[1, 2:] = 0
  clean_mask[2, 1:] = 0
  clean_mask[3] = 0
  clean_mask[4] = 0
  with torch.no_grad():
    ref_loss, ref_metrics = wm.window_loss(_hand_window(s0, target, a, xi, clean_mask))
  torch.testing.assert_close(loss.detach(), ref_loss)
  torch.testing.assert_close(metrics["mask_frac"], ref_metrics["mask_frac"])


def test_encode_xi_and_action_scale():
  spec = StateSpec.g1()
  lo = torch.tensor([0.5, -1.0, 2.0])
  hi = torch.tensor([1.5, 1.0, 2.0])  # last dim not randomized
  wm = _model(spec, xi_lo=lo, xi_hi=hi)
  enc = wm.encode_xi(torch.tensor([[0.5, 1.0, 2.0], [1.0, 0.0, 2.0]]))
  torch.testing.assert_close(enc, torch.tensor([[-1.0, 1.0, 0.0], [0.0, 0.0, 0.0]]))
  q_ref = torch.zeros(10, 29)
  q_ref[:, 0] = torch.linspace(0, 9, 10)
  sa = motion_action_scale(q_ref, torch.zeros(29), torch.full((29,), 0.5))
  assert math.isclose(float(sa[0]), float((q_ref[:, 0] / 0.5).std(unbiased=False)), rel_tol=1e-5)
  assert bool((sa[1:] == 0.5).all())


# ---- toy fit -------------------------------------------------------------------------------------


def _toy_sim(x, v, u, m, dt=DT, n_sub=4, k=20.0, c=0.5):
  """Wall - m1 - m2 - m3 spring chain, damped, forces on m1 and m3; 4 substeps like mjlab."""
  h = dt / n_sub
  for _ in range(n_sub):
    left = torch.cat((torch.zeros_like(x[:, :1]), x[:, :-1]), 1)
    right = torch.cat((x[:, 1:], x[:, -1:]), 1)
    f = -k * (x - left) + k * (right - x) - c * v
    f = f + torch.stack((u[:, 0], torch.zeros_like(u[:, 0]), u[:, 1]), 1)
    v = v + h * f / m
    x = x + h * v
  return x, v


def _toy_replay(N: int = 64, steps: int = 300, seed: int = 0) -> SequenceReplay:
  g = torch.Generator().manual_seed(seed)
  rb = SequenceReplay(N, steps, ReplaySpec(state_dim=6, action_dim=2, xi_dim=1))

  def reset(n):
    x = (torch.rand(n, 3, generator=g) - 0.5) * 0.2
    v = (torch.rand(n, 3, generator=g) - 0.5) * 0.4
    return x, v, 0.5 + 1.5 * torch.rand(n, 1, generator=g)  # ξ = mass

  x, v, m = reset(N)
  u, a_prev = torch.zeros(N, 2), torch.zeros(N, 2)
  ep = torch.zeros(N, dtype=torch.long)
  for _ in range(steps):
    u = 0.9 * u + torch.randn(N, 2, generator=g)
    a = u + 0.1 * torch.randn(N, 2, generator=g)
    x1, v1 = _toy_sim(x, v, a, m)
    term = x1[:, 2].abs() > 0.3
    trunc = (ep + 1 >= 60) & ~term
    rb.add(
      dict(
        s=torch.cat((x, v), 1),
        s_next=torch.cat((x1, v1), 1),
        a_exec=a,
        a_prev=a_prev,
        xi=m,
        r=torch.zeros(N),
        term=term,
        trunc=trunc,
        ref_idx=ep,
      )
    )
    done = (term | trunc)[:, None]
    xr, vr, mr = reset(N)
    x, v, m = torch.where(done, xr, x1), torch.where(done, vr, v1), torch.where(done, mr, m)
    a_prev, u = torch.where(done, 0.0, a), torch.where(done, 0.0, u)
    ep = torch.where(done[:, 0], 0, ep + 1)
  return rb


def test_wm_toy_fit():
  # Small MLPs gain nothing from many threads, and oversubscribed shared nodes make them crawl.
  n_threads = torch.get_num_threads()
  torch.set_num_threads(min(4, n_threads))
  try:
    _toy_fit()
  finally:
    torch.set_num_threads(n_threads)


def _toy_fit() -> None:
  torch.manual_seed(0)
  rb = _toy_replay()
  assert bool(rb.data["term"].any()), "toy data should contain terminations"
  spec = StateSpec(num_joints=3, robot_root=False)
  wm = WorldModel(
    spec,
    _identity_phi,
    phi_dim=6,
    xi_dim=1,
    action_dim=2,
    xi_lo=torch.tensor([0.5]),
    xi_hi=torch.tensor([2.0]),
    action_scale=rb.data["a_exec"].reshape(-1, 2).std(0),
    hidden_dims=(128, 128),
    term_hidden_dims=(64, 64),
  )
  wm.update_normalizers(rb.data["s"], rb.data["s_next"])
  eval_w = rb.sample_windows(1024, 8, generator=torch.Generator().manual_seed(1))
  with torch.no_grad():
    loss0, m0 = wm.window_loss(eval_w)
  opt = torch.optim.Adam(wm.parameters(), lr=1e-3)
  for _ in range(400):
    loss, _ = wm.window_loss(rb.sample_windows(256, 8))
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(wm.parameters(), 1.0)
    opt.step()
  with torch.no_grad():
    loss1, m1 = wm.window_loss(eval_w)
  drop = float(m0["loss_state"]) / float(m1["loss_state"])
  assert drop >= 10.0, (m0["loss_state"], m1["loss_state"])
  assert float(loss1) <= float(loss0) / 10.0, (loss0, loss1)
  # The autoregressive error should grow with the horizon but stay far below the zero-init model.
  assert float(m1["err_norm_step8"]) < 0.1 * float(m0["err_norm_step8"])
  assert float(m1["loss_bce"]) < float(m0["loss_bce"])


# ---- real G1 reference data (FK + features + termination errors, as g1_components wires them) ---


def _ref_states(lib, k: torch.Tensor) -> torch.Tensor:
  parts = [
    lib.body_pos_w[k, 0], lib.body_quat_w[k, 0], lib.joint_pos[k],
    lib.body_lin_vel_w[k, 0], lib.body_ang_vel_w[k, 0], lib.joint_vel[k],
  ]  # fmt: skip
  if lib.has_box:
    parts += [lib.box_pos_w[k], lib.box_quat_w[k], lib.box_lin_vel_w[k], lib.box_ang_vel_w[k]]
  return torch.cat(parts, -1)


@pytest.mark.parametrize("clip,has_box", [("locomotion1", False), ("dual_arm1_box", True)])
def test_real_reference_roundtrip_and_window_loss(clip: str, has_box: bool):
  npz = REPO / "dataset" / "tracking" / f"{clip}.npz"
  if not npz.exists():
    pytest.skip(f"{npz} missing")
  from mjlab_g1.rl.supermimic import features as feats
  from mjlab_g1.rl.supermimic.kinematics import ANCHOR_BODY_ID, TorchFK
  from mjlab_g1.rl.supermimic.reference import MotionLibrary
  from mjlab_g1.rl.supermimic.reward import TrackingReward, TrackingRewardCfg

  torch.manual_seed(13)
  lib, fk = MotionLibrary(npz), TorchFK.g1()
  assert lib.has_box == has_box
  reward = TrackingReward(TrackingRewardCfg.lift() if has_box else TrackingRewardCfg.walk())
  spec = StateSpec.g1(has_box)
  D = feats.XI_DIM[has_box]

  def term_feats(s, kin, ref):
    anchor_z = ref.body_pos_w[..., ANCHOR_BODY_ID, 2]
    return torch.stack([*reward.term_errors(kin, ref, s), anchor_z], dim=-1)

  wm = WorldModel(
    spec, lambda s, kin: feats.wm_features(s, kin, has_box), feats.WM_FEATURE_DIM[has_box], D,
    kin_fn=lambda s: feats.forward_kin(fk, s), term_extra_fn=term_feats,
    term_extra_dim=3 + int(has_box), ref_fn=lambda idx: lib.frame(idx),
    hidden_dims=(32, 32), term_hidden_dims=(8,),
  )  # fmt: skip

  # Consecutive 50 Hz frames: the residual targets reproduce the next frame exactly.
  k_all = torch.arange(lib.num_frames)
  s_all = _ref_states(lib, k_all)
  y = wm.residual_targets(s_all[:-1], s_all[1:])
  assert float(wm.state_error(wm.integrate(s_all[:-1], y), s_all[1:]).abs().max()) < 1e-5
  wm.update_normalizers(s_all[:-1], s_all[1:])
  assert bool(torch.isfinite(wm.out_std).all()) and bool(torch.isfinite(wm.nom_std).all())

  # Windows of logged reference transitions, one of them with a blown-up step.
  B, k = 6, 4
  k0 = torch.randint(0, lib.num_frames - k - 1, (B,))
  idx = k0[:, None] + torch.arange(k)
  target = _ref_states(lib, idx + 1)
  target[0, 2] = float("nan")
  w = _hand_window(_ref_states(lib, k0), target, 0.5 * torch.randn(B, k, 29), torch.zeros(B, D),
                   ref_idx=idx)  # fmt: skip
  loss, metrics = wm.window_loss(w)
  loss.backward()
  assert torch.isfinite(loss) and float(metrics["mask_frac"]) == pytest.approx(1 - 2 / (B * k))
  grads = [p.grad for p in wm.parameters() if p.grad is not None]
  assert all(bool(torch.isfinite(gr).all()) for gr in grads)
  # The zero-initialized output layer gets a non-zero gradient: the model can learn from here.
  assert float(wm.dynamics[-1].weight.grad.abs().sum()) > 0
  assert float(wm.term_head[-1].weight.grad.abs().sum()) > 0
