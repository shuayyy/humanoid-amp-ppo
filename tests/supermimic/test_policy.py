"""Actor/critic networks, imagined BPTT (Eq. 2), clip order and critic λ-targets (Eq. 3).

World model, FK, features and motion library are small toy stand-ins with the documented
interfaces (design §6); the reward is the real TrackingReward on toy kinematics. CPU only.
"""

from __future__ import annotations

import copy
import math
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from mjlab_g1.rl.supermimic.actor_critic import Actor, ActorInference, Critic, RunningNorm
from mjlab_g1.rl.supermimic.geometry import quat_apply, quat_from_axis_angle, quat_mul, quat_normalize
from mjlab_g1.rl.supermimic.imagination import (
  ImagFns,
  actor_step,
  actor_update,
  clip_and_step,
  critic_targets,
  fit_critic,
  imagine,
  make_imagine,
  start_history,
)
from mjlab_g1.rl.supermimic.reward import TrackingReward, TrackingRewardCfg

NB, NJ, K, L, XI = 30, 29, 3, 10, 33
F_OBS = 29 + 29 + 3 + 29  # toy obs frame
G_DIM = K * (29 + 29 + 3)
P_DIM = 5 + 64 + 3
BODY_NAMES = tuple(f"b{i}" for i in range(NB))
TRACKED = tuple(f"b{i}" for i in range(0, NB, 2))
EX = torch.tensor([1.0, 0.0, 0.0])
EZ = torch.tensor([0.0, 0.0, 1.0])

_g = torch.Generator().manual_seed(123)
OFFS = torch.randn(NB, 3, generator=_g) * 0.3
JIDX = torch.arange(NB) % NJ


# ---- toy kinematics / features / reference ---------------------------------------------------


def toy_fk(qpos: torch.Tensor, qvel: torch.Tensor) -> SimpleNamespace:
  root, rq, q = qpos[..., :3], quat_normalize(qpos[..., 3:7]), qpos[..., 7:36]
  jq, jqd = q[..., JIDX], qvel[..., 6:][..., JIDX]
  disp = OFFS + 0.05 * jq[..., None] * EX
  rq_b = rq[..., None, :].expand(*jq.shape, 4)
  return SimpleNamespace(
    pos=root[..., None, :] + quat_apply(rq_b, disp),
    quat=quat_mul(rq_b, quat_from_axis_angle(EZ, 0.2 * jq)),
    lin_vel=qvel[..., None, :3] + quat_apply(rq_b, 0.05 * jqd[..., None] * EX),
    ang_vel=qvel[..., None, 3:6] + 0.2 * jqd[..., None] * EZ,
  )


class ToyLib:
  def __init__(self, F: int = 40, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    t = torch.arange(F, dtype=torch.float32)[:, None] * 0.02
    qpos = torch.zeros(F, 36)
    qpos[:, 0] = t[:, 0] * 1.0
    qpos[:, 2] = 0.78
    qpos[:, 3] = 1.0
    phase = torch.rand(1, NJ, generator=g) * 6.28
    qpos[:, 7:] = 0.3 * torch.sin(6.0 * t + phase)
    qvel = torch.zeros(F, 35)
    qvel[:, 0] = 1.0
    qvel[:, 6:] = 1.8 * torch.cos(6.0 * t + phase)
    kin = toy_fk(qpos, qvel)
    self.num_frames = F
    self.qpos, self.qvel = qpos, qvel
    self.joint_pos, self.joint_vel = qpos[:, 7:], qvel[:, 6:]
    self.body_pos_w, self.body_quat_w = kin.pos, kin.quat
    self.body_lin_vel_w, self.body_ang_vel_w = kin.lin_vel, kin.ang_vel

  def frame(self, k: torch.Tensor) -> SimpleNamespace:
    k = k.clamp(0, self.num_frames - 1)
    return SimpleNamespace(
      joint_pos=self.joint_pos[k], joint_vel=self.joint_vel[k],
      body_pos_w=self.body_pos_w[k], body_quat_w=self.body_quat_w[k],
      body_lin_vel_w=self.body_lin_vel_w[k], body_ang_vel_w=self.body_ang_vel_w[k],
    )  # fmt: skip

  def alive(self, k: torch.Tensor) -> torch.Tensor:
    return k <= self.num_frames - 1


def toy_goal(s, kin, lib, k, off):
  kk = k[:, None] + off + torch.arange(K)
  ref = lib.frame(kk)
  rel = ref.body_pos_w[..., 0, :] - kin.pos[:, None, 0, :]  # depends on ŝ (differentiable)
  return torch.cat((ref.joint_pos, ref.joint_vel, rel), dim=-1).flatten(1)


def toy_obs_frame(s, a_prev, xi):
  return torch.cat((s[..., 7:36] + 0.01 * xi[..., :29], s[..., 42:71], s[..., 39:42], a_prev), dim=-1)


def toy_wm_features(s, kin):
  return torch.cat((s[..., 2:7], s[..., 7:71], kin.pos[..., 0, :]), dim=-1)


def toy_fns() -> ImagFns:
  return ImagFns(
    fk=toy_fk,
    goal=toy_goal,
    obs_frame=toy_obs_frame,
    wm_features=toy_wm_features,
    encode_xi=lambda xi: 2 * xi - 1,
    sensor_noise=lambda x: (2 * torch.rand_like(x) - 1) * 0.01,
  )


class ToyWM(nn.Module):
  """Joint PD toward the action + a learned residual (so WM params sit in the graph)."""

  def __init__(self):
    super().__init__()
    self.res = nn.Linear(71 + NJ + XI, 35)
    self.term = nn.Linear(71, 1)
    nn.init.normal_(self.res.weight, std=0.01)

  def step(self, s, a, xi_enc):
    qpos, qvel = s[..., :36], s[..., 36:71]
    pd = torch.cat((torch.zeros_like(qvel[..., :6]), 0.5 * (a - qpos[..., 7:]) - 0.1 * qvel[..., 6:]), -1)
    v1 = qvel + pd + self.res(torch.cat((s, a, xi_enc), -1))
    quat1 = quat_normalize(qpos[..., 3:7] + 0.01 * torch.cat((torch.zeros_like(v1[..., :1]), v1[..., 3:6]), -1))
    return torch.cat((qpos[..., :3] + 0.02 * v1[..., :3], quat1, qpos[..., 7:] + 0.02 * v1[..., 6:], v1), -1)

  def term_prob(self, s1, ref):
    return torch.sigmoid(self.term(s1).squeeze(-1) - 4.0)


def toy_reward() -> TrackingReward:
  cfg = TrackingRewardCfg(
    body_names=TRACKED, anchor_body="b6", lib_body_names=BODY_NAMES,
    soft_joint_pos_limits=torch.tensor([[-0.5, 0.5]] * NJ),
  )  # fmt: skip
  return TrackingReward(cfg)


def make_actor(**kw) -> Actor:
  return Actor(L * F_OBS, G_DIM, NJ, encoder_dims=(32, 16), decoder_dims=(32, 16), latent_dim=8, **kw)


def make_critic() -> Critic:
  return Critic(P_DIM, G_DIM, XI, hidden=(16, 8))


def make_start(lib: ToyLib, B: int = 8, seed: int = 0, H: int = 6) -> SimpleNamespace:
  g = torch.Generator().manual_seed(seed)
  k = torch.randint(0, lib.num_frames - H, (B,), generator=g)
  s = torch.cat((lib.qpos[k], lib.qvel[k]), -1) + 0.02 * torch.randn(B, 71, generator=g)
  s[:, 3:7] = quat_normalize(s[:, 3:7])
  xi = torch.rand(B, XI, generator=g)
  a_prev = 0.1 * torch.randn(B, NJ, generator=g)
  frame = toy_obs_frame(s, a_prev, xi)
  return SimpleNamespace(s=s, a_prev=a_prev, xi=xi, ref_idx=k, hist=frame[:, None].repeat(1, L, 1))


# ---- networks ----------------------------------------------------------------------------------


def test_actor_shapes_init_and_normalizers() -> None:
  torch.manual_seed(0)
  actor = make_actor()
  hist, goal = torch.randn(16, L, F_OBS), torch.randn(16, G_DIM)
  a = actor(hist.flatten(1), goal)
  assert a.shape == (16, NJ)
  with pytest.raises(ValueError):  # an unflattened history is a layout bug, not a convenience
    actor(hist, goal)
  assert a.abs().max() < 0.05  # last layer x0.01: starts near q0
  z = actor.latent(100 * goal)
  assert z.shape == (16, 8) and z.abs().max() <= 1.0  # tanh-bounded
  mean0 = actor.hist_norm.mean.clone()
  actor(hist.flatten(1), goal)
  assert torch.equal(actor.hist_norm.mean, mean0)  # forward never updates stats
  actor.eval()
  actor.update_normalizers(hist.flatten(1), goal)  # works in eval mode too (explicit call)
  assert not torch.equal(actor.hist_norm.mean, mean0)
  big = Actor(930, 640)  # paper widths build (Table A.4)
  assert [m.out_features for m in big.encoder.net if isinstance(m, nn.Linear)] == [2048, 1024, 512, 256, 32]
  assert [m.out_features for m in big.decoder.net if isinstance(m, nn.Linear)] == [2048, 1024, 512, 256, 29]
  assert big.decoder.net[0].in_features == 32 + 930


def test_running_norm_matches_batch_statistics() -> None:
  x = torch.randn(1000, 5) * torch.tensor([1.0, 2.0, 0.1, 5.0, 1.0]) + 3.0
  norm = RunningNorm(5)
  for chunk in x.split(137):
    norm.update(chunk)
  torch.testing.assert_close(norm.mean, x.mean(0), atol=1e-5, rtol=1e-5)
  torch.testing.assert_close(norm.var, x.var(0, unbiased=False), atol=1e-4, rtol=1e-4)


def test_critic_zero_init_and_broadcast() -> None:
  critic = make_critic()
  phi, goal = torch.randn(7, 4, P_DIM), torch.randn(7, 4, G_DIM)
  v = critic(phi, goal, torch.randn(4, XI), torch.rand(7, 4))
  assert v.shape == (7, 4) and torch.equal(v, torch.zeros(7, 4))
  assert critic.net[0].in_features == P_DIM + G_DIM + XI + 1


def test_inference_wrapper() -> None:
  actor = Actor(L * 5, 7, 4, encoder_dims=(8,), decoder_dims=(8,), latent_dim=2)
  hist, goal = torch.randn(3, L * 5), torch.randn(3, 7)
  inf = ActorInference(actor)
  out = inf.act({"actor_hist": hist, "actor_goal": goal})
  torch.testing.assert_close(out, actor(hist, goal))
  assert not out.requires_grad
  torch.testing.assert_close(inf(hist, goal), out)


def test_history_layout_and_replay_rebuild() -> None:
  """The actor sees fns.flatten_hist(history); a replay.Start-style batch (hist_s, hist_a_prev)
  is rebuilt frame by frame with obs_frame."""
  torch.manual_seed(6)
  lib = ToyLib()
  base = make_start(lib, B=4)
  hs = base.s[:, None].repeat(1, L, 1) + 0.01 * torch.randn(4, L, 71)
  ha = torch.randn(4, L, NJ)
  start = SimpleNamespace(s=base.s, a_prev=base.a_prev, xi=base.xi, ref_idx=base.ref_idx, hist_s=hs, hist_a_prev=ha)
  fns = toy_fns()
  hist = start_history(start, fns)
  for i in range(L):
    torch.testing.assert_close(hist[:, i], toy_obs_frame(hs[:, i], ha[:, i], base.xi))

  def term_major(h):  # a non-trivial layout, like features.flatten_history(..., "mjlab")
    return torch.cat([c.flatten(-2) for c in torch.split(h, [29, 29, 3, 29], dim=-1)], dim=-1)

  seen = []

  class Probe(nn.Module):
    def __init__(self, actor):
      super().__init__()
      self.actor = actor

    def forward(self, h, g):
      seen.append(h.detach())
      return self.actor(h, g)

  fns.flatten_hist = term_major
  imagine(start, Probe(make_actor()), ToyWM(), toy_reward(), make_critic(), lib, H=2, fns=fns, obs_noise=False)
  torch.testing.assert_close(seen[0], term_major(hist))


# ---- imagination: gradient flow ----------------------------------------------------------------


def test_grad_flow_real_reward() -> None:
  torch.manual_seed(1)
  lib, wm, actor, critic, reward = ToyLib(), ToyWM(), make_actor(), make_critic(), toy_reward()
  start = make_start(lib)
  snapshot = {k: v.clone() for k, v in vars(start).items()}
  opt = torch.optim.Adam(actor.parameters(), lr=1e-3, betas=(0.9, 0.95))

  ro = imagine(start, actor, wm, reward, critic, lib, H=6, gamma=0.99, fns=toy_fns())
  assert ro.J.requires_grad and torch.isfinite(ro.J).all()
  assert ro.r.abs().sum() > 0 and not ro.r.requires_grad and not ro.phi.requires_grad
  stats = actor_step(ro.actor_loss(), actor, opt)
  assert stats["skipped"] == 0.0 and math.isfinite(stats["grad_norm_preclip"])

  for name, p in actor.named_parameters():
    assert p.grad is not None and p.grad.abs().sum() > 0, name
  for p in list(wm.parameters()) + list(critic.parameters()):
    assert p.grad is None  # frozen while the graph was recorded
    assert p.requires_grad  # ...and unfrozen afterwards for their own optimizers
  for k, v in vars(start).items():
    assert not v.requires_grad and torch.equal(v, snapshot[k]), k  # buffer data untouched
  # Rollout layout algorithm.py consumes (time-major).
  H, B = 6, 8
  assert ro.s.shape == (H + 1, B, 71) and ro.g.shape == (H + 1, B, G_DIM)
  assert ro.r.shape == ro.d.shape == (H, B) and ro.m.shape == ro.c.shape == ro.V.shape == (H + 1, B)
  assert set(ro.r_terms) == set(reward.names)


def test_actor_gets_gradient_through_the_world_model() -> None:
  """With the reward depending on ŝ_{t+1} only (no action-rate term), the actor's gradient must
  come through the WM Jacobian ∂ŝ'/∂a."""
  torch.manual_seed(2)
  lib, wm, actor, critic = ToyLib(), ToyWM(), make_actor(), make_critic()
  cfg = TrackingRewardCfg(
    terms=tuple(t for t in toy_reward().terms if t.kind == "body_pos"), body_names=TRACKED,
    anchor_body="b6", lib_body_names=BODY_NAMES, soft_joint_pos_limits=torch.zeros(NJ, 2),
  )  # fmt: skip
  ro = imagine(make_start(lib), actor, wm, TrackingReward(cfg), critic, lib, H=3, fns=toy_fns())
  ro.actor_loss().backward()
  assert actor.decoder.net[-1].bias.grad.abs().sum() > 0


# ---- clip order ----------------------------------------------------------------------------------


def test_clip_global_norm_then_elementwise() -> None:
  lin = nn.Linear(4, 3)
  grads = [torch.tensor([[50.0, -0.3, 0.2, 0.01]] * 3), torch.tensor([-40.0, 0.5, 2.0])]
  for p, g in zip(lin.parameters(), grads, strict=True):
    p.grad = g.clone()
  before = [p.detach().clone() for p in lin.parameters()]
  stats = clip_and_step(lin, torch.optim.SGD(lin.parameters(), lr=1.0), 1.0, 0.1)

  total = torch.sqrt(sum(g.square().sum() for g in grads))
  coef = min(1.0, 1.0 / (float(total) + 1e-6))
  expected = [(g * coef).clamp(-0.1, 0.1) for g in grads]
  for p, b, e in zip(lin.parameters(), before, expected, strict=True):
    torch.testing.assert_close(p.detach(), b - e)
  assert stats["grad_norm_preclip"] == pytest.approx(float(total))
  # the other order (clamp first, then renormalize) is a different update: the test discriminates
  clamped = [g.clamp(-0.1, 0.1) for g in grads]
  n2 = torch.sqrt(sum(c.square().sum() for c in clamped))
  alt = [c * min(1.0, 1.0 / float(n2)) for c in clamped]
  assert not all(torch.allclose(a, e) for a, e in zip(alt, expected, strict=True))
  n_big = sum(int(((g * coef).abs() > 0.1).sum()) for g in grads)
  assert n_big == 4 and stats["frac_elem_clipped"] == pytest.approx(n_big / 15)


def test_nonfinite_gradient_skips_step() -> None:
  lin = nn.Linear(2, 2)
  before = [p.detach().clone() for p in lin.parameters()]
  loss = (lin(torch.ones(1, 2)) * torch.tensor(float("nan"))).sum()
  stats = actor_step(loss, lin, torch.optim.SGD(lin.parameters(), lr=1.0))
  assert stats["skipped"] == 1.0
  for p, b in zip(lin.parameters(), before, strict=True):
    assert torch.equal(p.detach(), b) and p.grad is None


def test_microbatches_give_the_same_gradient() -> None:
  torch.manual_seed(3)
  lib, wm, critic, reward = ToyLib(), ToyWM(), make_critic(), toy_reward()
  start = make_start(lib, B=8)
  grads, Js = [], []
  base = make_actor()
  for n in (1, 2):
    actor = copy.deepcopy(base)
    opt = torch.optim.SGD(actor.parameters(), lr=0.0)
    ro, stats = actor_update(start, actor, opt, wm, reward, critic, lib, 5, 0.99, fns=toy_fns(),
                             microbatches=n, obs_noise=False)  # fmt: skip
    grads.append([p.grad.clone() for p in actor.parameters()])
    Js.append(ro.J)
    assert "actor/J" in stats and "actor/|a|_p99" in stats
  torch.testing.assert_close(Js[0], Js[1])
  for g1, g2 in zip(*grads, strict=True):
    torch.testing.assert_close(g1, g2, atol=1e-6, rtol=1e-5)


# ---- survival weight / clip-end mask ------------------------------------------------------------


class ScriptWM(nn.Module):
  """ŝ_x counts steps; d̂ is read from a table at that step."""

  def __init__(self, d_table: torch.Tensor):
    super().__init__()
    self.p = nn.Parameter(torch.zeros(()))
    self.d_table = d_table

  def step(self, s, a, xi_enc):
    e0 = torch.zeros_like(s)
    e0[..., 0] = 1.0
    return s + e0 + self.p * 0.0 + 0.0 * a.sum(-1, keepdim=True)

  def term_prob(self, s1, ref):
    return self.d_table[s1[..., 0].round().long() - 1]


def script_reward(r_table: torch.Tensor):
  def reward(s1, kin1, ref, a, a_prev, kin_prev=None):
    r = r_table[s1[..., 0].round().long() - 1] + 0.0 * a.sum(-1)
    return r, {"r": r}

  return reward


def _script_setup(F: int, k0: int, v0: float):
  lib = ToyLib(F=F)
  start = make_start(lib, B=4, H=0)
  start.s[:, 0] = 0.0  # step counter
  start.ref_idx = torch.full((4,), k0)
  critic = make_critic()
  nn.init.constant_(critic.net[-1].bias, v0)
  return lib, start, critic


def test_survival_weight_by_hand() -> None:
  gamma, v0 = 0.9, 7.0
  d = torch.tensor([0.1, 0.2, 0.0, 1.0, 0.3])
  r = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
  lib, start, critic = _script_setup(F=40, k0=0, v0=v0)
  ro = imagine(start, make_actor(), ScriptWM(d), script_reward(r), critic, lib, H=5, gamma=gamma, fns=toy_fns())
  c = torch.tensor([1.0, 0.9, 0.72, 0.72, 0.0, 0.0])
  J = sum(gamma**t * c[t] * r[t] for t in range(5)) + gamma**5 * c[5] * v0
  torch.testing.assert_close(ro.c[:, 0], c)
  torch.testing.assert_close(ro.J, torch.full((4,), float(J)))
  # d̂ = 1 at step 3 zeroes everything after it: J equals the 4-step partial sum
  torch.testing.assert_close(ro.J[0], torch.tensor(1 + 0.9 * 0.9 * 2 + 0.81 * 0.72 * 3 + 0.729 * 0.72 * 4))


def test_clip_end_mask_stop_and_bootstrap() -> None:
  gamma, v0 = 0.9, 7.0
  d = torch.zeros(5)
  r = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
  lib, start, critic = _script_setup(F=10, k0=7, v0=v0)  # frames 7, 8, 9 exist; 10+ do not
  kw = dict(H=5, gamma=gamma, fns=toy_fns())
  ro = imagine(start, make_actor(), ScriptWM(d), script_reward(r), critic, lib, **kw)
  torch.testing.assert_close(ro.m[:, 0], torch.tensor([1.0, 1, 1, 0, 0, 0]))
  stop = 1 + gamma * 2 + gamma**2 * 3  # no reward, no bootstrap past the clip
  torch.testing.assert_close(ro.J, torch.full((4,), stop))
  ro_b = imagine(start, make_actor(), ScriptWM(d), script_reward(r), critic, lib, clip_end="bootstrap", **kw)
  torch.testing.assert_close(ro_b.J, torch.full((4,), stop + gamma**3 * v0))
  # truncation exactly at H: stop mode drops V_H, bootstrap mode keeps it
  lib, start, critic = _script_setup(F=10, k0=5, v0=v0)
  ro = imagine(start, make_actor(), ScriptWM(d), script_reward(r), critic, lib, **kw)
  ro_b = imagine(start, make_actor(), ScriptWM(d), script_reward(r), critic, lib, clip_end="bootstrap", **kw)
  full = sum(gamma**t * r[t] for t in range(5))
  torch.testing.assert_close(ro.J, torch.full((4,), float(full)))
  torch.testing.assert_close(ro_b.J, torch.full((4,), float(full + gamma**5 * v0)))


# ---- critic targets ------------------------------------------------------------------------------


def _brute_lambda_return(r, d, m, V, gamma, lam, bootstrap):
  """Explicit λ-return: (1-λ) Σ_n λ^{n-1} G^(n) + λ^{N-1} G^(N) with per-step discount γ·cont_j.

  stop: cont_j = (1-d_j) m_{j+1}, horizon H.  bootstrap: cont_j = 1-d_j, episode truncated at the
  first dead step T (value V_T bootstraps there)."""
  H, B = r.shape
  R = torch.zeros(H, B, dtype=torch.float64)

  def cont(j: int, b: int) -> float:
    c = gamma * (1 - d[j, b])
    return c if bootstrap else c * m[j + 1, b]

  def n_step(t: int, b: int, n: int) -> float:
    out, disc = 0.0, 1.0
    for i in range(n):
      out += disc * r[t + i, b]
      disc *= cont(t + i, b)
    return out + disc * V[t + n, b]

  for b in range(B):
    T = next((t for t in range(H + 1) if m[t, b] == 0), H) if bootstrap else H
    for t in range(H):
      if m[t, b] == 0:
        continue
      N = min(H, T) - t
      R[t, b] = (1 - lam) * sum(lam ** (n - 1) * n_step(t, b, n) for n in range(1, N)) + lam ** (
        N - 1
      ) * n_step(t, b, N)
  return R


@pytest.mark.parametrize("bootstrap", [False, True])
def test_critic_targets_match_brute_force(bootstrap: bool) -> None:
  g = torch.Generator().manual_seed(7)
  H, B, gamma, lam = 9, 12, 0.97, 0.8
  r = torch.randn(H, B, generator=g, dtype=torch.float64)
  d = torch.rand(H, B, generator=g, dtype=torch.float64) * 0.5
  V = torch.randn(H + 1, B, generator=g, dtype=torch.float64)
  ends = torch.randint(1, H + 3, (B,), generator=g)  # clip end per column (some past H)
  m = (torch.arange(H + 1)[:, None] < ends[None]).double()
  ro = SimpleNamespace(r=r, d=d, m=m, V=V, gamma=gamma, bootstrap_clip_end=bootstrap)
  R, w = critic_targets(ro, lam)
  brute = _brute_lambda_return(r, d, m, V, gamma, lam, bootstrap)
  torch.testing.assert_close(R * w, brute)
  torch.testing.assert_close(w, m[:H])
  # λ = 1, no termination, no clip end: plain discounted return bootstrapped at H
  ro1 = SimpleNamespace(r=r, d=0 * d, m=torch.ones_like(m), V=V, gamma=gamma, bootstrap_clip_end=bootstrap)
  R1, _ = critic_targets(ro1, 1.0)
  torch.testing.assert_close(R1[0], sum(gamma**t * r[t] for t in range(H)) + gamma**H * V[H])


def test_fit_critic_reduces_loss() -> None:
  torch.manual_seed(4)
  lib, wm, actor, critic, reward = ToyLib(), ToyWM(), make_actor(), make_critic(), toy_reward()
  ro = imagine(make_start(lib, B=32), actor, wm, reward, critic, lib, H=6, fns=toy_fns()).detached()
  critic.update_normalizers(ro.phi.flatten(0, 1), ro.g.flatten(0, 1))
  opt = torch.optim.Adam(critic.parameters(), lr=3e-3)
  first = fit_critic(critic, opt, ro, epochs=1, minibatches=4)
  for _ in range(30):
    last = fit_critic(critic, opt, ro, epochs=1, minibatches=4)
  assert last["critic/loss"] < 0.5 * first["critic/loss"]


def test_make_imagine_matches_direct_call() -> None:
  torch.manual_seed(5)
  lib, wm, actor, critic, reward = ToyLib(), ToyWM(), make_actor(), make_critic(), toy_reward()
  start = make_start(lib)
  imag = make_imagine(toy_fns(), obs_noise=False)
  ro = imag(start, actor, wm, reward, critic, lib, 4, 0.9)  # algorithm.py's 8-positional call
  direct = imagine(start, actor, wm, reward, critic, lib, H=4, gamma=0.9, fns=toy_fns(), obs_noise=False)
  torch.testing.assert_close(ro.J, direct.J)
  assert ro.gamma == 0.9 and ro.horizon == 4


def test_g1_feature_binding_shapes() -> None:
  F = pytest.importorskip("mjlab_g1.rl.supermimic.features")
  from mjlab_g1.rl.supermimic.kinematics import TorchFK

  fk = TorchFK.g1()
  for has_box, (S, P, O, X) in ((False, (71, 169, 93, 33)), (True, (84, 191, 102, 35))):
    fns = ImagFns.g1(fk, -torch.ones(X), torch.ones(X), has_box=has_box)
    s = torch.zeros(2, S)
    s[:, 3] = 1.0
    s[:, 2] = 0.78
    if has_box:
      s[:, 74] = 1.0
    kin = fns.fk(s[:, :36], s[:, 36:71])
    assert fns.wm_features(s, kin).shape == (2, P)
    frames = fns.obs_frame(s[:, None].repeat(1, L, 1), torch.zeros(2, L, NJ), torch.zeros(2, L, X))
    assert frames.shape == (2, L, O) and fns.sensor_noise(frames).shape == (2, L, O)
    # same flattening as the sim-side history buffer
    torch.testing.assert_close(fns.flatten_hist(frames), F.PolicyObsHistory.from_frames(frames).get())
    torch.testing.assert_close(fns.encode_xi(torch.zeros(2, X)), torch.zeros(2, X))


# ---- the real sibling modules (G1 dims, tiny widths) -------------------------------------------


def test_imagine_with_real_world_model_features_and_fk() -> None:
  F = pytest.importorskip("mjlab_g1.rl.supermimic.features")
  wm_mod = pytest.importorskip("mjlab_g1.rl.supermimic.world_model")
  from mjlab_g1.rl.supermimic.kinematics import TorchFK
  from mjlab_g1.rl.supermimic.reference import MotionLibrary

  from pathlib import Path

  npz = Path(__file__).resolve().parents[2] / "dataset/tracking/locomotion1.npz"
  if not npz.exists():
    pytest.skip("dataset/tracking/locomotion1.npz missing")
  torch.manual_seed(7)
  lib, fk = MotionLibrary(npz), TorchFK.g1()
  B, H = 4, 3
  xi_lo, xi_hi = -torch.ones(33), torch.ones(33)
  fns = ImagFns.g1(fk, xi_lo, xi_hi, has_box=False)
  reward = TrackingReward(TrackingRewardCfg.walk(soft_joint_pos_limits=fk.soft_joint_limits))

  calls = []

  def term_extra(s, kin, ref):
    calls.append(ref)
    return torch.stack(reward.term_errors(kin, ref), dim=-1)

  wm = wm_mod.WorldModel(
    wm_mod.StateSpec.g1(), lambda s, kin: F.wm_features(s, kin, False), 169, 33,
    kin_fn=lambda s: F.forward_kin(fk, s), term_extra_fn=term_extra, term_extra_dim=2,
    ref_fn=lambda k: lib.frame(k), hidden_dims=(32, 32), term_hidden_dims=(8,),
  )  # fmt: skip
  k = torch.tensor([0, 20, 50, lib.num_frames - 2])
  s = torch.cat((lib.body_pos_w[k, 0], lib.body_quat_w[k, 0], lib.joint_pos[k],
                 lib.body_lin_vel_w[k, 0], lib.body_ang_vel_w[k, 0], lib.joint_vel[k]), -1)  # fmt: skip
  start = SimpleNamespace(
    s=s, a_prev=torch.zeros(B, NJ), xi=torch.zeros(B, 33), ref_idx=k,
    hist_s=s[:, None].repeat(1, L, 1), hist_a_prev=torch.zeros(B, L, NJ),
  )  # fmt: skip
  actor = Actor(L * 93, 640, NJ, encoder_dims=(16,), decoder_dims=(16,), latent_dim=4)
  critic = Critic(169, 640, 33, hidden=(8,))
  opt = torch.optim.Adam(actor.parameters(), lr=1e-3)
  ro, stats = actor_update(start, actor, opt, wm, reward, critic, lib, H, 0.99, fns=fns)
  assert ro.s.shape == (H + 1, B, 71) and ro.g.shape == (H + 1, B, 640) and ro.phi.shape[-1] == 169
  assert torch.isfinite(ro.J).all() and stats["skipped"] == 0.0
  # starting on the reference with a constant-velocity WM: high but not perfect reward
  assert ((ro.r[0] > 0.5 * reward.max_step_reward) & (ro.r[0] <= reward.max_step_reward + 1e-6)).all()
  torch.testing.assert_close(ro.m[:, -1], torch.tensor([1.0, 1.0, 0.0, 0.0]))  # clip ends at F-1
  for p in actor.parameters():
    assert p.grad is not None and torch.isfinite(p.grad).all()
  assert all(p.grad is None for p in wm.parameters())
  assert len(calls) == H and torch.equal(calls[0].joint_pos, lib.frame(k).joint_pos)  # via ref_from_idx


# ---- review additions ----------------------------------------------------------------------------


def test_running_norm_skips_nonfinite_rows() -> None:
  x = torch.randn(200, 4) * 2.0 + 1.0
  bad = x.clone()
  bad[3, 1], bad[50, 0], bad[77, 3] = float("nan"), float("inf"), -float("inf")
  ok = torch.isfinite(bad).all(-1)
  norm, ref = RunningNorm(4), RunningNorm(4)
  for chunk, ref_chunk in zip(bad.split(64), bad[ok].split(64), strict=False):
    norm.update(chunk)
  ref.update(bad[ok])
  assert torch.isfinite(norm.mean).all() and torch.isfinite(norm.var).all()
  torch.testing.assert_close(norm.mean, ref.mean, atol=1e-5, rtol=1e-5)
  torch.testing.assert_close(norm.var, ref.var, atol=1e-4, rtol=1e-4)
  assert float(norm.count) == float(ok.sum())
  before = (norm.mean.clone(), norm.var.clone(), norm.count.clone())
  norm.update(torch.full((5, 4), float("nan")))  # an all-bad batch changes nothing
  for b, a in zip(before, (norm.mean, norm.var, norm.count), strict=True):
    assert torch.equal(a, b)


class _Probe(nn.Module):
  """Records the (flattened) history the wrapped actor sees at each call."""

  def __init__(self, actor: nn.Module):
    super().__init__()
    self.actor, self.seen = actor, []

  def forward(self, h, g):
    self.seen.append(h.detach())
    return self.actor(h, g)


def test_start_noise_shared_on_padded_frames() -> None:
  """The sim fills the history with the first NOISY frame at reset, so padded copies of an
  episode's first frame carry the same η; later frames get fresh η."""
  torch.manual_seed(9)
  lib = ToyLib()
  base = make_start(lib, B=3)
  hs = base.s[:, None].repeat(1, L, 1) + 0.01 * torch.randn(3, L, 71)
  ha = torch.randn(3, L, NJ)
  # ep_step 0 (all padded), 3 (7 padded copies of slot 40), >= 9 (no padding)
  slots = torch.stack([torch.full((L,), 40), (40 + torch.arange(L) - 6).clamp(min=40), torch.arange(L) + 100])
  start = SimpleNamespace(s=base.s, a_prev=base.a_prev, xi=base.xi, ref_idx=base.ref_idx,
                          hist_s=hs, hist_a_prev=ha, hist_slots=slots)  # fmt: skip
  fns = toy_fns()
  clean = start_history(start, fns)
  probe = _Probe(make_actor())
  imagine(start, probe, ToyWM(), toy_reward(), make_critic(), lib, H=1, fns=fns)
  eta = probe.seen[0].reshape(3, L, -1) - clean
  assert (eta.abs() > 0).any(-1).all()  # every frame got noise
  torch.testing.assert_close(eta[0], eta[0, :1].expand(L, -1))  # all copies of one noisy frame
  torch.testing.assert_close(eta[1, :7], eta[1, 6:7].expand(7, -1))
  for i in range(6, L - 1):
    assert not torch.allclose(eta[1, i], eta[1, i + 1])
    assert not torch.allclose(eta[2, i], eta[2, i + 1])


def _real_wm(fns: ImagFns, phi_fn, reward: TrackingReward, lib, *, ref_fn="lib", seed: int = 0):
  wm_mod = pytest.importorskip("mjlab_g1.rl.supermimic.world_model")
  F = pytest.importorskip("mjlab_g1.rl.supermimic.features")
  torch.manual_seed(seed)
  wm = wm_mod.WorldModel(
    wm_mod.StateSpec.g1(), phi_fn, 169, 33,
    kin_fn=lambda s: fns.fk(s[..., :36], s[..., 36:71]),
    term_extra_fn=lambda s, kin, ref: torch.stack(reward.term_errors(kin, ref), dim=-1),
    term_extra_dim=2, ref_fn=(lambda k: lib.frame(k)) if ref_fn == "lib" else None,
    hidden_dims=(32, 32), term_hidden_dims=(8,),
  )  # fmt: skip
  with torch.no_grad():  # non-trivial dynamics/termination so φ actually matters
    wm.dynamics[-1].weight.normal_(0, 0.05)
    s = torch.zeros(8, 71)
    s[:, 3], s[:, 2] = 1.0, 0.78
    wm.phi_norm.update(F.wm_features(s, fns.fk(s[:, :36], s[:, 36:71]), False) + torch.randn(8, 169))
    wm.out_stats.update(0.01 * torch.randn(16, wm.spec.out_dim))
  return wm


def _real_setup(B: int = 3):
  from pathlib import Path

  from mjlab_g1.rl.supermimic.kinematics import TorchFK
  from mjlab_g1.rl.supermimic.reference import MotionLibrary

  npz = Path(__file__).resolve().parents[2] / "dataset/tracking/locomotion1.npz"
  if not npz.exists():
    pytest.skip("dataset/tracking/locomotion1.npz missing")
  lib, fk = MotionLibrary(npz), TorchFK.g1()
  fns = ImagFns.g1(fk, -torch.ones(33), torch.ones(33), has_box=False)
  reward = TrackingReward(TrackingRewardCfg.walk(soft_joint_pos_limits=fk.soft_joint_limits))
  k = torch.tensor([5, 60, 120])[:B]
  s = torch.cat((lib.body_pos_w[k, 0], lib.body_quat_w[k, 0], lib.joint_pos[k],
                 lib.body_lin_vel_w[k, 0], lib.body_ang_vel_w[k, 0], lib.joint_vel[k]), -1)  # fmt: skip
  start = SimpleNamespace(s=s, a_prev=torch.zeros(B, NJ), xi=torch.zeros(B, 33), ref_idx=k,
                          hist_s=s[:, None].repeat(1, L, 1), hist_a_prev=torch.zeros(B, L, NJ))  # fmt: skip
  return lib, fns, reward, start


def test_phi_shared_with_world_model_is_exact() -> None:
  """Handing imagination's φ(ŝ) to the WM (same φ function) changes nothing but the cost."""
  F = pytest.importorskip("mjlab_g1.rl.supermimic.features")
  lib, fns, reward, start = _real_setup()
  calls = [0]

  def counted(s, kin):
    calls[0] += 1
    return F.wm_features(s, kin, False)

  fns.wm_features = counted
  wm_shared = _real_wm(fns, counted, reward, lib)
  wm_own = _real_wm(fns, lambda s, kin: counted(s, kin), reward, lib)  # a different function object
  wm_own.load_state_dict(wm_shared.state_dict())
  base = Actor(L * 93, 640, NJ, encoder_dims=(16,), decoder_dims=(16,), latent_dim=4)
  critic = Critic(169, 640, 33, hidden=(8,))
  nn.init.normal_(critic.net[-1].weight, std=0.1)
  H, out = 4, []
  for wm in (wm_shared, wm_own):
    actor = copy.deepcopy(base)
    calls[0] = 0
    ro = imagine(start, actor, wm, reward, critic, lib, H=H, fns=fns, obs_noise=False)
    ro.actor_loss().backward()
    out.append((ro, [p.grad.clone() for p in actor.parameters()], calls[0]))
  (ro_a, g_a, n_a), (ro_b, g_b, n_b) = out
  assert n_a == H + 1 and n_b > n_a  # one φ per imagined state when shared
  torch.testing.assert_close(ro_a.J, ro_b.J)
  torch.testing.assert_close(ro_a.d, ro_b.d)
  assert (ro_a.d > 0).all() and (ro_a.d < 1).all()
  for x, y in zip(g_a, g_b, strict=True):
    torch.testing.assert_close(x, y, atol=1e-7, rtol=1e-5)


def test_term_prob_without_ref_fn_uses_reward_frame() -> None:
  """A WM built without ref_fn returns the raw index from ref_from_idx; imagination must then
  hand the termination head the reward's RefFrame, not an index tensor."""
  lib, fns, reward, start = _real_setup()
  wm_ref = _real_wm(fns, fns.wm_features, reward, lib)
  wm_none = _real_wm(fns, fns.wm_features, reward, lib, ref_fn=None)
  wm_none.load_state_dict(wm_ref.state_dict())
  actor = Actor(L * 93, 640, NJ, encoder_dims=(16,), decoder_dims=(16,), latent_dim=4)
  critic = Critic(169, 640, 33, hidden=(8,))
  kw = dict(H=3, fns=fns, obs_noise=False)
  ro_ref = imagine(start, actor, wm_ref, reward, critic, lib, **kw)
  ro_none = imagine(start, actor, wm_none, reward, critic, lib, **kw)
  torch.testing.assert_close(ro_none.d, ro_ref.d)


class _ReplayWM(nn.Module):
  """Returns the recorded next states: imagination then reproduces a real env trajectory."""

  def __init__(self, states: list[torch.Tensor]):
    super().__init__()
    self.states, self.t = states, 0

  def step(self, s, a, xi_enc):
    self.t += 1
    return self.states[self.t] + 0.0 * a.sum(-1, keepdim=True)

  def term_prob(self, s1, ref):
    return torch.zeros(s1.shape[:-1])


class _ReplayActor(nn.Module):
  def __init__(self, actions: list[torch.Tensor]):
    super().__init__()
    self.actions, self.t = actions, 0
    self.p = nn.Parameter(torch.zeros(()))

  def forward(self, h, g):
    a = self.actions[self.t] + 0.0 * self.p
    self.t += 1
    return a


def test_imagined_rewards_match_env_on_a_replayed_trajectory() -> None:
  """End-to-end indexing check against the real SMTrackEnv: feeding imagination the env's own
  states and actions must reproduce the env's per-step rewards (reference frame k_t + offset,
  re-anchoring on ŝ_t, a_prev threading) and its goal observation at the start."""
  env_mod = pytest.importorskip("mjlab_g1.envs.smtrack_env")
  from mjlab_g1.tasks.smtrack import make_env_cfg

  torch.set_num_threads(min(4, torch.get_num_threads()))
  n, warm, H = 4, 3, 8
  cfg = make_env_cfg("G1-SMTrack-Walk")
  cfg.scene.num_envs = n
  cfg.seed = 1
  env = env_mod.SMTrackEnv(cfg, device="cpu")
  try:
    env.reset()
    lib, fk = env.motion_lib, env.torch_fk
    rew = TrackingReward(TrackingRewardCfg.from_env(env))
    am = env.action_manager
    scale = am.get_term(am.active_terms[0]).scale
    q0 = env.scene["robot"].data.default_joint_pos[0]
    gen = torch.Generator().manual_seed(0)

    def act(k):
      return (lib.joint_pos[lib.clamp(k + 1)] - q0) / scale + 0.2 * torch.randn(n, 29, generator=gen)

    a_prev, obs = torch.zeros(n, 29), None
    for _ in range(warm):
      a = act(env.get_ref_index())
      obs, _, term, trunc, _ = env.step(a)
      a_prev = torch.where((term | trunc)[:, None], torch.zeros_like(a), a)
    s0, k0, xi = env.get_state().clone(), env.get_ref_index().clone(), env.get_xi().clone()
    goal_env = obs["actor_goal"].clone()
    states, acts, r_env, r_terms_env = [s0], [], [], []
    done = torch.zeros(n, dtype=torch.bool)
    for _ in range(H):
      a = act(env.get_ref_index())
      _, r, term, trunc, _ = env.step(a)
      done |= term | trunc
      states.append(env.pre_reset_state.clone())
      acts.append(a)
      r_env.append(r.clone())
      r_terms_env.append(env.get_reward_terms().clone())
    # a trajectory is comparable only up to its first episode end (then the env resets)
    keep = ~done
    assert keep.any(), "every env reset inside the window; lower the action noise"

    fns = ImagFns.g1(fk, env.xi_lo, env.xi_hi, has_box=False)
    start = SimpleNamespace(s=s0, a_prev=a_prev, xi=xi, ref_idx=k0,
                            hist_s=s0[:, None].repeat(1, L, 1), hist_a_prev=torch.zeros(n, L, 29))  # fmt: skip
    critic = Critic(169, 640, 33, hidden=(8,))
    ro = imagine(start, _ReplayActor(acts), _ReplayWM(states), rew, critic, lib, H=H, fns=fns,
                 obs_noise=False)  # fmt: skip
    torch.testing.assert_close(ro.r[:, keep], torch.stack(r_env)[:, keep], atol=1e-5, rtol=0)
    for i, name in enumerate(env.reward_term_names):
      torch.testing.assert_close(ro.r_terms[name][:, keep], torch.stack(r_terms_env)[:, keep, i],
                                 atol=1e-5, rtol=0, msg=name)  # fmt: skip
    torch.testing.assert_close(ro.m[:-1], lib.alive(k0[None] + torch.arange(H)[:, None]).float())
    # imagination's first goal is the env's actor_goal observation of the start state
    torch.testing.assert_close(ro.g[0], goal_env, atol=1e-5, rtol=0)
  finally:
    env.close()


@pytest.mark.parametrize("bootstrap", [False, True])
def test_algorithm_lambda_returns_match_critic_targets(bootstrap: bool) -> None:
  """algorithm.py fits the critic with its own copy of Eq. 3; it must equal critic_targets."""
  alg = pytest.importorskip("mjlab_g1.rl.supermimic.algorithm")
  g = torch.Generator().manual_seed(11)
  H, B, gamma, lam = 7, 10, 0.99, 0.95
  r = torch.randn(H, B, generator=g, dtype=torch.float64)
  d = torch.rand(H, B, generator=g, dtype=torch.float64) * 0.3
  V = torch.randn(H + 1, B, generator=g, dtype=torch.float64)
  ends = torch.randint(1, H + 3, (B,), generator=g)
  m = (torch.arange(H + 1)[:, None] < ends[None]).double()
  ro = SimpleNamespace(r=r, d=d, m=m, V=V, gamma=gamma, bootstrap_clip_end=bootstrap)
  R, w = critic_targets(ro, lam)
  R_alg = alg.lambda_returns(r, d, m, V, gamma, lam, bootstrap=bootstrap)
  torch.testing.assert_close(R_alg * w, R * w)
