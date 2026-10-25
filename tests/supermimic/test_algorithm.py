"""SuperMimic loop tests (CPU, no mjlab): config/schedule, Eq. 3 targets, actor clip order and
gradient isolation, a 3-iteration run, startup parity abort, checkpoint roundtrip, inference.

The env is a fake implementing the SMTrackEnv contract (brief §2) on a toy joint chain
(s = [q 29, q̇ 29]); the motion library, FK, features and reward are toys. The replay buffer,
world model (joint-chain StateSpec), actor/critic, imagination and PolicyObsHistory are the real
modules, wired the way algorithm.g1_components wires them for the G1.
"""

from __future__ import annotations

import dataclasses
import functools
import json
import math
from types import SimpleNamespace

import pytest
import torch

from mjlab_g1.rl.supermimic import actor_critic, imagination, replay, world_model
from mjlab_g1.rl.supermimic.algorithm import Components, lambda_returns
from mjlab_g1.rl.supermimic.config import SuperMimicRunnerCfg
from mjlab_g1.rl.supermimic.features import PolicyObsHistory
from mjlab_g1.rl.supermimic.runner import SuperMimicRunner

@pytest.fixture(autouse=True)
def _single_thread():
  # Tiny tensors: intra-op threads only add contention (100x slower on a loaded node).
  n = torch.get_num_threads()
  torch.set_num_threads(1)
  yield
  torch.set_num_threads(n)


NJ = 29
S = 2 * NJ
D = 33  # [friction 1, encoder bias 29, torso com 3]
DT = 0.02
NUM_FRAMES = 40
K_GOAL = 2
F_OBS = 3 * NJ
G_DIM = 2 * K_GOAL * NJ + 1


# ----- toys -----------------------------------------------------------------------------------


class ToyLib:
  def __init__(self, num_frames: int = NUM_FRAMES):
    t = torch.arange(num_frames, dtype=torch.float32)[:, None]
    ph = torch.linspace(0.0, 3.0, NJ)[None]
    self.joint_pos = 0.3 * torch.sin(0.15 * t + ph)
    self.joint_vel = 0.3 * 0.15 / DT * torch.cos(0.15 * t + ph)
    self.num_frames = num_frames
    self.clip_last = num_frames - 1

  def _k(self, k):
    return torch.as_tensor(k).long()

  def frame(self, k):
    kc = self._k(k).clamp(0, self.clip_last)
    return SimpleNamespace(joint_pos=self.joint_pos[kc], joint_vel=self.joint_vel[kc])

  def alive(self, k):
    return self._k(k) <= self.clip_last

  def t_left(self, k, horizon: int = 64):
    return (self.clip_last - self._k(k)).clamp(0, horizon).float() / horizon


class ToyReward:
  names = ("track", "vel", "action_rate")
  max_step_reward = 1.5 * DT

  def __call__(self, s1, kin1, ref, a, a_prev, kin_prev=None):
    q, qd = s1[..., :NJ], s1[..., NJ:]
    terms = {
      "track": torch.exp(-(q - ref.joint_pos).square().mean(-1) / 0.1) * DT,
      "vel": 0.5 * torch.exp(-(qd - ref.joint_vel).square().mean(-1) / 4.0) * DT,
      "action_rate": -0.1 * (a - a_prev).square().sum(-1) * DT,
    }
    return terms["track"] + terms["vel"] + terms["action_rate"], terms

  def term_errors(self, kin1, ref, s1):
    return ((s1[..., 0] - ref.joint_pos[..., 0]).abs(),)

  def terminated(self, kin1, ref, s1):
    return self.term_errors(kin1, ref, s1)[0] > 0.25


def toy_fk(qpos, qvel):
  # algorithm/imagination slice s into [:36] / [36:71]; the toy kin is just s again.
  return torch.cat([qpos, qvel], dim=-1)


def toy_feats() -> SimpleNamespace:
  def obs_frame(s, a_prev, xi):
    return torch.cat([s[..., :NJ] + xi[..., 1 : 1 + NJ], s[..., NJ:], a_prev], dim=-1)

  def goal(s, kin, lib, k, off):
    ref = lib.frame(k.long().unsqueeze(-1) + off + torch.arange(K_GOAL))
    err = s[..., :1] - ref.joint_pos[..., 0, :1]
    return torch.cat([ref.joint_pos.flatten(-2), ref.joint_vel.flatten(-2), err], dim=-1)

  def encode_xi(xi, lo, hi):
    span = hi - lo
    return torch.where(span > 0, 2.0 * (xi - lo) / span.clamp_min(1e-12) - 1.0, torch.zeros_like(xi))

  return SimpleNamespace(
    obs_frame=obs_frame,
    goal=goal,
    wm_features=lambda s, kin: s,
    encode_xi=encode_xi,
    sensor_noise=lambda f: 0.01 * (2.0 * torch.rand_like(f) - 1.0),
    PolicyObsHistory=functools.partial(PolicyObsHistory, layout="frame"),
  )


class FakeEnv:
  """VecEnv-wrapper-like fake with the SMTrackEnv contract (unwrapped is itself).

  Damped PD joints with a friction-scaled pull on joint 0, so some episodes terminate
  (|q0 - q0_ref| > 0.25) and the rest truncate at the clip end. The reward/termination are the
  same toy functions the learner uses, so startup parity is exact (plus `reward_bias`).
  """

  def __init__(self, num_envs: int = 8, reward_bias: float = 0.0, seed: int = 0):
    self.num_envs = num_envs
    self.num_actions = NJ
    self.device = torch.device("cpu")
    self.unwrapped = self
    self.motion_lib = ToyLib()
    self.reward_fn = ToyReward()
    self.reward_term_names = list(ToyReward.names)
    self.has_box = False
    self.env_origins = torch.zeros(num_envs, 3)
    self.xi_lo = torch.cat([torch.tensor([0.3]), torch.full((NJ,), -0.01), torch.tensor([-0.025, -0.05, -0.05])])
    self.xi_hi = torch.cat([torch.tensor([1.2]), torch.full((NJ,), 0.01), torch.tensor([0.025, 0.05, 0.05])])
    self.max_episode_length = NUM_FRAMES
    self.reward_bias = reward_bias
    self.gen = torch.Generator().manual_seed(seed)
    self._s = torch.zeros(num_envs, S)
    self._xi = torch.zeros(num_envs, D)
    self._k = torch.zeros(num_envs, dtype=torch.long)
    self._a_last = torch.zeros(num_envs, NJ)
    self._terms = torch.zeros(num_envs, len(ToyReward.names))
    self.episode_length_buf = torch.zeros(num_envs, dtype=torch.long)
    self._reset(torch.arange(num_envs))
    self.pre_reset_state = self._s.clone()

  def _reset(self, ids: torch.Tensor) -> None:
    n = ids.numel()
    k0 = torch.randint(0, NUM_FRAMES - 8, (n,), generator=self.gen)
    ref = self.motion_lib.frame(k0)
    self._s[ids, :NJ] = ref.joint_pos + 0.05 * torch.randn(n, NJ, generator=self.gen)
    self._s[ids, NJ:] = ref.joint_vel + 0.1 * torch.randn(n, NJ, generator=self.gen)
    self._xi[ids] = self.xi_lo + (self.xi_hi - self.xi_lo) * torch.rand(n, D, generator=self.gen)
    self._k[ids] = k0
    self._a_last[ids] = 0.0
    self.episode_length_buf[ids] = 0

  @property
  def action_manager(self):
    # mjlab ActionManager.action: the last raw action (0 after a reset).
    return SimpleNamespace(action=self._a_last)

  # SMTrackEnv contract.
  def get_state(self):
    return self._s.clone()

  def get_xi(self):
    return self._xi.clone()

  def get_ref_index(self):
    return self._k  # live buffer, advanced in place like mjlab's time_steps

  def get_reward_terms(self):
    return self._terms

  def get_observations(self):
    return {}

  def step(self, a: torch.Tensor):
    q, qd = self._s[:, :NJ], self._s[:, NJ:]
    qd1 = 0.85 * qd + 25.0 * (0.25 * a - q)
    qd1[:, 0] -= 8.0 * self._xi[:, 0]
    s1 = torch.cat([q + DT * qd1, qd1], dim=-1)
    ref = self.motion_lib.frame(self._k)
    r, terms = self.reward_fn(s1, s1, ref, a, self._a_last)
    term = self.reward_fn.terminated(s1, ref, s1)
    trunc = (self._k >= self.motion_lib.clip_last) & ~term
    done = term | trunc
    self._terms = torch.stack([terms[n] for n in ToyReward.names], dim=-1)
    self.pre_reset_state = s1.clone()
    self._s = s1
    self._a_last = a.clone()
    self.episode_length_buf += 1
    self._k += 1
    ids = done.nonzero().squeeze(-1)
    if ids.numel() > 0:
      self._reset(ids)
    return {}, r + self.reward_bias, done.long(), {"time_outs": trunc.clone()}


def toy_components(env: FakeEnv, cfg: SuperMimicRunnerCfg) -> Components:
  lib, rew, feats = env.motion_lib, env.reward_fn, toy_feats()
  L, off = cfg.policy.obs_history, cfg.reward_ref_offset

  def term_feats(s, kin, ref):
    return torch.stack([*rew.term_errors(kin, ref, s), ref.joint_pos[..., 0]], dim=-1)

  wm = world_model.WorldModel.from_cfg(
    cfg.wm,
    world_model.StateSpec(num_joints=NJ, robot_root=False),
    lambda s, kin: s,
    S,
    D,
    action_dim=NJ,
    kin_fn=lambda s: s,
    term_extra_fn=term_feats,
    term_extra_dim=2,
    ref_fn=lambda ref_idx: lib.frame(ref_idx + off),
    action_scale=torch.full((NJ,), 2.0),
    xi_lo=env.xi_lo,
    xi_hi=env.xi_hi,
    dt=DT,
  )
  fns = imagination.ImagFns(  # default flatten_hist = frame-major, like the toy PolicyObsHistory
    fk=toy_fk,
    goal=feats.goal,
    obs_frame=feats.obs_frame,
    wm_features=feats.wm_features,
    encode_xi=lambda xi: feats.encode_xi(xi, env.xi_lo, env.xi_hi),
    sensor_noise=feats.sensor_noise,
  )
  buf = replay.SequenceReplay(
    env.num_envs,
    cfg.replay_steps_per_env,
    replay.ReplaySpec(S, NJ, D, len(ToyReward.names)),
    hist_len=L,
    ref_valid=lambda ref_idx: lib.alive(ref_idx + off),
  )
  return Components(
    world_model=wm,
    actor=actor_critic.Actor.from_cfg(cfg.policy, L * F_OBS, G_DIM, NJ),
    critic=actor_critic.Critic.from_cfg(cfg.critic, S, G_DIM, D),
    replay=buf,
    reward=rew,
    fk=toy_fk,
    lib=lib,
    feats=feats,
    imagine=imagination.make_imagine(fns, ref_offset=off, clip_end=cfg.clip_end, hist_len=L),
    xi_lo=env.xi_lo,
    xi_hi=env.xi_hi,
  )


def make_runner(tmp_path=None, seed: int = 0, env: FakeEnv | None = None, **overrides):
  cfg = SuperMimicRunnerCfg.smoke(seed=seed, **overrides)
  torch.manual_seed(seed)
  env = env or FakeEnv(8, seed=seed)
  comps = toy_components(env, cfg)
  log_dir = None if tmp_path is None else str(tmp_path)
  return SuperMimicRunner(env, cfg, log_dir=log_dir, device="cpu", components=comps), env


# ----- config ---------------------------------------------------------------------------------


def test_config_defaults_schedule_and_dict_roundtrip():
  cfg = SuperMimicRunnerCfg()
  assert cfg.max_iterations == 320_000 and cfg.env_steps_per_iter == 2.5
  assert cfg.cosine_iters() == round(0.78 * 320_000)
  assert (cfg.wm.batch_windows, cfg.wm.num_batches, cfg.wm.window_k) == (2048, 10, 8)
  assert (cfg.policy.horizon, cfg.policy.num_imagined, cfg.policy.betas) == (32, 4096, (0.9, 0.95))
  assert cfg.clip_actions is None and cfg.memory.compile is True
  assert SuperMimicRunnerCfg.smoke().memory.compile is False
  T = cfg.cosine_iters()
  fs = [cfg.lr_factor(i) for i in (0, T // 4, T // 2, T, T + 1, 10**9)]
  assert fs[0] == pytest.approx(1.0) and fs[2] == pytest.approx(0.515)
  assert all(a > b for a, b in zip(fs[:3], fs[1:4]))
  assert fs[3] == fs[4] == fs[5] == pytest.approx(0.03)
  cfg.lr_cosine_iters = 100
  assert cfg.lr_factor(100) == pytest.approx(0.03)

  smoke = SuperMimicRunnerCfg.smoke()
  assert smoke.max_iterations == 3 and smoke.policy.horizon == 4 and smoke.wm.window_k == 2
  assert smoke.wm.batch_windows == 16 and smoke.policy.num_imagined == 16
  assert smoke.wm.hidden_dims == (128, 128, 128)  # 2048 / 16
  d = dataclasses.asdict(smoke)
  d["wm"]["hidden_dims"] = list(d["wm"]["hidden_dims"])  # as yaml would give it back
  assert SuperMimicRunnerCfg.from_dict(d) == smoke


# ----- Eq. 3 targets --------------------------------------------------------------------------


def _bruteforce_lambda(r, d, m, V, gamma, lam):
  H = r.shape[0]
  c = gamma * (1.0 - d) * m[1:]  # continuation of transition t
  R = torch.zeros_like(r)
  for t in range(H):
    n_max = H - t

    def G(n):
      out, disc = torch.zeros_like(r[0]), torch.ones_like(r[0])
      for i in range(n):
        out = out + disc * r[t + i]
        disc = disc * c[t + i]
      return out + disc * V[t + n]

    acc = sum((1 - lam) * lam ** (n - 1) * G(n) for n in range(1, n_max))
    R[t] = acc + lam ** (n_max - 1) * G(n_max)
  return R


def test_lambda_returns_match_bruteforce_and_imagination():
  g = torch.Generator().manual_seed(0)
  H, B = 6, 5
  r = torch.randn(H, B, generator=g)
  d = torch.rand(H, B, generator=g) * 0.3
  V = torch.randn(H + 1, B, generator=g)
  m = torch.ones(H + 1, B)
  m[4:, 0] = 0.0  # clip ends inside the horizon for env 0
  m[2:, 3] = 0.0
  R = lambda_returns(r, d, m, V, 0.99, 0.95)
  torch.testing.assert_close(R, _bruteforce_lambda(r, d, m, V, 0.99, 0.95))
  for boot in (False, True):
    ro = SimpleNamespace(r=r, d=d, m=m, V=V, gamma=0.99, bootstrap_clip_end=boot)
    ref, _ = imagination.critic_targets(ro, 0.95)
    torch.testing.assert_close(lambda_returns(r, d, m, V, 0.99, 0.95, bootstrap=boot), ref)


# ----- actor step -----------------------------------------------------------------------------


def test_actor_clip_order_and_gradient_isolation():
  runner, _ = make_runner()
  runner._start_collection()
  stats = SimpleNamespace(add=lambda *a, **k: None, add_sum=lambda *a, **k: None)
  for _ in range(12):
    runner._collect_step(stats)
  alg = runner.alg
  big = ToyReward()
  base = big.__call__
  alg.reward = lambda *a, **k: tuple(  # huge reward scale -> huge actor gradient
    (x * 1e6 if i == 0 else {n: v * 1e6 for n, v in x.items()}) for i, x in enumerate(base(*a, **k))
  )
  assert all(p.grad is None for p in alg.wm.parameters())
  ro, m = alg.actor_update()
  assert float(m["actor/grad_norm_preclip"]) > 1.0
  grads = [p.grad for p in alg.actor.parameters()]
  assert all(gr is not None for gr in grads)
  assert max(gr.abs().max().item() for gr in grads) <= 0.1 + 1e-7  # element clip last
  assert torch.sqrt(sum(gr.square().sum() for gr in grads)) <= 1.0 + 1e-5  # global clip first
  assert sum(gr.abs().sum().item() for gr in grads) > 0.0
  # WM and critic are frozen for the actor step: no grads written, flags restored.
  assert all(p.grad is None for p in alg.wm.parameters())
  assert all(p.grad is None for p in alg.critic.parameters())
  assert all(p.requires_grad for p in alg.wm.parameters())
  assert all(p.requires_grad for p in alg.critic.parameters())
  assert ro.r.shape == (4, 16) and ro.m.shape == (5, 16) and ro.V.shape == (5, 16)
  assert not ro.phi.requires_grad and not ro.J.requires_grad


def test_microbatched_actor_step_runs():
  runner, _ = make_runner(memory=dataclasses.replace(SuperMimicRunnerCfg.smoke().memory, imag_microbatches=2))
  runner._start_collection()
  stats = SimpleNamespace(add=lambda *a, **k: None, add_sum=lambda *a, **k: None)
  for _ in range(12):
    runner._collect_step(stats)
  ro, m = runner.alg.actor_update()
  assert ro.J.shape == (16,) and ro.g.shape[1] == 16
  assert math.isfinite(float(m["actor/J"]))
  out = runner.alg.critic_update(ro)
  assert math.isfinite(float(out["critic/loss"]))


def _collected_runner(steps: int = 12, **overrides):
  runner, env = make_runner(**overrides)
  runner._start_collection()
  for _ in range(steps):
    runner._collect_step(_NullStats())
  return runner, env


class _NullStats:
  def add(self, *a, **k):
    pass

  add_sum = add


def test_microbatched_actor_gradient_is_exact():
  # Design §7.3: accumulation over imagined micro-batches must give the full-batch gradient.
  runner, _ = _collected_runner()
  alg = runner.alg
  start = alg.replay.sample_starts(alg.cfg.policy.num_imagined)
  alg.replay.sample_starts = lambda batch: start  # same starts for both runs
  alg._imagine = functools.partial(alg._imagine, obs_noise=False)  # chunking reorders the RNG
  actor0 = {k: v.clone() for k, v in alg.actor.state_dict().items()}
  grads, norms = {}, {}
  for n in (1, 2, 4):
    alg.actor.load_state_dict(actor0)
    alg.actor_opt.state.clear()
    alg.cfg.memory.imag_microbatches = n
    _, m = alg.actor_update()
    grads[n] = [p.grad.clone() for p in alg.actor.parameters()]
    norms[n] = float(m["actor/grad_norm_preclip"])
  for n in (2, 4):
    assert norms[n] == pytest.approx(norms[1], rel=1e-4)
    for a, b in zip(grads[1], grads[n]):
      torch.testing.assert_close(b, a, rtol=1e-4, atol=1e-8)


def test_updates_leave_input_normalizers_untouched():
  # Critique M3: only sim collection feeds the normalizers, never the WM/actor/critic updates.
  runner, _ = _collected_runner()
  alg = runner.alg

  def buffers():
    mods = {"wm": alg.wm, "actor": alg.actor, "critic": alg.critic}
    return {f"{m}.{n}": b.clone() for m, mod in mods.items() for n, b in mod.named_buffers()}

  before = buffers()
  assert any("hist_norm" in k for k in before) and any("out_stats" in k for k in before)
  alg.wm_update()
  ro, _ = alg.actor_update()
  alg.critic_update(ro)
  after = buffers()
  for k in before:
    assert torch.equal(before[k], after[k]), k
  runner._collect_step(_NullStats())  # ...while collection does move them
  moved = buffers()
  for k in ("actor.hist_norm.mean", "critic.phi_norm.mean", "wm.phi_norm.count"):
    assert not torch.equal(after[k], moved[k]), k


def test_nonfinite_gradient_skips_the_step():
  runner, _ = _collected_runner()
  alg = runner.alg
  base = alg.reward

  def nan_reward(*a, **k):
    r, terms = base(*a, **k)
    return r * float("nan"), terms

  alg.reward = nan_reward
  before = [p.detach().clone() for p in alg.actor.parameters()]
  _, m = alg.actor_update()
  assert float(m["actor/skipped_steps"]) == 1.0
  assert all(torch.equal(a, b) for a, b in zip(before, alg.actor.parameters()))
  assert len(alg.actor_opt.state) == 0  # Adam never stepped: no momentum drift either
  alg.reward = base
  _, m = alg.actor_update()
  assert float(m["actor/skipped_steps"]) == 0.0
  assert not all(torch.equal(a, b) for a, b in zip(before, alg.actor.parameters()))


# ----- full loop ------------------------------------------------------------------------------


def test_three_iterations_end_to_end(tmp_path):
  runner, env = make_runner(tmp_path)
  cfg = runner.cfg
  base_lr = cfg.wm.learning_rate
  runner.learn(cfg.max_iterations)

  # Accumulator 2.5 -> 2, 3, 2 vectorized steps after the 16-step prefill.
  prefill = max(cfg.prefill_steps, math.ceil(cfg.parity_transitions / env.num_envs), cfg.wm.window_k + 1)
  assert runner.alg.replay.size == prefill + 7
  assert runner.total_env_steps == (prefill + 7) * env.num_envs
  assert runner.env_step_acc == pytest.approx(0.5)
  assert runner.current_learning_iteration == 3

  rows = [json.loads(line) for line in (tmp_path / "metrics.jsonl").read_text().splitlines()]
  assert [r["iteration"] for r in rows] == [1, 2, 3]
  for row in rows:
    for key in (
      "wall_time", "env_steps", "sim/r_mean", "wm/loss", "wm/loss_state", "wm/err_norm_step1",
      "wm/grad_norm", "actor/J", "actor/grad_norm_preclip", "imag/c_H", "critic/loss",
      "critic/expl_var", "perf/sim_ms", "perf/wm_ms", "perf/actor_ms", "perf/critic_ms",
      "lr_factor", "sim/|a|_p99", "imag/|a|_p99",
    ):  # fmt: skip
      assert key in row, key
      assert math.isfinite(row[key]), (key, row[key])
  assert rows[0]["parity/r_maxabs"] < 1e-6 and rows[0]["parity/transitions"] > 0
  # Shared cosine schedule stepped once per iteration on all three optimizers.
  lrs = [r["lr_factor"] for r in rows]
  assert lrs[0] == pytest.approx(1.0) and lrs[0] > lrs[1] > lrs[2] >= cfg.lr_final_frac - 1e-9
  for opt in (runner.alg.wm_opt, runner.alg.actor_opt, runner.alg.critic_opt):
    assert opt.param_groups[0]["lr"] == pytest.approx(base_lr * cfg.lr_factor(2))
  assert any(r.get("sim/episodes", 0) > 0 for r in rows) or rows[0].get("parity/trunc_events", 0) > 0
  for it in (1, 2, 3):
    assert (tmp_path / f"model_{it}.pt").exists()
  assert any(p.name.startswith("events.out.tfevents") for p in tmp_path.iterdir())


def test_startup_parity_aborts_on_reward_mismatch():
  env = FakeEnv(8, reward_bias=0.01)
  runner, _ = make_runner(env=env)
  with pytest.raises(RuntimeError, match="reward parity"):
    runner.learn(1)


def _prefill_steps(cfg: SuperMimicRunnerCfg, num_envs: int) -> int:
  return max(cfg.prefill_steps, math.ceil(cfg.parity_transitions / num_envs), cfg.wm.window_k + 1)


def test_parity_holds_on_an_env_stepped_before_the_runner():
  # a_{t-1} must be the env's last action (action_rate reward, last_action obs), not 0.
  env = FakeEnv(8, seed=0)
  g = torch.Generator().manual_seed(1)
  for _ in range(3):
    env.step(torch.randn(8, NJ, generator=g))
  assert env._a_last.abs().sum() > 0
  runner, _ = make_runner(env=env)
  runner.learn(1)  # would abort: 0.1 * |a|^2 * dt ~ 0.06 > 1e-3 with a_prev = 0
  assert runner.parity_report["parity/r_maxabs"] < 1e-6


def test_second_learn_call_continues_collection(tmp_path):
  runner, env = make_runner(tmp_path)
  runner.learn(2)
  size, steps = runner.alg.replay.size, runner.total_env_steps
  runner.learn(1)  # no second prefill / parity pass; same episodes continue
  assert runner.current_learning_iteration == 3
  assert runner.alg.replay.size == size + 2  # accumulator 2.5: 2, 3, then 2
  assert runner.total_env_steps == steps + 2 * env.num_envs
  rows = [json.loads(x) for x in (tmp_path / "metrics.jsonl").read_text().splitlines()]
  assert [r["iteration"] for r in rows] == [1, 2, 3]
  assert rows[-1]["iter_per_s"] > 0


def test_resume_from_checkpoint_continues(tmp_path):
  runner, env = make_runner(tmp_path / "a")
  runner.learn(2)
  ckpt = tmp_path / "a" / "model_2.pt"
  assert ckpt.exists() and runner.env_step_acc == pytest.approx(0.0)  # 2.5 -> 2, 3
  resumed, env2 = make_runner(tmp_path / "b", seed=5)
  resumed.load(str(ckpt))
  steps0 = resumed.total_env_steps
  assert resumed.current_learning_iteration == 2 and steps0 == runner.total_env_steps
  resumed.learn(1)  # relative: iteration index 2 only
  cfg = resumed.cfg
  assert resumed.current_learning_iteration == 3
  assert (tmp_path / "b" / "model_3.pt").exists()
  # Replay is not checkpointed (save_replay unimplemented): the prefill reruns on resume.
  prefill = _prefill_steps(cfg, env2.num_envs)
  assert resumed.alg.replay.size == prefill + 2
  assert resumed.total_env_steps == steps0 + (prefill + 2) * env2.num_envs
  assert resumed.env_step_acc == pytest.approx(0.5)
  for opt, base in (
    (resumed.alg.wm_opt, cfg.wm.learning_rate),
    (resumed.alg.actor_opt, cfg.policy.learning_rate),
    (resumed.alg.critic_opt, cfg.critic.learning_rate),
  ):
    assert opt.param_groups[0]["lr"] == pytest.approx(base * cfg.lr_factor(2))
  rows = [json.loads(x) for x in (tmp_path / "b" / "metrics.jsonl").read_text().splitlines()]
  assert [r["iteration"] for r in rows] == [3]


class MjlabFlagsEnv(FakeEnv):
  """Stock mjlab flags: `reset_terminated` plus time_outs that are NOT masked by terminated, so an
  env failing on its clip-end step has both flags set (SMTrackEnv masks them; mjlab does not)."""

  def step(self, a):
    clip_end = self._k >= self.motion_lib.clip_last
    obs, r, done, extras = super().step(a)
    done = done.bool()
    self.reset_terminated = done & ~extras["time_outs"]
    return obs, r, done.long(), {"time_outs": done & clip_end}


def test_reset_terminated_wins_over_time_outs():
  env = MjlabFlagsEnv(8, seed=0)
  env._k[:] = 0
  env._k[:2] = env.motion_lib.clip_last  # envs 0, 1 are on their last frame
  runner, _ = make_runner(env=env)
  runner._start_collection()
  force = torch.zeros(8, dtype=torch.bool)
  force[0] = True  # env 0 fails on the clip-end step, env 1 just times out
  env.reward_fn.terminated = lambda kin1, ref, s1: force.clone()
  runner._collect_step(_NullStats())
  buf = runner.alg.replay
  slot = (buf.ptr - 1) % buf.capacity
  term, trunc = buf.data["term"][:, slot], buf.data["trunc"][:, slot]
  assert term.tolist() == [True] + [False] * 7
  assert trunc.tolist() == [False, True] + [False] * 6


def test_env_log_extras_use_rsl_rl_names():
  from mjlab_g1.rl.supermimic.runner import _add_env_log, _Stats

  stats = _Stats(torch.device("cpu"))
  log = {
    "Episode_Reward/motion_body_pos": torch.tensor(0.5),
    "Episode_Termination/anchor_pos": 3,
    "mean_len": torch.tensor(2.0),
    "flag": True,
  }
  _add_env_log(stats, {"log": log, "time_outs": torch.zeros(2)})
  _add_env_log(stats, {"log": {"Episode_Reward/motion_body_pos": torch.tensor(1.5)}})
  _add_env_log(stats, {"time_outs": torch.zeros(2)})
  assert stats.flush() == pytest.approx(
    {"Episode_Reward/motion_body_pos": 1.0, "Episode_Termination/anchor_pos": 3.0, "Episode/mean_len": 2.0}
  )


def test_tf32_flag_is_applied_both_ways():
  matmul = torch.backends.cuda.matmul
  saved = matmul.fp32_precision
  try:
    smoke = SuperMimicRunnerCfg.smoke()
    make_runner(memory=dataclasses.replace(smoke.memory, tf32=False))
    assert matmul.fp32_precision == "ieee"
    make_runner(memory=dataclasses.replace(smoke.memory, tf32=True))
    assert matmul.fp32_precision == "tf32"
  finally:
    matmul.fp32_precision = saved


def test_wm_normalizer_freeze_knob():
  smoke = SuperMimicRunnerCfg.smoke()
  runner, env = make_runner(wm=dataclasses.replace(smoke.wm, freeze_normalizers_after=1))
  runner.learn(3)
  # Prefill and iteration 0 (2 steps) update the WM statistics; iterations 1, 2 do not.
  n = (_prefill_steps(runner.cfg, env.num_envs) + 2) * env.num_envs
  assert runner.alg.wm.out_stats.count.item() == n
  assert runner.alg.wm.normalizers_frozen


# ----- checkpoint + inference -----------------------------------------------------------------


def test_checkpoint_roundtrip(tmp_path):
  runner, env = make_runner(tmp_path, seed=0)
  runner.learn(2)
  path = tmp_path / "ckpt.pt"
  runner.save(str(path))

  other, _ = make_runner(seed=1)
  g = torch.Generator().manual_seed(3)
  hist = torch.randn(5, 10 * F_OBS, generator=g)
  goal = torch.randn(5, G_DIM, generator=g)
  s = torch.randn(5, S, generator=g)
  a = torch.randn(5, NJ, generator=g)
  xe = torch.rand(5, D, generator=g) * 2 - 1
  tl = torch.rand(5, generator=g)
  with torch.no_grad():
    assert not torch.allclose(runner.alg.actor(hist, goal), other.alg.actor(hist, goal))
  other.load(str(path))
  with torch.no_grad():
    torch.testing.assert_close(other.alg.actor(hist, goal), runner.alg.actor(hist, goal))
    torch.testing.assert_close(other.alg.wm.step(s, a, xe), runner.alg.wm.step(s, a, xe))
    torch.testing.assert_close(other.alg.critic(s, goal, xe, tl), runner.alg.critic(s, goal, xe, tl))
  torch.testing.assert_close(other.alg.wm.out_std, runner.alg.wm.out_std)
  assert other.current_learning_iteration == 2
  assert other.env_step_acc == pytest.approx(runner.env_step_acc)
  assert other.total_env_steps == runner.total_env_steps
  assert other.alg.lr_factor == pytest.approx(runner.cfg.lr_factor(2))
  st_a, st_b = runner.alg.actor_opt.state_dict(), other.alg.actor_opt.state_dict()
  assert st_a["state"][0]["step"] == st_b["state"][0]["step"]

  # Actor-only warm start: weights load, iteration counter does not.
  fresh, _ = make_runner(seed=2)
  fresh.load(str(path), load_cfg={"actor": True})
  assert fresh.current_learning_iteration == 0
  with torch.no_grad():
    torch.testing.assert_close(fresh.alg.actor(hist, goal), runner.alg.actor(hist, goal))


def test_inference_policy_is_deterministic_and_tracks_resets():
  runner, env = make_runner(seed=0)
  policy = runner.get_inference_policy()
  a1 = policy(env.get_observations())
  a2 = SuperMimicRunner.get_inference_policy(runner)(None)  # fresh history, same state
  torch.testing.assert_close(a1, a2)
  assert a1.shape == (env.num_envs, NJ) and not a1.requires_grad
  for _ in range(NUM_FRAMES + 5):  # crosses clip-end truncations and terminations
    _, _, dones, _ = env.step(policy(None))
    policy.reset(dones.nonzero().squeeze(-1))
  a = policy(None)
  assert torch.isfinite(a).all()


# ----- real G1 wiring (g1_components) without an mjlab env -------------------------------------


def _write_g1_motion(path, num_frames: int = 60) -> None:
  import numpy as np

  from mjlab_g1.rl.supermimic.kinematics import G1_DEFAULT_JOINT_POS, G1_KNEES_BENT_ROOT_POS, TorchFK

  fk = TorchFK.g1()
  t = torch.arange(num_frames, dtype=torch.float32)[:, None] * DT
  ph = torch.linspace(0.0, 3.0, NJ)[None]
  q = torch.tensor(G1_DEFAULT_JOINT_POS)[None] + 0.15 * torch.sin(2.0 * t + ph)
  qd = 0.3 * torch.cos(2.0 * t + ph)
  v = torch.tensor([0.3, 0.0, 0.0]).expand(num_frames, 3)
  root = torch.tensor(G1_KNEES_BENT_ROOT_POS)[None] + t * v
  quat = torch.tensor([1.0, 0.0, 0.0, 0.0]).expand(num_frames, 4)
  with torch.no_grad():
    kin = fk(torch.cat([root, quat, q], -1), torch.cat([v, torch.zeros(num_frames, 3), qd], -1))
  np.savez(
    path, joint_pos=q.numpy(), joint_vel=qd.numpy(), body_pos_w=kin.pos.numpy(),
    body_quat_w=kin.quat.numpy(), body_lin_vel_w=kin.lin_vel.numpy(),
    body_ang_vel_w=kin.ang_vel.numpy(), fps=np.array(50),
  )  # fmt: skip


class G1FakeEnv(FakeEnv):
  """Walk-task stand-in with the real G1 state layout [qpos 36, qvel 35] and the real analytic
  TrackingReward (so parity is exact): joints track q0 + 0.4 a, the root sinks under a
  friction-scaled pull (anchor-height terminations) and otherwise keeps its reset velocity."""

  def __init__(self, lib, num_envs: int = 4, seed: int = 0):
    from mjlab_g1.rl.supermimic.kinematics import G1_DEFAULT_JOINT_POS, TorchFK
    from mjlab_g1.rl.supermimic.reward import TrackingReward, TrackingRewardCfg

    self.fk = TorchFK.g1()
    self.q0 = torch.tensor(G1_DEFAULT_JOINT_POS)
    self.alpha = 0.4
    self._lib = lib
    self._rew = TrackingReward(TrackingRewardCfg.walk())
    super().__init__(num_envs, seed=seed)

  def _setup(self):
    self.motion_lib, self.reward_fn = self._lib, self._rew
    self.reward_term_names = list(self._rew.names)

  def _reset(self, ids: torch.Tensor) -> None:
    if not hasattr(self, "_g1_ready"):
      self._setup()
      self._s = torch.zeros(self.num_envs, 71)
      self._terms = torch.zeros(self.num_envs, len(self.reward_term_names))
      self._g1_ready = True
    lib, n = self.motion_lib, ids.numel()
    k0 = torch.randint(0, lib.num_frames - 10, (n,), generator=self.gen)
    noise = lambda *shape: torch.randn(*shape, generator=self.gen)  # noqa: E731
    self._s[ids] = torch.cat([
      lib.body_pos_w[k0, 0] + 0.02 * noise(n, 3), lib.body_quat_w[k0, 0],
      lib.joint_pos[k0] + 0.05 * noise(n, NJ), lib.body_lin_vel_w[k0, 0],
      lib.body_ang_vel_w[k0, 0], lib.joint_vel[k0] + 0.1 * noise(n, NJ),
    ], -1)  # fmt: skip
    self._xi[ids] = self.xi_lo + (self.xi_hi - self.xi_lo) * torch.rand(n, D, generator=self.gen)
    self._k[ids] = k0
    self._a_last[ids] = 0.0
    self.episode_length_buf[ids] = 0

  def step(self, a: torch.Tensor):
    s = self._s
    q, v, qd = s[:, 7:36], s[:, 36:39], s[:, 42:71]
    qd1 = 0.85 * qd + 25.0 * (self.q0 + self.alpha * a - q)
    v1 = v.clone()
    v1[:, 2] -= DT * 6.0 * self._xi[:, 0]
    s1 = torch.cat([s[:, :3] + DT * v1, s[:, 3:7], q + DT * qd1, v1, s[:, 39:42], qd1], -1)
    ref = self.motion_lib.frame(self._k)
    kin1 = self.fk(s1[:, :36], s1[:, 36:])
    r, terms = self.reward_fn(s1, kin1, ref, a, self._a_last, kin_prev=self.fk(s[:, :36], s[:, 36:]))
    term = self.reward_fn.terminated(kin1, ref, s1)
    trunc = (self._k >= self.motion_lib.clip_last) & ~term
    done = term | trunc
    self._terms = torch.stack([terms[n] for n in self.reward_term_names], dim=-1)
    self.pre_reset_state = s1.clone()
    self._s = s1
    self._a_last = a.clone()
    self.episode_length_buf += 1
    self._k += 1
    ids = done.nonzero().squeeze(-1)
    if ids.numel() > 0:
      self._reset(ids)
    return {}, r, done.long(), {"time_outs": trunc.clone()}


def test_g1_components_three_iterations(tmp_path):
  from mjlab_g1.rl.supermimic.algorithm import g1_components
  from mjlab_g1.rl.supermimic.reference import MotionLibrary

  _write_g1_motion(tmp_path / "walk.npz")
  lib = MotionLibrary(tmp_path / "walk.npz")
  env = G1FakeEnv(lib, num_envs=4)
  cfg = SuperMimicRunnerCfg.smoke(seed=0)
  torch.manual_seed(0)
  comps = g1_components(
    cfg, lib=lib, reward=env.reward_fn, xi_lo=env.xi_lo, xi_hi=env.xi_hi, num_envs=env.num_envs,
    action_dim=NJ, action_scale=env.alpha, num_reward_terms=len(env.reward_term_names),
    step_dt=DT, has_box=False, device="cpu",
  )  # fmt: skip
  # Critic t_left spans the whole clip (cfg.critic.t_left_horizon=None), not 64 steps.
  assert comps.imagine.keywords["t_left_horizon"] == lib.num_frames - 1
  runner = SuperMimicRunner(env, cfg, log_dir=str(tmp_path / "run"), device="cpu", components=comps)
  runner.learn(cfg.max_iterations)

  rows = [json.loads(x) for x in (tmp_path / "run" / "metrics.jsonl").read_text().splitlines()]
  assert len(rows) == 3
  assert rows[0]["parity/r_maxabs"] < 1e-5 and rows[0]["parity/term_mismatch"] == 0
  for row in rows:
    for key in ("wm/loss", "wm/loss_bce", "actor/J", "actor/grad_norm_preclip", "critic/loss"):
      assert math.isfinite(row[key]), (key, row[key])
  # Real dims: 71-d state, 930-d mjlab-layout history, 640-d goal.
  assert runner.alg.replay.data["s"].shape[-1] == 71
  assert runner.alg.actor.hist_dim == 930 and runner.alg.actor.goal_dim == 640
  a = runner.get_inference_policy()(None)
  assert a.shape == (env.num_envs, NJ) and torch.isfinite(a).all()


def test_compiled_paths_match_eager(tmp_path):
  """torch.compile of the imagined step and of the WM window loss is numerically a no-op."""
  from mjlab_g1.rl.supermimic import imagination
  from mjlab_g1.rl.supermimic.algorithm import g1_components
  from mjlab_g1.rl.supermimic.reference import MotionLibrary

  _write_g1_motion(tmp_path / "walk.npz")
  lib = MotionLibrary(tmp_path / "walk.npz")
  env = G1FakeEnv(lib, num_envs=4)
  cfg = SuperMimicRunnerCfg.smoke(seed=0)
  torch.manual_seed(0)
  comps = g1_components(
    cfg, lib=lib, reward=env.reward_fn, xi_lo=env.xi_lo, xi_hi=env.xi_hi, num_envs=env.num_envs,
    action_dim=NJ, action_scale=env.alpha, num_reward_terms=len(env.reward_term_names),
    step_dt=DT, has_box=False, device="cpu",
  )  # fmt: skip
  runner = SuperMimicRunner(env, cfg, log_dir=str(tmp_path / "run"), device="cpu", components=comps)
  runner.learn(1)  # fills the replay
  alg = runner.alg

  W = alg.replay.sample_windows(cfg.wm.batch_windows, cfg.wm.window_k)
  loss_eager, _ = alg.wm.window_loss(W)
  loss_comp, _ = torch.compile(alg.wm.window_loss, dynamic=False)(W)
  torch.testing.assert_close(loss_comp, loss_eager, rtol=1e-4, atol=1e-5)

  start = alg.replay.sample_starts(cfg.policy.num_imagined)
  p = cfg.policy
  args = (start, alg.actor, alg.wm, alg.reward, alg.critic, alg.lib, p.horizon, p.gamma)
  step_fn = torch.compile(imagination._imagine_step, dynamic=False)
  grads, returns = [], []
  for imagine in (comps.imagine, functools.partial(comps.imagine, step_fn=step_fn)):
    torch.manual_seed(1)  # same sensor-noise draws
    alg.actor.zero_grad(set_to_none=True)
    ro = imagine(*args)
    ro.J.sum().backward()
    returns.append(ro.J.detach())
    grads.append([q.grad.clone() for q in alg.actor.parameters() if q.grad is not None])
  torch.testing.assert_close(returns[1], returns[0], rtol=1e-4, atol=1e-6)
  assert len(grads[0]) == len(grads[1]) > 0
  # Fused kernels reassociate float32 sums, so compare the whole gradient vector by relative
  # norm (measured 1.4e-3 through the BPTT chain; different noise draws would be O(1)).
  ge = torch.cat([g.flatten() for g in grads[0]])
  gc = torch.cat([g.flatten() for g in grads[1]])
  assert ge.norm() > 0
  assert ((gc - ge).norm() / ge.norm()).item() < 5e-3
