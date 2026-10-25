"""SuperMimic: world-model fit, first-order BPTT actor step and critic fit (Alg. A.1).

Per iteration (driven by runner.py): collect -> wm_update() -> actor_update() -> critic_update().
The actor step runs before the critic fit, so it bootstraps from the previous iteration's
critic (Alg. A.1 lines 7-8).

The sibling modules are injected through `Components`, so the loop can be tested with toy
doubles. `build_components()` / `g1_components()` are the only places that know the real
constructors.

Shapes: N envs, S state (71 walk / 84 lift), A = 29 actions, B batch, H horizon. Rollout
tensors are time-major: g/phi/t_left/m/V are [H+1, B, ...], r/d/a are [H, B, ...].
"""

from __future__ import annotations

import dataclasses
import functools
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Callable, Iterable

import torch
from torch import nn

from mjlab_g1.rl.supermimic.config import SuperMimicRunnerCfg, as_runner_cfg

QPOS_DIM = 36  # [root_pos 3, root_quat_w 4, joint_pos 29] (brief §1)
QVEL_DIM = 35  # [root_lin_vel_w 3, root_ang_vel_w 3, joint_vel 29]


@dataclass
class Components:
  """Everything the algorithm drives, with the contract it relies on.

  world_model: nn.Module; window_loss(Window) -> (loss, metrics dict) and
    update_normalizers(s, s_next) (sim data only).
  actor: nn.Module; forward(hist [N, L*F], goal [N,G]) -> a [N,A], normalizing its inputs
    internally; update_normalizers(hist, goal). `hist` is PolicyObsHistory.get(), in the same
    layout imagination flattens to.
  critic: nn.Module; forward(phi, goal, xi_enc, t_left) -> [*]; update_normalizers(phi, goal).
  replay: add(dict of s, s_next, a_exec, a_prev, xi, r, r_terms, term, trunc, ep_step,
    ref_idx), sample_windows(B, k) -> Window, sample_starts(B) -> Start (history length and the
    clip-end start filter are replay constructor defaults).
  reward: __call__(s1, kin1, ref, a, a_prev, kin_prev=None) -> (r [*], {term: [*]}),
    terminated(kin1, ref, s1) -> bool [*], max_step_reward (float).
  fk: (qpos [*,36], qvel [*,35]) -> kin (opaque, handed to the feature functions).
  lib: MotionLibrary: frame(k), alive(k), num_frames.
  feats: namespace with obs_frame(s, a_prev, xi_raw) -> [*,F], goal(s, kin, lib, k, off) ->
    [*,G], wm_features(s, kin) -> [*,P], encode_xi(xi, lo, hi), sensor_noise(frame) -> η like
    frame, PolicyObsHistory(N, L, F, device=) with reset(ids, frame) / push(frame) / get().
  imagine: (start, actor, wm, reward, critic, lib, H, gamma) -> rollout with J [B] (carrying
    the actor graph) and detached g, phi, xi_enc, t_left, r, d, m, V, a (time-major).
  """

  world_model: nn.Module
  actor: nn.Module
  critic: nn.Module
  replay: Any
  reward: Any
  fk: Callable[[torch.Tensor, torch.Tensor], Any]
  lib: Any
  feats: Any
  imagine: Callable[..., Any]
  xi_lo: torch.Tensor
  xi_hi: torch.Tensor


@dataclass
class RolloutView:
  """Detached imagined rollout as the critic fit consumes it."""

  J: torch.Tensor  # [B]
  g: torch.Tensor  # [H+1, B, G]
  phi: torch.Tensor  # [H+1, B, P]
  xi_enc: torch.Tensor  # [B, X]
  t_left: torch.Tensor  # [H+1, B]
  r: torch.Tensor  # [H, B] r̂_{t+1}
  d: torch.Tensor  # [H, B] d̂_{t+1}
  m: torch.Tensor  # [H+1, B] alive mask of step t
  V: torch.Tensor  # [H+1, B] critic values before this iteration's fit
  a: torch.Tensor  # [H, B, A]
  r_terms: dict[str, torch.Tensor] = field(default_factory=dict)  # name -> [H, B]


class SuperMimic:
  """Holds WM, actor, critic, their three Adam optimizers and the shared cosine schedule."""

  def __init__(self, cfg: SuperMimicRunnerCfg | dict, comps: Components, device: str = "cpu"):
    self.cfg = as_runner_cfg(cfg)
    self.device = torch.device(device)
    self.comps = comps
    self.wm = comps.world_model
    self.actor = comps.actor
    self.critic = comps.critic
    self.replay = comps.replay
    self.reward = comps.reward
    self.fk = comps.fk
    self.lib = comps.lib
    self.feats = comps.feats
    self.xi_lo = comps.xi_lo.to(self.device)
    self.xi_hi = comps.xi_hi.to(self.device)
    self.r_max_per_step = float(comps.reward.max_step_reward)

    w, p, c = self.cfg.wm, self.cfg.policy, self.cfg.critic
    self.wm_opt = torch.optim.Adam(
      self.wm.parameters(), lr=w.learning_rate, betas=tuple(w.betas), eps=w.eps
    )
    self.actor_opt = torch.optim.Adam(
      self.actor.parameters(), lr=p.learning_rate, betas=tuple(p.betas), eps=p.eps
    )
    self.critic_opt = torch.optim.Adam(
      self.critic.parameters(), lr=c.learning_rate, betas=tuple(c.betas), eps=c.eps
    )
    self._opts = {"wm": self.wm_opt, "actor": self.actor_opt, "critic": self.critic_opt}
    self._base_lr = {"wm": w.learning_rate, "actor": p.learning_rate, "critic": c.learning_rate}

    self._imagine = comps.imagine
    self._wm_loss = self.wm.window_loss
    if self.cfg.memory.compile:
      # Compile the networks and the FK in place: state_dict keys and the optimizers'
      # parameters are unchanged. The running normalizers stay eager (they update state).
      for module in (
        self.wm.dynamics,
        self.wm.term_head,
        self.actor.encoder.net,
        self.actor.decoder.net,
        self.critic.net,
        self.fk,
      ):
        module.compile()
      # The two launch-bound hot spots (hundreds of small FK/quaternion/feature kernels per
      # step, measured 2.6 s per WM update eager): one imagined step, reused H times so the
      # BPTT loop is not unrolled into one graph, and the k-step world-model window loss.
      from mjlab_g1.rl.supermimic import imagination

      step_fn = torch.compile(imagination._imagine_step, dynamic=False)
      self._imagine = functools.partial(comps.imagine, step_fn=step_fn)
      self._wm_loss = torch.compile(self.wm.window_loss, dynamic=False)

    self.lr_factor = 1.0
    self.set_lr(0)

  @staticmethod
  def construct_algorithm(obs: Any, env: Any, cfg: SuperMimicRunnerCfg | dict, device: str):
    """rsl_rl-style factory; `obs` is unused (features are rebuilt from raw state)."""
    del obs
    cfg = as_runner_cfg(cfg)
    return SuperMimic(cfg, build_components(env, cfg, device), device)

  # Schedule.

  def set_lr(self, it: int) -> float:
    """Shared cosine schedule on all three optimizers (Table A.5); `it` is 0-based."""
    f = self.cfg.lr_factor(it)
    for name, opt in self._opts.items():
      for group in opt.param_groups:
        group["lr"] = self._base_lr[name] * f
    self.lr_factor = f
    return f

  # Features (thin adapters so the call conventions live in one place).

  def kin(self, s: torch.Tensor) -> Any:
    return self.fk(s[..., :QPOS_DIM], s[..., QPOS_DIM : QPOS_DIM + QVEL_DIM])

  def goal(self, s: torch.Tensor, kin: Any, k: torch.Tensor) -> torch.Tensor:
    return self.feats.goal(s, kin, self.lib, k, self.cfg.reward_ref_offset)

  def obs_frame(
    self, s: torch.Tensor, a_prev: torch.Tensor, xi: torch.Tensor, noise: bool
  ) -> torch.Tensor:
    f = self.feats.obs_frame(s, a_prev, xi)
    return f + self.feats.sensor_noise(f) if noise else f

  def xi_enc(self, xi: torch.Tensor) -> torch.Tensor:
    return self.feats.encode_xi(xi, self.xi_lo, self.xi_hi)

  def phi(self, s: torch.Tensor, kin: Any) -> torch.Tensor:
    return self.feats.wm_features(s, kin)

  # Collection.

  @torch.no_grad()
  def act(
    self,
    raw_state: torch.Tensor,
    hist: torch.Tensor,
    ref_index: torch.Tensor,
    noise_std: float | None = None,
    update_stats: bool = False,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    """Deterministic actor + N(0, σ²) per dimension (Table A.5). Returns (a_exec, a_det).

    `update_stats` feeds the actor/critic input normalizers with this sim data (sim data only,
    never imagined data). Inference passes noise_std=0 and update_stats=False.
    """
    kin = self.kin(raw_state)
    g = self.goal(raw_state, kin, ref_index)
    if update_stats:
      self.actor.update_normalizers(hist, g)
      self.critic.update_normalizers(self.phi(raw_state, kin), g)
    a = self.actor(hist, g)
    std = self.cfg.action_noise_std if noise_std is None else noise_std
    a_exec = a + std * torch.randn_like(a) if std > 0 else a
    return a_exec, a

  @torch.no_grad()
  def process_transition(self, tr: dict[str, torch.Tensor]) -> None:
    """Store one vectorized transition and update the WM's cumulative statistics."""
    self.replay.add(tr)
    self.wm.update_normalizers(tr["s"], tr["s_next"])

  # World model (Eq. 1).

  def wm_update(self) -> dict[str, torch.Tensor]:
    w = self.cfg.wm
    self.wm.train()
    out: dict[str, torch.Tensor] = {}
    loss_sum = torch.zeros((), device=self.device)
    gn_sum = torch.zeros((), device=self.device)
    skipped = 0
    for i in range(w.num_batches):
      W = self.replay.sample_windows(w.batch_windows, w.window_k)
      loss, metrics = self._wm_loss(W)
      self.wm_opt.zero_grad(set_to_none=True)
      loss.backward()
      gn, ok = _clip_and_guard(self.wm.parameters(), w.max_grad_norm)
      skipped += _step_if(self.wm_opt, ok)
      gn_sum += gn
      loss_sum += loss.detach()
      if i == 0:
        # First batch is scored before this iteration's updates (Fig. 3c-style error).
        for k, v in metrics.items():
          out[k if "/" in k else f"wm/{k}"] = _as_tensor(v, self.device)
    out["wm/loss"] = loss_sum / w.num_batches
    out["wm/grad_norm"] = gn_sum / w.num_batches
    out["wm/skipped_steps"] = torch.tensor(float(skipped), device=self.device)
    return out

  # Actor (Eq. 2).

  def actor_update(self) -> tuple[RolloutView, dict[str, torch.Tensor]]:
    """One BPTT step through the frozen WM and critic. Returns (detached rollout, metrics)."""
    p = self.cfg.policy
    start = self.replay.sample_starts(p.num_imagined)
    n_chunks = max(1, int(self.cfg.memory.imag_microbatches))
    self.actor_opt.zero_grad(set_to_none=True)
    views: list[RolloutView] = []
    # Freeze params only: gradients still flow through WM/critic inputs into the actor.
    restore = _freeze(self.wm, self.critic)
    try:
      for chunk in _split_batch(start, n_chunks, p.num_imagined):
        ro = self._imagine(
          chunk, self.actor, self.wm, self.reward, self.critic, self.lib, p.horizon, p.gamma
        )
        # SHAC reduction -mean_batch(J)/H; exact under micro-batching (sum / total B).
        loss = -ro.J.sum() / (p.num_imagined * p.horizon)
        loss.backward()
        views.append(_view(ro))
    finally:
      restore()

    # Table A.5 order: global-norm clip first, then element-wise clip.
    params = [q for q in self.actor.parameters() if q.grad is not None]
    gn, ok = _clip_and_guard(params, p.max_grad_norm)
    n_el = sum(q.grad.numel() for q in params)
    n_big = sum((q.grad.abs() > p.elementwise_clip).sum() for q in params)
    for q in params:
      q.grad.clamp_(-p.elementwise_clip, p.elementwise_clip)
    skipped = _step_if(self.actor_opt, ok)

    ro = _cat_views(views)
    m = ro.m[:-1]
    denom = m.sum().clamp(min=1.0)
    a = ro.a.abs().flatten()
    if a.numel() > 1_000_000:  # quantile() input-size limit; a subsample is enough
      a = a[torch.randint(a.numel(), (1_000_000,), device=a.device)]
    q = torch.quantile(a, torch.tensor([0.5, 0.99], device=a.device, dtype=a.dtype))
    metrics = {
      "actor/J": ro.J.mean(),
      "actor/loss": -ro.J.mean() / p.horizon,
      "actor/grad_norm_preclip": gn,
      "actor/frac_elem_clipped": torch.as_tensor(n_big, device=self.device).float() / max(n_el, 1),
      "actor/skipped_steps": torch.tensor(float(skipped), device=self.device),
      "imag/r_mean": (ro.r * m).sum() / denom,
      "imag/d_mean": (ro.d * m).sum() / denom,
      "imag/c_H": torch.prod(1.0 - ro.d, dim=0).mean(),
      "imag/alive_frac": m.mean(),
      "imag/|a|_p50": q[0],
      "imag/|a|_p99": q[1],
    }
    for name, v in ro.r_terms.items():
      metrics[f"imag/r_{name}"] = (v * m).sum() / denom
    return ro, metrics

  # Critic (Eq. 3).

  def critic_update(self, ro: RolloutView) -> dict[str, torch.Tensor]:
    c, p = self.cfg.critic, self.cfg.policy
    H, B = ro.r.shape
    with torch.no_grad():
      R = lambda_returns(
        ro.r, ro.d, ro.m, ro.V, p.gamma, c.lam, bootstrap=self.cfg.clip_end == "bootstrap"
      )
      w = ro.m[:-1]
      ev = _explained_variance(R, ro.V[:-1], w)
    n = H * B
    phi = ro.phi[:H].reshape(n, -1)
    g = ro.g[:H].reshape(n, -1)
    xe = ro.xi_enc.unsqueeze(0).expand(H, *ro.xi_enc.shape).reshape(n, -1)
    tl = ro.t_left[:H].reshape(n)
    R, w = R.reshape(n), w.reshape(n)
    loss_sum = torch.zeros((), device=self.device)
    gn_sum = torch.zeros((), device=self.device)
    steps = skipped = 0
    for _ in range(c.num_epochs):
      for idx in torch.randperm(n, device=self.device).chunk(c.num_mini_batches):
        pred = self.critic(phi[idx], g[idx], xe[idx], tl[idx]).reshape(-1)
        wi = w[idx]
        loss = (wi * (pred - R[idx]).square()).sum() / wi.sum().clamp(min=1.0)
        self.critic_opt.zero_grad(set_to_none=True)
        loss.backward()
        gn, ok = _clip_and_guard(self.critic.parameters(), c.max_grad_norm)
        skipped += _step_if(self.critic_opt, ok)
        gn_sum += gn
        loss_sum += loss.detach()
        steps += 1
    return {
      "critic/loss": loss_sum / max(steps, 1),
      "critic/grad_norm": gn_sum / max(steps, 1),
      "critic/skipped_steps": torch.tensor(float(skipped), device=self.device),
      "critic/expl_var": ev,
      "critic/V_mean": (ro.V[:-1] * ro.m[:-1]).sum() / ro.m[:-1].sum().clamp(min=1.0),
      "critic/target_mean": (R * w).sum() / w.sum().clamp(min=1.0),
    }

  # Checkpointing.

  def save(self) -> dict:
    return {
      "actor_state_dict": self.actor.state_dict(),
      "critic_state_dict": self.critic.state_dict(),
      "world_model_state_dict": self.wm.state_dict(),
      "optimizer_state_dict": self.actor_opt.state_dict(),  # PPO-compatible key = actor's
      "critic_optimizer_state_dict": self.critic_opt.state_dict(),
      "wm_optimizer_state_dict": self.wm_opt.state_dict(),
      "lr_factor": self.lr_factor,
      "cfg": self.cfg.to_dict(),
    }

  def load(self, d: dict, load_cfg: dict | None = None, strict: bool = True) -> bool:
    """PPO.load semantics: None loads everything; otherwise only the keys set True.

    Keys: actor, critic, world_model, optimizer (all three), iteration. Returns whether the
    caller should restore the iteration counter.
    """
    if load_cfg is None:
      load_cfg = {
        "actor": True, "critic": True, "world_model": True, "optimizer": True, "iteration": True
      }  # fmt: skip
    if load_cfg.get("actor"):
      self.actor.load_state_dict(d["actor_state_dict"], strict=strict)
    if load_cfg.get("critic"):
      self.critic.load_state_dict(d["critic_state_dict"], strict=strict)
    if load_cfg.get("world_model"):
      self.wm.load_state_dict(d["world_model_state_dict"], strict=strict)
    if load_cfg.get("optimizer"):
      self.actor_opt.load_state_dict(d["optimizer_state_dict"])
      if "critic_optimizer_state_dict" in d:
        self.critic_opt.load_state_dict(d["critic_optimizer_state_dict"])
      if "wm_optimizer_state_dict" in d:
        self.wm_opt.load_state_dict(d["wm_optimizer_state_dict"])
    return bool(load_cfg.get("iteration", False))

  def train_mode(self) -> None:
    for mod in (self.wm, self.actor, self.critic):
      mod.train()

  def eval_mode(self) -> None:
    for mod in (self.wm, self.actor, self.critic):
      mod.eval()


def lambda_returns(
  r: torch.Tensor,
  d: torch.Tensor,
  m: torch.Tensor,
  V: torch.Tensor,
  gamma: float,
  lam: float,
  bootstrap: bool = False,
) -> torch.Tensor:
  """Eq. 3 targets on a detached rollout; r, d: [H,B] (transition t -> t+1), m, V: [H+1,B].

  stop (design D4): R_H = V_H, R_t = r_t + γ(1 − d_t) m_{t+1} [(1 − λ)V_{t+1} + λR_{t+1}].
  bootstrap: no m_{t+1} cut, and R_t = V_t for states past the clip end (imagination.py's
  critic_targets, which this must match).
  """
  H = r.shape[0]
  R = torch.empty_like(r)
  nxt = V[H]
  for t in reversed(range(H)):
    if bootstrap:
      val = r[t] + gamma * (1.0 - d[t]) * ((1.0 - lam) * V[t + 1] + lam * nxt)
      val = m[t] * val + (1.0 - m[t]) * V[t]
    else:
      val = r[t] + gamma * (1.0 - d[t]) * m[t + 1] * ((1.0 - lam) * V[t + 1] + lam * nxt)
    R[t] = val
    nxt = val
  return R


# Real wiring.


def build_components(env: Any, cfg: SuperMimicRunnerCfg, device: str) -> Components:
  """Read the SMTrackEnv contract (brief §2) and wire the real G1 modules."""
  from mjlab_g1.rl.supermimic import reward as reward_mod
  from mjlab_g1.rl.supermimic.reference import MotionLibrary

  u = env.unwrapped
  dev = torch.device(device)
  has_box = bool(u.has_box)
  lib = u.motion_lib
  if lib.has_box != has_box:
    raise ValueError(f"motion library has_box={lib.has_box} but env has_box={has_box}")
  if torch.device(lib.device) != dev:
    lib = MotionLibrary(lib.path, dev)  # MotionLibrary.to() is in place: keep the env's copy
  rew = reward_mod.TrackingReward(reward_mod.TrackingRewardCfg.from_env(u), device=dev)
  names = list(u.reward_term_names)
  extra = [n for n in rew.names if n not in names]
  if extra:
    raise ValueError(f"analytic reward terms {extra} are not env terms {names}")
  missing = [n for n in names if n not in rew.names]
  if missing:  # from_env drops zero-weight terms it has no formula for; parity checks the total
    print(f"[SuperMimic] env reward terms without an analytic counterpart: {missing}")
  am = u.action_manager
  scale = am.get_term(am.active_terms[0]).scale
  scale = scale[0] if isinstance(scale, torch.Tensor) and scale.dim() == 2 else scale
  return g1_components(
    cfg,
    lib=lib,
    reward=rew,
    xi_lo=u.xi_lo,
    xi_hi=u.xi_hi,
    num_envs=env.num_envs,
    action_dim=env.num_actions,
    action_scale=scale,
    num_reward_terms=len(names),
    step_dt=float(u.step_dt),
    has_box=has_box,
    device=device,
  )


def g1_components(
  cfg: SuperMimicRunnerCfg,
  *,
  lib: Any,
  reward: Any,
  xi_lo: torch.Tensor,
  xi_hi: torch.Tensor,
  num_envs: int,
  action_dim: int,
  action_scale: torch.Tensor | float,
  num_reward_terms: int,
  step_dt: float,
  has_box: bool,
  device: str,
) -> Components:
  """Real G1 modules from explicit inputs (no env access, so tests can build it directly)."""
  from mjlab_g1.rl.supermimic import actor_critic, features, imagination, kinematics
  from mjlab_g1.rl.supermimic import replay as replay_mod
  from mjlab_g1.rl.supermimic import world_model as wm_mod

  dev = torch.device(device)
  off = cfg.reward_ref_offset
  K, L, A = cfg.policy.future_frames, cfg.policy.obs_history, action_dim
  xi_lo = torch.as_tensor(xi_lo).to(dev, torch.float32)
  xi_hi = torch.as_tensor(xi_hi).to(dev, torch.float32)
  S, D = features.STATE_DIM[has_box], xi_lo.numel()
  F, G = features.obs_frame_dim(has_box), features.goal_dim(has_box, K)
  P = features.WM_FEATURE_DIM[has_box]
  fk = kinematics.TorchFK.g1(device=dev)
  # One history layout for sim, imagination and the env's actor_hist group (mjlab term-major).
  layout = "mjlab"
  fns = imagination.ImagFns.g1(fk, xi_lo, xi_hi, has_box, K=K, hist_layout=layout)
  feats = SimpleNamespace(
    obs_frame=fns.obs_frame,
    goal=fns.goal,
    wm_features=fns.wm_features,
    encode_xi=features.encode_xi,
    sensor_noise=fns.sensor_noise,
    PolicyObsHistory=functools.partial(features.PolicyObsHistory, layout=layout),
  )

  def kin_fn(s: torch.Tensor) -> Any:
    return fk(s[..., :QPOS_DIM], s[..., QPOS_DIM : QPOS_DIM + QVEL_DIM])

  def term_feats(s: torch.Tensor, kin: Any, ref: Any) -> torch.Tensor:
    # BCE head extras (design §3.2): analytic e_pz, e_tilt (, e_box) and the ref anchor height.
    anchor_z = ref.body_pos_w[..., kinematics.ANCHOR_BODY_ID, 2]
    return torch.stack([*reward.term_errors(kin, ref, s), anchor_z], dim=-1)

  q0 = torch.tensor(kinematics.G1_DEFAULT_JOINT_POS, device=dev)
  alpha = torch.as_tensor(action_scale, dtype=torch.float32, device=dev).expand(A)
  wm = wm_mod.WorldModel.from_cfg(
    cfg.wm,
    wm_mod.StateSpec.g1(has_box),
    fns.wm_features,
    P,
    D,
    action_dim=A,
    kin_fn=kin_fn,
    term_extra_fn=term_feats,
    term_extra_dim=3 + int(has_box),
    ref_fn=lambda ref_idx: lib.frame(ref_idx + off),  # same frame the reward/termination use
    action_scale=wm_mod.motion_action_scale(lib.joint_pos, q0, alpha),
    xi_lo=xi_lo,
    xi_hi=xi_hi,
    dt=step_dt,
  ).to(dev)
  actor = actor_critic.Actor.from_cfg(cfg.policy, L * F, G, A).to(dev)
  critic = actor_critic.Critic.from_cfg(cfg.critic, P, G, D).to(dev)
  buf = replay_mod.SequenceReplay(
    num_envs,
    cfg.replay_steps_per_env,
    replay_mod.ReplaySpec(S, A, D, num_reward_terms=num_reward_terms),
    dev,
    hist_len=L,
    ref_valid=lambda ref_idx: lib.alive(ref_idx + off),  # first imagined reward must exist
  )
  t_left_h = cfg.critic.t_left_horizon or max(1, int(lib.num_frames) - 1)  # None: whole clip
  imag = imagination.make_imagine(
    fns, ref_offset=off, clip_end=cfg.clip_end, t_left_horizon=t_left_h, hist_len=L
  )
  return Components(
    world_model=wm,
    actor=actor,
    critic=critic,
    replay=buf,
    reward=reward,
    fk=fk,
    lib=lib,
    feats=feats,
    imagine=imag,
    xi_lo=xi_lo,
    xi_hi=xi_hi,
  )


# Helpers.


def _clip_and_guard(params: Iterable[nn.Parameter], max_norm: float) -> tuple[torch.Tensor, bool]:
  """Global-norm clip -> (pre-clip norm, finite?). A finite norm implies every grad is finite."""
  params = [q for q in params if q.grad is not None]
  if not params:
    return torch.zeros(()), False
  gn = torch.nn.utils.clip_grad_norm_(params, max_norm)
  # One host sync per update. Zeroing a NaN gradient is not enough: Adam's momentum would still
  # move the weights, so the caller skips the step instead.
  return gn.detach(), bool(torch.isfinite(gn))


def _step_if(opt: torch.optim.Optimizer, ok: bool) -> int:
  """opt.step() on finite gradients; otherwise drop them. Returns 1 if the step was skipped."""
  if ok:
    opt.step()
    return 0
  opt.zero_grad(set_to_none=True)
  return 1


def _freeze(*mods: nn.Module) -> Callable[[], None]:
  saved = [(q, q.requires_grad) for mod in mods for q in mod.parameters()]
  for q, _ in saved:
    q.requires_grad_(False)

  def restore() -> None:
    for q, flag in saved:
      q.requires_grad_(flag)

  return restore


def _as_tensor(v: Any, device: torch.device) -> torch.Tensor:
  if isinstance(v, torch.Tensor):
    return v.detach().float().mean()
  return torch.tensor(float(v), device=device)


def _explained_variance(target: torch.Tensor, pred: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
  n = w.sum().clamp(min=1.0)

  def var(x: torch.Tensor) -> torch.Tensor:
    mu = (x * w).sum() / n
    return ((x - mu).square() * w).sum() / n

  return 1.0 - var(target - pred) / var(target).clamp(min=1e-8)


def _fields(obj: Any) -> dict[str, Any]:
  if dataclasses.is_dataclass(obj):
    return {f.name: getattr(obj, f.name) for f in dataclasses.fields(obj)}
  if isinstance(obj, dict):
    return dict(obj)
  return dict(vars(obj))


def _rebuild(obj: Any, values: dict[str, Any]) -> Any:
  if dataclasses.is_dataclass(obj):
    return dataclasses.replace(obj, **values)
  if isinstance(obj, dict):
    return values
  return SimpleNamespace(**values)


def _split_batch(start: Any, n: int, batch: int) -> list[Any]:
  """Split a batch-first Start into n micro-batches (tensors whose leading dim == batch)."""
  if n <= 1:
    return [start]
  vals = _fields(start)
  out = []
  for ix in torch.arange(batch).chunk(n):
    sub = {
      k: (v[ix.to(v.device)] if isinstance(v, torch.Tensor) and v.shape[:1] == (batch,) else v)
      for k, v in vals.items()
    }
    out.append(_rebuild(start, sub))
  return out


def _view(ro: Any) -> RolloutView:
  terms = getattr(ro, "r_terms", None) or {}
  return RolloutView(
    J=ro.J.detach(),
    g=ro.g.detach(),
    phi=ro.phi.detach(),
    xi_enc=ro.xi_enc.detach(),
    t_left=ro.t_left.detach().to(ro.r.dtype),
    r=ro.r.detach(),
    d=ro.d.detach(),
    m=ro.m.detach().to(ro.r.dtype),
    V=ro.V.detach(),
    a=ro.a.detach(),
    r_terms={k: v.detach() for k, v in terms.items()},
  )


def _cat_views(views: list[RolloutView]) -> RolloutView:
  if len(views) == 1:
    return views[0]

  def cat(name: str, dim: int) -> torch.Tensor:
    return torch.cat([getattr(v, name) for v in views], dim)

  return RolloutView(
    J=cat("J", 0),
    xi_enc=cat("xi_enc", 0),
    **{k: cat(k, 1) for k in ("g", "phi", "t_left", "r", "d", "m", "V", "a")},
    r_terms={k: torch.cat([v.r_terms[k] for v in views], 1) for k in views[0].r_terms},
  )
