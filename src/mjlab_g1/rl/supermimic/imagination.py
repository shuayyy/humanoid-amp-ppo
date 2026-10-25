"""Imagined BPTT rollouts through the world model (Eq. 2) and the critic's λ-targets (Eq. 3).

Per step t (design §4.1), with k_t the reward frame index (mjlab pairing: reward of s_{t+1} uses
ref[k_t + offset]):
  g_t = goal(ŝ_t)            a_t = π(o_hist + η, g_t)        ŝ_{t+1} = f(ŝ_t, a_t, ξ)
  r_t = r(ŝ_{t+1}, ref, a_t, a_{t-1})    d_t = d̂(ŝ_{t+1})    m_t = alive(k_t)
  J  += γ^t c_t m_t r_t ;  c_{t+1} = c_t (1 - d_t)
  J  += γ^H c_H m_H V(ŝ_H)          (m: 0 once the clip has ended -> no reward, no bootstrap)

WM and critic parameters are frozen while the graph is recorded, so the actor loss never writes
their .grad; gradients still flow through their inputs (state, action).

The actor sees the observation history flattened by `ImagFns.flatten_hist`, which must be the
layout the sim feeds it (features.PolicyObsHistory: mjlab term-major by default). The start
history gets fresh η per frame, except padded copies of an episode's first frame, which share one
draw as in the sim (see `start_history_noise`).

φ(ŝ) is computed once per imagined state and handed to the WM's step/term_prob when the WM uses
the same φ function (as algorithm.g1_components wires it); otherwise the WM computes its own.
"""

from __future__ import annotations

import dataclasses
import functools
import inspect
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Iterator, Literal

import torch
from torch import nn

NQ, NV = 36, 35  # robot qpos / qvel widths in s (brief §1)


@dataclass
class ImagFns:
  """Feature callables imagination needs (design §6 signatures; see `g1` for features.py)."""

  fk: Callable[[torch.Tensor, torch.Tensor], Any]  # (qpos [*,36], qvel [*,35]) -> BodyKin
  goal: Callable[[torch.Tensor, Any, Any, torch.Tensor, int], torch.Tensor]  # (s, kin, lib, k, off)
  obs_frame: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]  # (s, a_prev, ξ raw)
  wm_features: Callable[[torch.Tensor, Any], torch.Tensor]  # (s, kin) -> φ
  encode_xi: Callable[[torch.Tensor], torch.Tensor]  # raw ξ -> [-1, 1] encoding
  sensor_noise: Callable[[torch.Tensor], torch.Tensor]  # frames [*, F] -> fresh η, same shape
  flatten_hist: Callable[[torch.Tensor], torch.Tensor] = lambda h: h.flatten(-2)  # [B,L,F] -> actor

  @classmethod
  def g1(
    cls,
    fk: Callable[..., Any],
    xi_lo: torch.Tensor,
    xi_hi: torch.Tensor,
    has_box: bool,
    K: int = 10,
    hist_layout: str = "mjlab",
  ) -> "ImagFns":
    """Bind the real features.py functions (their has_box / K / layout arguments)."""
    from mjlab_g1.rl.supermimic import features as F

    def encode_xi(xi: torch.Tensor) -> torch.Tensor:
      return F.encode_xi(xi, xi_lo.to(xi), xi_hi.to(xi))

    return cls(
      fk=fk,
      goal=lambda s, kin, lib, k, off: F.goal(s, kin, lib, k, K=K, offset=off, has_box=has_box),
      obs_frame=lambda s, a_prev, xi: F.obs_frame(s, None, a_prev, xi, has_box),
      wm_features=lambda s, kin: F.wm_features(s, kin, has_box),
      encode_xi=encode_xi,
      sensor_noise=lambda x: F.sample_obs_noise(x.shape[:-1], has_box, x.device, x.dtype),
      flatten_hist=lambda h: F.flatten_history(h, None, hist_layout),
    )


@dataclass
class Rollout:
  """One imagined batch. J carries the actor graph; everything else is detached (critic data)."""

  J: torch.Tensor  # [B]
  s: torch.Tensor  # [H+1, B, S] imagined states ŝ_t
  g: torch.Tensor  # [H+1, B, G] raw goal g_t
  phi: torch.Tensor  # [H+1, B, P] raw φ(ŝ_t)
  xi_enc: torch.Tensor  # [B, X]
  t_left: torch.Tensor  # [H+1, B]
  r: torch.Tensor  # [H, B] reward of transition t
  d: torch.Tensor  # [H, B] termination prob. of ŝ_{t+1}
  m: torch.Tensor  # [H+1, B] alive(k_t): 1 while the reward frame exists
  c: torch.Tensor  # [H+1, B] survival weight before step t
  V: torch.Tensor  # [H+1, B] critic values (no grad)
  a: torch.Tensor  # [H, B, A]
  r_terms: dict[str, torch.Tensor] = field(default_factory=dict)  # name -> [H, B]
  gamma: float = 0.99
  bootstrap_clip_end: bool = False

  @property
  def horizon(self) -> int:
    return self.r.shape[0]

  def actor_loss(self) -> torch.Tensor:
    """-mean_batch(J) / H (SHAC Eq. 5 reduction; ours)."""
    return -self.J.mean() / self.horizon

  def detached(self) -> "Rollout":
    return replace(self, J=self.J.detach())


@contextmanager
def frozen(*modules: nn.Module | None) -> Iterator[None]:
  """requires_grad_(False) on the modules' parameters for the duration (restored after).

  Ops recorded while frozen treat the parameters as constants, so restoring the flag before
  backward() is safe: no AccumulateGrad node exists for them in that graph.
  """
  params = [p for m in modules if isinstance(m, nn.Module) for p in m.parameters()]
  flags = [p.requires_grad for p in params]
  for p in params:
    p.requires_grad_(False)
  try:
    yield
  finally:
    for p, f in zip(params, flags, strict=True):
      p.requires_grad_(f)


def make_imagine(fns: ImagFns, **kw: Any) -> Callable[..., "Rollout"]:
  """imagine with fns / ref_offset / clip_end / ... bound: the 8-positional-arg callable
  algorithm.Components.imagine expects (start, actor, wm, reward, critic, lib, H, gamma)."""
  return functools.partial(imagine, fns=fns, **kw)


def _get(obj: Any, name: str) -> Any:
  return obj.get(name) if isinstance(obj, dict) else getattr(obj, name, None)


def start_history_noise(start: Any, hist: torch.Tensor, fns: ImagFns) -> torch.Tensor:
  """η for the start history [B, L, F]: fresh per frame, except that the padded copies of an
  episode's first frame share one draw. The sim fills the whole buffer with the first NOISY
  frame at reset (features.PolicyObsHistory.reset), so its padded frames are identical."""
  eta = fns.sensor_noise(hist)
  slots = _get(start, "hist_slots")
  if not isinstance(slots, torch.Tensor) or slots.shape != hist.shape[:2]:
    return eta
  # First index holding the same replay slot (argmax returns the first True): padding is a
  # prefix of duplicates of the episode's first frame.
  src = (slots[:, :, None] == slots[:, None, :]).to(torch.uint8).argmax(-1)
  return eta.gather(1, src[..., None].expand_as(eta))


def start_history(start: Any, fns: ImagFns, hist_len: int = 10) -> torch.Tensor:
  """Clean obs history [B, L, F] (oldest -> newest) of a start batch.

  Either given directly (`hist`, [B,L,F] or frame-major [B,L*F]) or rebuilt from the replay's
  per-frame inputs `hist_s` [B,L,S] / `hist_a_prev` [B,L,A] (replay.Start).
  """
  hist = _get(start, "hist")
  if hist is not None:
    return hist if hist.dim() == 3 else hist.reshape(hist.shape[0], hist_len, -1)
  hs, ha, xi = _get(start, "hist_s"), _get(start, "hist_a_prev"), _get(start, "xi")
  if hs is None or ha is None:
    raise AttributeError("start batch needs `hist` or `hist_s` + `hist_a_prev`")
  return fns.obs_frame(hs, ha, xi[:, None].expand(-1, hs.shape[1], -1))


def _alive(lib: Any, k: torch.Tensor) -> torch.Tensor:
  alive = lib.alive(k) if hasattr(lib, "alive") else k <= lib.num_frames - 1
  return alive.float()


def _t_left(lib: Any, k: torch.Tensor, horizon: int) -> torch.Tensor:
  if hasattr(lib, "t_left"):
    return lib.t_left(k, horizon).float()
  return (lib.num_frames - 1 - k).clamp(0, horizon).float() / horizon


_SIG_CACHE: dict[tuple[type, str, str], bool] = {}


def _takes(obj: Any, method: str, arg: str) -> bool:
  key = (type(obj), method, arg)
  if key not in _SIG_CACHE:
    try:
      _SIG_CACHE[key] = arg in inspect.signature(getattr(obj, method)).parameters
    except (TypeError, ValueError):
      _SIG_CACHE[key] = False
  return _SIG_CACHE[key]


def _shares_phi(wm: Any, fns: ImagFns) -> bool:
  """True when the WM's φ is the very function imagination uses (algorithm.g1_components builds
  both from fns.wm_features), so the φ computed here can be handed to the WM unchanged."""
  wm_fns = getattr(wm, "_fns", None)
  return isinstance(wm_fns, dict) and wm_fns.get("phi") is fns.wm_features


def _wm_kw(wm: Any, method: str, kin: Any, phi: torch.Tensor | None) -> dict[str, Any]:
  # world_model.WorldModel takes the FK (and φ) we already have instead of recomputing them.
  kw: dict[str, Any] = {}
  if _takes(wm, method, "kin"):
    kw["kin"] = kin
  if phi is not None and _takes(wm, method, "phi"):
    kw["phi"] = phi
  return kw


def _wm_step(
  wm: Any, s: torch.Tensor, a: torch.Tensor, xi_enc: torch.Tensor, kin: Any, phi: torch.Tensor | None
) -> torch.Tensor:
  return wm.step(s, a, xi_enc, **_wm_kw(wm, "step", kin, phi))


def _wm_term_prob(
  wm: Any, s1: torch.Tensor, kin1: Any, ref: Any, k: torch.Tensor, phi1: torch.Tensor | None
) -> torch.Tensor:
  # Map the buffer-style index exactly as WorldModel.window_loss does (ref_from_idx applies the
  # reward offset), so imagined d̂ sees the termination features it was trained on. A WM built
  # without ref_fn hands the raw index back: use the reward's frame then.
  wref = wm.ref_from_idx(k) if hasattr(wm, "ref_from_idx") else ref
  if wref is k:
    wref = ref
  return wm.term_prob(s1, wref, **_wm_kw(wm, "term_prob", kin1, phi1))


def _kin_split(kin: Any) -> tuple[type | None, Any]:
  """Dataclass kinematics -> (type, field tuple) so they can cross a torch.compile boundary."""
  if dataclasses.is_dataclass(kin) and not isinstance(kin, type):
    return type(kin), tuple(getattr(kin, f.name) for f in dataclasses.fields(kin))
  return None, kin


def _kin_join(kin_type: type | None, fields: Any) -> Any:
  return fields if kin_type is None else kin_type(*fields)


def _imagine_step(
  s: torch.Tensor,
  kin_fields: Any,
  phi: torch.Tensor,
  hist: torch.Tensor,
  a_prev: torch.Tensor,
  k: torch.Tensor,
  xi: torch.Tensor,
  xi_enc: torch.Tensor,
  *,
  actor: nn.Module,
  wm: Any,
  reward: Callable[..., tuple[torch.Tensor, dict[str, torch.Tensor]]],
  lib: Any,
  fns: ImagFns,
  ref_offset: int,
  obs_noise: bool,
  share_phi: bool,
  kin_type: type | None,
) -> tuple:
  """One imagined step: goal -> action -> ŝ_{t+1} -> reward, termination, next history.

  The unit `torch.compile` fuses (the H-step loop stays in Python, so one compiled graph is
  reused H times instead of unrolling BPTT). Kinematics cross the boundary as a tuple of
  tensors (`_kin_split`), so no user object has to be reconstructed from the graph.
  """
  kin = _kin_join(kin_type, kin_fields)
  g = fns.goal(s, kin, lib, k, ref_offset)
  a = actor(fns.flatten_hist(hist), g)  # deterministic: no action noise in imagination
  s1 = _wm_step(wm, s, a, xi_enc, kin, phi if share_phi else None)
  kin1 = fns.fk(s1[..., :NQ], s1[..., NQ : NQ + NV])
  phi1 = fns.wm_features(s1, kin1)
  ref = lib.frame(k + ref_offset)
  r, terms = reward(s1, kin1, ref, a, a_prev, kin_prev=kin)
  d = _wm_term_prob(wm, s1, kin1, ref, k, phi1 if share_phi else None)
  frame = fns.obs_frame(s1, a, xi)
  if obs_noise:
    frame = frame + fns.sensor_noise(frame)
  hist1 = torch.cat((hist[:, 1:], frame[:, None]), dim=1)
  return g, a, s1, _kin_split(kin1)[1], phi1, r, terms, d, hist1


def imagine(
  start: Any,
  actor: nn.Module,
  wm: Any,
  reward: Callable[..., tuple[torch.Tensor, dict[str, torch.Tensor]]],
  critic: nn.Module,
  lib: Any,
  H: int = 32,
  gamma: float = 0.99,
  *,
  fns: ImagFns,
  ref_offset: int = 0,
  clip_end: Literal["stop", "bootstrap"] = "stop",
  obs_noise: bool = True,
  t_left_horizon: int = 64,
  hist_len: int = 10,
  step_fn: Callable[..., tuple] | None = None,
) -> Rollout:
  """Roll H steps from replay starts: s [B,S], a_prev [B,A], xi [B,D] (raw), ref_idx [B] and the
  clean history (see `start_history`). Every start tensor is detached here.

  clip_end="stop": continuation 0 once the reference ends (design D4). "bootstrap": the first
  state past the clip end is valued with V (truncation bootstrap, spec §9 item 14).
  """
  if clip_end not in ("stop", "bootstrap"):
    raise ValueError(f"clip_end must be 'stop' or 'bootstrap', got {clip_end!r}")
  bootstrap = clip_end == "bootstrap"
  # Buffer tensors are data: never let a gradient reach the replay or the env.
  s = _get(start, "s").detach()
  a_prev = _get(start, "a_prev").detach()
  xi = _get(start, "xi").detach()
  k = _get(start, "ref_idx").detach().long()
  hist = start_history(start, fns, hist_len).detach()
  B = s.shape[0]
  xi_enc = fns.encode_xi(xi).detach()
  if obs_noise:
    hist = hist + start_history_noise(start, hist, fns)  # η re-added to the clean history (§3.3)
  share_phi = _shares_phi(wm, fns)
  step = _imagine_step if step_fn is None else step_fn
  step_kw = dict(
    actor=actor,
    wm=wm,
    reward=reward,
    lib=lib,
    fns=fns,
    ref_offset=ref_offset,
    obs_noise=obs_noise,
    share_phi=share_phi,
  )

  c = torch.ones(B, dtype=s.dtype, device=s.device)
  J = torch.zeros(B, dtype=s.dtype, device=s.device)
  states, phis, goals, t_lefts, rs, ds, ms, cs, acts = [], [], [], [], [], [], [], [], []
  r_terms: dict[str, list[torch.Tensor]] = {}

  with frozen(wm, critic):
    kin = fns.fk(s[..., :NQ], s[..., NQ : NQ + NV])
    phi = fns.wm_features(s, kin)  # φ(ŝ_t): critic input, and the WM's own input when shared
    m_prev = None
    for t in range(H):
      m = _alive(lib, k + ref_offset)
      tl = _t_left(lib, k + ref_offset, t_left_horizon)
      kin_type, kin_fields = _kin_split(kin)
      g, a, s1, kin1_fields, phi1, r, terms, d, hist = step(
        s, kin_fields, phi, hist, a_prev, k, xi, xi_enc, kin_type=kin_type, **step_kw
      )
      kin1 = _kin_join(kin_type, kin1_fields)
      if bootstrap and m_prev is not None:
        # truncation: first state whose reward frame no longer exists
        b = m_prev * (1.0 - m)
        J = J + (gamma**t) * c * b * critic(phi, g, xi_enc, tl)
      states.append(s.detach())
      phis.append(phi.detach())
      goals.append(g.detach())
      ms.append(m)
      cs.append(c.detach())
      t_lefts.append(tl)

      J = J + (gamma**t) * c * m * r
      c = c * (1.0 - d)

      rs.append(r.detach())
      ds.append(d.detach())
      acts.append(a.detach())
      for name, val in terms.items():
        r_terms.setdefault(name, []).append(val.detach())
      a_prev, s, kin, phi, m_prev = a, s1, kin1, phi1, m
      k = k + 1

    g = fns.goal(s, kin, lib, k, ref_offset)
    m = _alive(lib, k + ref_offset)
    tl = _t_left(lib, k + ref_offset, t_left_horizon)
    v_H = critic(phi, g, xi_enc, tl)
    w_H = m + m_prev * (1.0 - m) if (bootstrap and m_prev is not None) else m
    J = J + (gamma**H) * c * w_H * v_H
    states.append(s.detach())
    phis.append(phi.detach())
    goals.append(g.detach())
    ms.append(m)
    cs.append(c.detach())
    t_lefts.append(tl)

    phi_all, goal_all, tl_all = torch.stack(phis), torch.stack(goals), torch.stack(t_lefts)
    with torch.no_grad():
      V = critic(phi_all, goal_all, xi_enc, tl_all)

  return Rollout(
    J=J,
    s=torch.stack(states),
    g=goal_all,
    phi=phi_all,
    xi_enc=xi_enc,
    t_left=tl_all,
    r=torch.stack(rs),
    d=torch.stack(ds),
    m=torch.stack(ms),
    c=torch.stack(cs),
    V=V,
    a=torch.stack(acts),
    r_terms={n: torch.stack(v) for n, v in r_terms.items()},
    gamma=gamma,
    bootstrap_clip_end=bootstrap,
  )


@torch.no_grad()
def critic_targets(
  rollout: Any, lam: float = 0.95, gamma: float | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
  """Eq. 3 λ-returns on the detached rollout -> (R [H,B], loss weight m_t [H,B]).

  stop:      R_t = r_t + γ(1-d_t) m_{t+1} [(1-λ)V_{t+1} + λR_{t+1}],  R_H = V_H
  bootstrap: same without m_{t+1}, and R_t = V_t for states past the clip end.
  """
  gamma = rollout.gamma if gamma is None else gamma
  r, d, m, V = rollout.r, rollout.d, rollout.m, rollout.V
  H = r.shape[0]
  R = torch.empty_like(r)
  nxt = V[H]
  for t in reversed(range(H)):
    if rollout.bootstrap_clip_end:
      val = r[t] + gamma * (1.0 - d[t]) * ((1.0 - lam) * V[t + 1] + lam * nxt)
      val = m[t] * val + (1.0 - m[t]) * V[t]
    else:
      val = r[t] + gamma * (1.0 - d[t]) * m[t + 1] * ((1.0 - lam) * V[t + 1] + lam * nxt)
    R[t] = val
    nxt = val
  return R, m[:H]


def clip_and_step(
  module: nn.Module,
  optimizer: torch.optim.Optimizer,
  max_grad_norm: float = 1.0,
  elem_clip: float = 0.1,
) -> dict[str, float]:
  """Global-norm clip, THEN element-wise clamp (Table A.5 order), then step.

  A non-finite gradient skips the step (grads cleared) instead of corrupting the weights.
  """
  params = [p for p in module.parameters() if p.grad is not None]
  if not params:
    return {"grad_norm_preclip": 0.0, "frac_elem_clipped": 0.0, "skipped": 1.0}
  pre = torch.nn.utils.clip_grad_norm_(params, max_grad_norm)
  if not torch.isfinite(pre):
    optimizer.zero_grad(set_to_none=True)
    return {"grad_norm_preclip": float(pre), "frac_elem_clipped": 0.0, "skipped": 1.0}
  n_total = sum(p.grad.numel() for p in params)
  n_clip = sum((p.grad.abs() > elem_clip).sum() for p in params)
  for p in params:
    p.grad.clamp_(-elem_clip, elem_clip)
  optimizer.step()
  return {
    "grad_norm_preclip": float(pre),
    "frac_elem_clipped": float(n_clip) / max(n_total, 1),
    "skipped": 0.0,
  }


def actor_step(
  loss: torch.Tensor,
  actor: nn.Module,
  optimizer: torch.optim.Optimizer,
  max_grad_norm: float = 1.0,
  elem_clip: float = 0.1,
) -> dict[str, float]:
  """zero_grad -> backward -> clip_grad_norm(1.0) -> clamp(±0.1) -> step."""
  optimizer.zero_grad(set_to_none=True)
  loss.backward()
  return clip_and_step(actor, optimizer, max_grad_norm, elem_clip)


def actor_update(
  start: Any,
  actor: nn.Module,
  optimizer: torch.optim.Optimizer,
  wm: Any,
  reward: Callable[..., Any],
  critic: nn.Module,
  lib: Any,
  H: int = 32,
  gamma: float = 0.99,
  *,
  fns: ImagFns,
  microbatches: int = 1,
  max_grad_norm: float = 1.0,
  elem_clip: float = 0.1,
  **imagine_kw: Any,
) -> tuple[Rollout, dict[str, float]]:
  """One actor step on B starts, optionally as exact gradient accumulation over micro-batches
  (design §7.3 fallback 1: clipping happens once, after accumulation)."""
  s = _get(start, "s")
  B = s.shape[0]
  optimizer.zero_grad(set_to_none=True)
  pieces: list[Rollout] = []
  for idx in torch.arange(B, device=s.device).chunk(max(1, microbatches)):
    part = _slice_start(start, idx, B)
    ro = imagine(part, actor, wm, reward, critic, lib, H, gamma, fns=fns, **imagine_kw)
    (-ro.J.sum() / (B * H)).backward()  # sums to -mean(J)/H over the full batch
    pieces.append(ro.detached())
  stats = clip_and_step(actor, optimizer, max_grad_norm, elem_clip)
  ro = pieces[0] if len(pieces) == 1 else _cat_rollouts(pieces)
  stats.update(rollout_metrics(ro))
  return ro, stats


def fit_critic(
  critic: nn.Module,
  optimizer: torch.optim.Optimizer,
  rollout: Rollout,
  lam: float = 0.95,
  gamma: float | None = None,
  epochs: int = 5,
  minibatches: int = 4,
  max_grad_norm: float = 1.0,
) -> dict[str, float]:
  """Eq. 3 fit: targets computed once, then `epochs` x `minibatches` shuffled MSE steps over the
  alive (m_t = 1) imagined states."""
  R, w = critic_targets(rollout, lam, gamma)
  H, B = R.shape
  keep = w.reshape(-1) > 0
  phi = rollout.phi[:H].reshape(H * B, -1)[keep]
  goal = rollout.g[:H].reshape(H * B, -1)[keep]
  xi = rollout.xi_enc.expand(H, B, -1).reshape(H * B, -1)[keep]
  tl = rollout.t_left[:H].reshape(-1)[keep]
  y = R.reshape(-1)[keep]
  n = y.numel()
  if n == 0:
    return {"critic/loss": 0.0, "critic/expl_var": 0.0}
  losses = []
  for _ in range(epochs):
    for mb in torch.randperm(n, device=y.device).chunk(minibatches):
      loss = (critic(phi[mb], goal[mb], xi[mb], tl[mb]) - y[mb]).square().mean()
      optimizer.zero_grad(set_to_none=True)
      loss.backward()
      torch.nn.utils.clip_grad_norm_(critic.parameters(), max_grad_norm)
      optimizer.step()
      losses.append(loss.detach())
  with torch.no_grad():
    resid = y - critic(phi, goal, xi, tl)
    expl_var = 1.0 - resid.var() / y.var().clamp(min=1e-12)
  return {"critic/loss": float(torch.stack(losses).mean()), "critic/expl_var": float(expl_var)}


@torch.no_grad()
def rollout_metrics(ro: Rollout) -> dict[str, float]:
  w = ro.m[:-1]
  denom = w.sum().clamp(min=1.0)
  a_abs = ro.a.abs().flatten().float()
  if a_abs.numel() > 1_000_000:  # quantile() has an input-size limit; a subsample is enough
    a_abs = a_abs[torch.randint(a_abs.numel(), (1_000_000,), device=a_abs.device)]
  out = {
    "actor/J": float(ro.J.mean()),
    "imag/c_H": float(ro.c[-1].mean()),
    "imag/r_mean": float((ro.r * w).sum() / denom),
    "imag/d_mean": float((ro.d * w).sum() / denom),
    "imag/alive_frac": float(w.mean()),
    "actor/|a|_p50": float(a_abs.quantile(0.5)),
    "actor/|a|_p99": float(a_abs.quantile(0.99)),
  }
  for name, v in ro.r_terms.items():
    out[f"imag/r_{name}"] = float((v * w).sum() / denom)
  return out


def _slice_start(start: Any, idx: torch.Tensor, batch: int) -> Any:
  """Rows `idx` of every batch-first tensor field of a start batch (dict, dataclass or object)."""

  def cut(v: Any) -> Any:
    return v[idx.to(v.device)] if isinstance(v, torch.Tensor) and v.shape[:1] == (batch,) else v

  if isinstance(start, dict):
    return {k: cut(v) for k, v in start.items()}
  if dataclasses.is_dataclass(start):
    return dataclasses.replace(start, **{f.name: cut(getattr(start, f.name)) for f in dataclasses.fields(start)})
  new = type(start).__new__(type(start))
  new.__dict__.update({k: cut(v) for k, v in vars(start).items()})
  return new


def _cat_rollouts(parts: list[Rollout]) -> Rollout:
  def cat(name: str, dim: int) -> torch.Tensor:
    return torch.cat([getattr(p, name) for p in parts], dim=dim)

  return Rollout(
    J=cat("J", 0),
    s=cat("s", 1),
    g=cat("g", 1),
    phi=cat("phi", 1),
    xi_enc=cat("xi_enc", 0),
    t_left=cat("t_left", 1),
    r=cat("r", 1),
    d=cat("d", 1),
    m=cat("m", 1),
    c=cat("c", 1),
    V=cat("V", 1),
    a=cat("a", 1),
    r_terms={n: torch.cat([p.r_terms[n] for p in parts], dim=1) for n in parts[0].r_terms},
    gamma=parts[0].gamma,
    bootstrap_clip_end=parts[0].bootstrap_clip_end,
  )
