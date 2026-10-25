"""SuperMimic runner: sim collection, startup parity checks, the training loop and logging.

`SuperMimicRunner(env, train_cfg, log_dir, device)` mirrors rsl_rl's OnPolicyRunner call shape
(train/play scripts pass `train_cfg` as a dict) but does not subclass it: the loop, the
logger (fractional env steps per iteration) and the inference policy all differ.

The env is the RSL-RL VecEnv wrapper around SMTrackEnv (brief §2). SuperMimic ignores the env's
observation groups: it rebuilds the policy observation from raw state with `features.obs_frame`
so that sim and imagination see identical features.
"""

from __future__ import annotations

import json
import math
import os
import time
from typing import Any

import torch

from mjlab_g1.rl.supermimic.algorithm import Components, SuperMimic, build_components
from mjlab_g1.rl.supermimic.config import SuperMimicRunnerCfg, as_runner_cfg

# Per-term layout of the env's `actor_hist` group (brief §2) and its uniform noise bounds; used
# only by the warn-only observation parity check.
_OBS_TERM_DIMS = (29, 29, 3, 3, 29)
_OBS_TERM_NOISE = (0.01, 0.5, 0.2, 0.05, 0.0)
_OBS_BOX_DIMS = (3, 6)
_OBS_BOX_NOISE = (0.01, 0.02)


class SuperMimicRunner:
  def __init__(
    self,
    env: Any,
    train_cfg: SuperMimicRunnerCfg | dict,
    log_dir: str | None = None,
    device: str = "cpu",
    components: Components | None = None,
  ):
    self.cfg = as_runner_cfg(train_cfg)
    self.env = env
    self.device = torch.device(device)
    self.log_dir = log_dir
    _configure_backends(self.cfg)
    torch.manual_seed(self.cfg.seed)  # learner RNG; the env seeds itself from its own cfg
    if components is None:
      components = build_components(env, self.cfg, str(self.device))
    self.alg = SuperMimic(self.cfg, components, str(self.device))
    self.current_learning_iteration = 0  # completed iterations
    self.env_step_acc = 0.0  # fractional accumulator for env_steps_per_iter
    self.total_env_steps = 0  # transitions (vectorized steps x N)
    self.git_status_repos: list[str] = []
    self.writer = None
    self._metrics_path: str | None = None
    self._warned_no_timeouts = False
    self._collecting = False  # sim history / a_prev / episode counters are live
    self.parity_report: dict[str, float] = {}

  # Public API.

  def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False) -> None:
    """Prefill (with parity checks), then `num_learning_iterations` iterations of Alg. A.1.

    Iterations count from `current_learning_iteration` (relative on resume). The prefill runs
    once per runner: a second call continues the same episodes, whose history, a_prev and
    replay ep_step would otherwise be reset mid-episode (desyncing a_prev from the env).
    """
    del init_at_random_ep_len  # start frames come from the env's adaptive sampler
    cfg = self.cfg
    self.alg.train_mode()
    self._open_logs()
    stats = _Stats(self.device)
    t_learn = time.time()
    start_it = self.current_learning_iteration

    if not self._collecting:
      if start_it > 0:
        self._maybe_freeze_wm_stats(start_it)  # resumed past the freeze point: keep stats fixed
      self._start_collection()
      t0 = time.perf_counter()
      self._prefill(stats)
      stats.add("perf/prefill_s", time.perf_counter() - t0)

    end_it = start_it + num_learning_iterations
    for it in range(start_it, end_it):
      self._maybe_freeze_wm_stats(it)
      lr_f = self.alg.set_lr(it)
      self.env_step_acc += cfg.env_steps_per_iter
      n_steps = int(self.env_step_acc)
      self.env_step_acc -= n_steps

      t = _Timer(self.device)
      for j in range(n_steps):
        self._collect_step(stats, record_actions=(j == n_steps - 1))
      stats.add("perf/sim_ms", t.lap())
      stats.add_dict(self.alg.wm_update())
      stats.add("perf/wm_ms", t.lap())
      rollout, m_actor = self.alg.actor_update()
      stats.add_dict(m_actor)
      stats.add("perf/actor_ms", t.lap())
      stats.add_dict(self.alg.critic_update(rollout))
      stats.add("perf/critic_ms", t.lap())
      del rollout
      stats.add("lr_factor", lr_f)
      stats.add("collect/steps_per_iter", n_steps)
      self.current_learning_iteration = it + 1

      done = it + 1
      if done % cfg.log_interval == 0 or done == end_it:
        self._log(done, stats, time.time() - t_learn, done - start_it)
      if self.log_dir is not None and (done % cfg.save_interval == 0 or done == end_it):
        self.save(os.path.join(self.log_dir, f"model_{done}.pt"))

    if self.writer is not None:
      self.writer.flush()

  def save(self, path: str, infos: dict | None = None) -> None:
    d = self.alg.save()
    d["iter"] = self.current_learning_iteration
    d["env_step_acc"] = self.env_step_acc
    d["total_env_steps"] = self.total_env_steps
    d["rng_state"] = {
      "torch": torch.get_rng_state(),
      "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    d["infos"] = infos
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save(d, path)

  def load(
    self,
    path: str,
    load_cfg: dict | None = None,
    strict: bool = True,
    map_location: str | None = None,
  ) -> dict | None:
    d = torch.load(path, weights_only=False, map_location=map_location or self.device)
    if self.alg.load(d, load_cfg, strict):
      self.current_learning_iteration = int(d["iter"])
      self.env_step_acc = float(d.get("env_step_acc", 0.0))
      self.total_env_steps = int(d.get("total_env_steps", 0))
      rng = d.get("rng_state") or {}
      if rng.get("torch") is not None:
        torch.set_rng_state(rng["torch"].cpu())
      if rng.get("cuda") is not None and torch.cuda.is_available():
        try:
          torch.cuda.set_rng_state_all(rng["cuda"])
        except Exception:  # different device count on resume
          pass
    self.alg.set_lr(self.current_learning_iteration)
    return d.get("infos")

  def get_inference_policy(self, device: str | None = None, from_obs: bool = False) -> Any:
    """Deterministic policy for eval/play.

    Default: `SuperMimicInferencePolicy`, which rebuilds the observation from raw env state with
    the training feature code (see its call protocol). `from_obs=True` instead returns
    `actor_critic.ActorInference` on the env's `actor_hist` / `actor_goal` groups (same mjlab
    layout; exportable, but only as faithful as the env's obs parity); call it under no_grad.
    """
    del device  # runs on the learner's device (raw env state lives there)
    self.alg.eval_mode()
    if from_obs:
      from mjlab_g1.rl.supermimic.actor_critic import ActorInference

      return ActorInference(self.alg.actor).eval()
    return SuperMimicInferencePolicy(self.alg, self.env)

  def add_git_repo_to_log(self, repo_file_path: str) -> None:
    self.git_status_repos.append(repo_file_path)

  def _maybe_freeze_wm_stats(self, it: int) -> None:
    # Critique M2 opt-in; the flag is not checkpointed, so it is re-derived from the iteration.
    n = self.cfg.wm.freeze_normalizers_after
    if n is not None and it >= n and hasattr(self.alg.wm, "normalizers_frozen"):
      self.alg.wm.normalizers_frozen = True

  # Collection.

  def _read_env(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Clone: the env may update these buffers in place on the next step.
    u = self.env.unwrapped
    s = u.get_state().to(self.device, torch.float32).clone()
    xi = u.get_xi().to(self.device, torch.float32).clone()
    k = u.get_ref_index().to(self.device).long().clone()
    return s, xi, k

  def _start_collection(self) -> None:
    N = self.env.num_envs
    A = self.env.num_actions
    self._s, self._xi, self._k = self._read_env()
    # a_{t-1} as the env sees it (last_action obs, action_rate reward): 0 after a reset, but the
    # env may already have been stepped before this runner existed.
    a_env = getattr(getattr(self.env.unwrapped, "action_manager", None), "action", None)
    if isinstance(a_env, torch.Tensor) and tuple(a_env.shape) == (N, A):
      self._a_prev = a_env.to(self.device, torch.float32).clone()
    else:
      self._a_prev = torch.zeros(N, A, device=self.device)
    self._collecting = True
    frame = self.alg.obs_frame(self._s, self._a_prev, self._xi, noise=True)
    self._hist = self.alg.feats.PolicyObsHistory(
      N, self.cfg.policy.obs_history, frame.shape[-1], device=self.device
    )
    self._hist.reset(torch.arange(N, device=self.device), frame)
    self._ep_step = torch.zeros(N, dtype=torch.long, device=self.device)
    self._ep_ret = torch.zeros(N, device=self.device)
    self._ep_len = torch.zeros(N, device=self.device)
    self._ep_k0 = self._k.clone()

  def _prefill(self, stats: "_Stats") -> None:
    cfg = self.cfg
    N = self.env.num_envs
    parity_steps = math.ceil(cfg.parity_transitions / N) if cfg.parity_check else 0
    n = max(cfg.prefill_steps, parity_steps, cfg.wm.window_k + 1)
    parity = _ParityCheck(self, self.env.unwrapped) if cfg.parity_check else None
    for _ in range(n):
      self._collect_step(stats, parity=parity)
    if parity is not None:
      self.parity_report = parity.finish()
      for k, v in self.parity_report.items():
        stats.add(k, v)

  def _collect_step(
    self, stats: "_Stats", parity: "_ParityCheck | None" = None, record_actions: bool = False
  ) -> None:
    u = self.env.unwrapped
    a_exec, a_det = self.alg.act(self._s, self._hist.get(), self._k, update_stats=True)
    obs, rew, dones, extras = self.env.step(a_exec.to(self.env.device))
    s_next = u.pre_reset_state.to(self.device, torch.float32).clone()
    rew = rew.to(self.device, torch.float32).reshape(-1)
    dones = dones.to(self.device).reshape(-1).bool()
    term_buf = getattr(u, "reset_terminated", None)
    if isinstance(term_buf, torch.Tensor):
      # mjlab's own flag: also right when an env falls on the clip-end step (both flags set).
      term = term_buf.to(self.device).reshape(-1).bool().clone() & dones
    else:
      time_outs = extras.get("time_outs")
      if time_outs is None:
        if not self._warned_no_timeouts:
          print("[SuperMimic] WARNING: extras['time_outs'] missing; every done is terminal")
          self._warned_no_timeouts = True
        time_outs = torch.zeros_like(dones)
      term = dones & ~time_outs.to(self.device).reshape(-1).bool()
    trunc = dones & ~term
    r_terms = u.get_reward_terms().to(self.device, torch.float32)
    _add_env_log(stats, extras)

    tr = {
      "s": self._s,
      "s_next": s_next,
      "a_exec": a_exec,
      "a_prev": self._a_prev,
      "xi": self._xi,
      "r": rew,
      "r_terms": r_terms,
      "term": term,
      "trunc": trunc,
      "ep_step": self._ep_step,
      "ref_idx": self._k,
    }
    if parity is not None:
      parity.check(tr, obs)
    self.alg.process_transition(tr)

    # Episode bookkeeping (device-side sums; host sync only at log time).
    self._ep_ret += rew
    self._ep_len += 1
    df = dones.float()
    possible = (self.alg.lib.num_frames - self._ep_k0).clamp(min=1).float()
    norm = self._ep_ret / (self.alg.r_max_per_step * possible)
    stats.add_sum("sim/episodes", df.sum())
    stats.add_sum("sim/ep_len_sum", (self._ep_len * df).sum())
    stats.add_sum("sim/ep_ret_sum", (self._ep_ret * df).sum())
    stats.add_sum("sim/norm_return_sum", (norm * df).sum())
    stats.add_sum("sim/terminations", term.float().sum())
    stats.add_sum("sim/transitions", float(rew.numel()))
    stats.add_sum("sim/r_sum", rew.sum())
    stats.add_sum("sim/r_terms_sum", r_terms.sum(0))
    if record_actions:
      a = a_det.abs().flatten().float()
      q = torch.quantile(a, torch.tensor([0.5, 0.99], device=a.device))
      stats.add("sim/|a|_p50", q[0])
      stats.add("sim/|a|_p99", q[1])
      stats.add("sim/|a|_max", a.max())
    self.total_env_steps += rew.numel()

    # Advance to s_{t+1} (post-reset for done envs).
    self._s, self._xi, k_next = self._read_env()
    self._k = k_next
    self._a_prev = a_exec.clone()
    self._a_prev[dones] = 0.0
    self._ep_step = self._ep_step + 1
    self._ep_step[dones] = 0
    frame = self.alg.obs_frame(self._s, self._a_prev, self._xi, noise=True)
    self._hist.push(frame)
    done_ids = dones.nonzero().squeeze(-1)
    if done_ids.numel() > 0:
      self._hist.reset(done_ids, frame[done_ids])
      self._ep_ret[done_ids] = 0.0
      self._ep_len[done_ids] = 0.0
      self._ep_k0[done_ids] = self._k[done_ids]

  # Logging.

  def _open_logs(self) -> None:
    if self.log_dir is None:
      return
    os.makedirs(self.log_dir, exist_ok=True)
    self._metrics_path = os.path.join(self.log_dir, "metrics.jsonl")
    if self.writer is None and self.cfg.logger == "tensorboard":
      try:
        from torch.utils.tensorboard import SummaryWriter

        self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=30)
      except Exception as e:  # tensorboard missing: metrics.jsonl still gets written
        print(f"[SuperMimic] tensorboard unavailable ({e}); logging to metrics.jsonl only")
    elif self.cfg.logger != "tensorboard":
      print(f"[SuperMimic] logger '{self.cfg.logger}' not supported; using metrics.jsonl only")

  def _log(self, it: int, stats: "_Stats", wall: float, its_this_run: int) -> None:
    vals = stats.flush()
    out: dict[str, Any] = {
      "iteration": it,
      "wall_time": wall,
      "env_steps": self.total_env_steps,
    }
    # Turn sums into means.
    eps_n = vals.pop("sim/episodes", 0.0)
    trans = vals.pop("sim/transitions", 0.0)
    for key, name in (
      ("sim/ep_len_sum", "sim/mean_episode_length"),
      ("sim/ep_ret_sum", "sim/mean_episode_return"),
      ("sim/norm_return_sum", "sim/norm_return"),
    ):
      v = vals.pop(key, 0.0)
      if eps_n > 0:
        out[name] = v / eps_n
    term_n = vals.pop("sim/terminations", 0.0)
    r_sum = vals.pop("sim/r_sum", 0.0)
    r_terms = vals.pop("sim/r_terms_sum", None)
    out["sim/episodes"] = eps_n
    if trans > 0:
      out["sim/r_mean"] = r_sum / trans
      out["sim/term_rate"] = term_n / trans
    if eps_n > 0:
      out["sim/term_frac_of_episodes"] = term_n / eps_n
    names = list(getattr(self.env.unwrapped, "reward_term_names", []))
    if r_terms is not None and trans > 0:
      r_terms = r_terms if isinstance(r_terms, list) else [r_terms]
      for i, v in enumerate(r_terms):
        out[f"sim/r_{names[i] if i < len(names) else i}"] = v / trans
    out.update(vals)
    if self.device.type == "cuda":
      out["perf/mem_gb"] = torch.cuda.max_memory_allocated(self.device) / 2**30
    out["iter_per_s"] = its_this_run / max(wall, 1e-9)  # `it` is absolute on resume

    if self._metrics_path is not None:
      with open(self._metrics_path, "a") as f:
        f.write(json.dumps(out) + "\n")
    if self.writer is not None:
      for k, v in out.items():
        if k != "iteration" and isinstance(v, (int, float)):
          self.writer.add_scalar(k, v, it)
    print(
      f"[SuperMimic] it {it} | env_steps {self.total_env_steps} | "
      f"r/step {out.get('sim/r_mean', float('nan')):.4f} | "
      f"norm_ret {out.get('sim/norm_return', float('nan')):.3f} | "
      f"ep_len {out.get('sim/mean_episode_length', float('nan')):.1f} | "
      f"wm {out.get('wm/loss', float('nan')):.4f} | J {out.get('actor/J', float('nan')):.4f} | "
      f"V {out.get('critic/loss', float('nan')):.5f} | {wall:.0f}s",
      flush=True,
    )


class SuperMimicInferencePolicy:
  """Deterministic SuperMimic policy for eval/play.

  It ignores the observation dict and reads raw state through `env.unwrapped` (get_state,
  get_xi, get_ref_index), keeping its own PolicyObsHistory built by the training feature code
  (alg.feats: same layout, padding and a_prev convention as collection).

  Call protocol (eval):
    policy = runner.get_inference_policy()
    obs = env.get_observations()
    for _ in range(T):
      a = policy(obs)                     # reads the env state that `obs` belongs to
      obs, rew, dones, extras = env.step(a)
      policy.reset(dones.nonzero().squeeze(-1))   # optional: envs with episode_length_buf == 0
                                                  # are also re-initialized automatically
  Observation noise is off by default (eval: no obs noise); `obs_noise=True` re-enables it.
  """

  def __init__(self, alg: SuperMimic, env: Any, obs_noise: bool = False):
    self.alg = alg
    self.env = env
    self.obs_noise = obs_noise
    self.device = alg.device
    self._hist = None
    self._a_prev = None
    self._pending: torch.Tensor | None = None

  def reset(self, env_ids: torch.Tensor | None = None) -> None:
    """Mark envs as freshly reset (history refilled, a_prev = 0 on the next call)."""
    if self._hist is None:
      return
    N = self.env.num_envs
    mask = torch.zeros(N, dtype=torch.bool, device=self.device)
    if env_ids is None:
      mask[:] = True
    else:
      mask[env_ids.to(self.device).long()] = True
    self._pending = mask if self._pending is None else (self._pending | mask)

  @torch.no_grad()
  def __call__(self, obs: Any = None) -> torch.Tensor:
    del obs
    u = self.env.unwrapped
    N = self.env.num_envs
    s = u.get_state().to(self.device, torch.float32)
    xi = u.get_xi().to(self.device, torch.float32)
    k = u.get_ref_index().to(self.device).long()
    new = torch.zeros(N, dtype=torch.bool, device=self.device)
    if self._pending is not None:
      new |= self._pending
      self._pending = None
    ep = getattr(u, "episode_length_buf", None)
    if ep is not None:
      new |= ep.to(self.device).reshape(-1) == 0
    if self._hist is None:
      self._a_prev = torch.zeros(N, self.env.num_actions, device=self.device)
      frame = self.alg.obs_frame(s, self._a_prev, xi, noise=self.obs_noise)
      self._hist = self.alg.feats.PolicyObsHistory(
        N, self.alg.cfg.policy.obs_history, frame.shape[-1], device=self.device
      )
      self._hist.reset(torch.arange(N, device=self.device), frame)
    else:
      self._a_prev[new] = 0.0
      frame = self.alg.obs_frame(s, self._a_prev, xi, noise=self.obs_noise)
      self._hist.push(frame)
      ids = new.nonzero().squeeze(-1)
      if ids.numel() > 0:
        self._hist.reset(ids, frame[ids])
    _, a = self.alg.act(s, self._hist.get(), k, noise_std=0.0, update_stats=False)
    self._a_prev = a.clone()
    return a


class _ParityCheck:
  """Startup parity on real transitions (brief §4): analytic reward/termination vs the env."""

  def __init__(self, runner: SuperMimicRunner, u: Any):
    self.r = runner
    self.u = u
    self.n = 0
    self.r_err = 0.0
    self.term_err: dict[str, float] = {}
    self.term_events = 0
    self.term_pred_events = 0
    self.term_mismatch = 0
    self.trunc_mismatch = 0
    self.trunc_events = 0
    self.obs_excess = 0.0
    self.obs_checked = False

  @torch.no_grad()
  def check(self, tr: dict[str, torch.Tensor], obs: Any) -> None:
    alg = self.r.alg
    cfg = self.r.cfg
    s1 = tr["s_next"]
    kin1 = alg.kin(s1)
    ref = alg.lib.frame(tr["ref_idx"] + cfg.reward_ref_offset)
    # kin_prev: mjlab re-anchors the body-pose reference on the anchor of s_t.
    r_an, terms = alg.reward(s1, kin1, ref, tr["a_exec"], tr["a_prev"], kin_prev=alg.kin(tr["s"]))
    self.r_err = max(self.r_err, (r_an.reshape(-1) - tr["r"]).abs().max().item())
    names = list(getattr(self.u, "reward_term_names", []))
    for i, name in enumerate(names):
      if name in terms:
        e = (terms[name].reshape(-1) - tr["r_terms"][:, i]).abs().max().item()
        self.term_err[name] = max(self.term_err.get(name, 0.0), e)

    pred = alg.reward.terminated(kin1, ref, s1).reshape(-1).bool()
    self.term_events += int(tr["term"].sum())
    self.term_pred_events += int(pred.sum())
    self.term_mismatch += int((pred != tr["term"]).sum())

    # Clip-end truncation must coincide with the reference running out (imagination's mask
    # alive(k + offset) turning 0 at the next step).
    pred_trunc = ~alg.lib.alive(tr["ref_idx"] + cfg.reward_ref_offset + 1).reshape(-1).bool()
    live = ~tr["term"]
    self.trunc_events += int(tr["trunc"].sum())
    self.trunc_mismatch += int(((pred_trunc != tr["trunc"]) & live).sum())

    self._check_obs(obs)
    self.n += tr["r"].numel()

  def _check_obs(self, obs: Any) -> None:
    # Warn-only: the env's newest actor_hist frame must match obs_frame on the post-step state
    # within the env's uniform noise bounds (catches qvel/gyro frame conventions).
    try:
      hist = obs["actor_hist"]
    except Exception:
      return
    alg = self.r.alg
    u = self.u
    s = u.get_state().to(alg.device, torch.float32)
    xi = u.get_xi().to(alg.device, torch.float32)
    dims = list(_OBS_TERM_DIMS) + (list(_OBS_BOX_DIMS) if getattr(u, "has_box", False) else [])
    noise = list(_OBS_TERM_NOISE) + (list(_OBS_BOX_NOISE) if getattr(u, "has_box", False) else [])
    L = self.r.cfg.policy.obs_history
    hist = hist.to(alg.device, torch.float32)
    if hist.shape[-1] != L * sum(dims):
      return
    # a_prev for the frame is the executed action (0 for envs that just reset): use the env's
    # own last-action block instead of guessing.
    newest, o = [], 0
    for d in dims:
      blk = hist[:, o : o + L * d].reshape(-1, L, d)
      newest.append(blk[:, -1])
      o += L * d
    a_prev = newest[4]
    ours = alg.obs_frame(s, a_prev, xi, noise=False)
    if ours.shape[-1] != sum(dims):
      return
    o, worst = 0, 0.0
    for d, nb, env_part in zip(dims, noise, newest):
      excess = (ours[:, o : o + d] - env_part).abs() - nb
      worst = max(worst, excess.max().item())
      o += d
    self.obs_excess = max(self.obs_excess, worst)
    self.obs_checked = True

  def finish(self) -> dict[str, float]:
    cfg = self.r.cfg
    events = max(self.term_events, self.term_pred_events)
    allowed = max(1, math.ceil((1.0 - cfg.parity_term_min_acc) * events))
    report = {
      "parity/transitions": float(self.n),
      "parity/r_maxabs": self.r_err,
      "parity/term_events": float(self.term_events),
      "parity/term_pred_events": float(self.term_pred_events),
      "parity/term_mismatch": float(self.term_mismatch),
      "parity/trunc_events": float(self.trunc_events),
      "parity/trunc_mismatch": float(self.trunc_mismatch),
    }
    for name, e in self.term_err.items():
      report[f"parity/r_{name}_maxabs"] = e
    if self.obs_checked:
      report["parity/obs_excess_max"] = self.obs_excess
    print("[SuperMimic] startup parity: " + ", ".join(f"{k}={v:.3g}" for k, v in report.items()))
    if self.r_err > cfg.parity_reward_tol:
      worst = sorted(self.term_err.items(), key=lambda kv: -kv[1])[:3]
      raise RuntimeError(
        f"SuperMimic reward parity failed: max|r_analytic - r_env| = {self.r_err:.3g} > "
        f"{cfg.parity_reward_tol:g} over {self.n} transitions; worst terms {worst}. "
        "Check reward_ref_offset, the anchor body, dt scaling and stale-xpos pairing."
      )
    if self.term_mismatch > allowed:
      raise RuntimeError(
        f"SuperMimic termination parity failed: {self.term_mismatch} mismatches "
        f"(env {self.term_events} vs analytic {self.term_pred_events} terminations, allowed "
        f"{allowed}). Check the termination thresholds, anchor body and reward_ref_offset."
      )
    if self.term_events == 0:
      print("[SuperMimic] WARNING: no terminations during the parity window (check is vacuous)")
    if self.trunc_mismatch > 0:
      print(
        f"[SuperMimic] WARNING: {self.trunc_mismatch} truncations disagree with "
        "lib.alive(ref_idx + 1); imagination's clip-end mask will not match the sim"
      )
    if self.obs_checked and self.obs_excess > 1e-3:
      print(
        f"[SuperMimic] WARNING: obs_frame differs from the env's actor_hist by "
        f"{self.obs_excess:.3g} beyond the noise bounds"
      )
    return report


class _Stats:
  """Device-side metric accumulator: one host sync per log interval."""

  def __init__(self, device: torch.device):
    self.device = device
    self._mean: dict[str, list] = {}
    self._sum: dict[str, Any] = {}

  def add(self, key: str, v: Any) -> None:
    self._mean.setdefault(key, []).append(_t(v, self.device))

  def add_dict(self, d: dict[str, Any]) -> None:
    for k, v in d.items():
      self.add(k, v)

  def add_sum(self, key: str, v: Any) -> None:
    v = _t(v, self.device, reduce=False)
    self._sum[key] = v if key not in self._sum else self._sum[key] + v

  def flush(self) -> dict[str, Any]:
    keys = list(self._mean)
    flat = [torch.stack(self._mean[k]).mean() for k in keys]
    out: dict[str, Any] = {}
    if flat:
      vals = torch.stack(flat).tolist()
      out.update(dict(zip(keys, vals)))
    for k, v in self._sum.items():
      out[k] = v.item() if v.numel() == 1 else v.tolist()
    self._mean.clear()
    self._sum.clear()
    return out


class _Timer:
  def __init__(self, device: torch.device):
    self.device = device
    self.t = self._now()

  def _now(self) -> float:
    if self.device.type == "cuda":
      torch.cuda.synchronize(self.device)
    return time.perf_counter()

  def lap(self) -> float:
    now = self._now()
    ms = (now - self.t) * 1e3
    self.t = now
    return ms


def _add_env_log(stats: _Stats, extras: Any) -> None:
  """mjlab's per-reset episode info (Episode_Reward/*, Episode_Termination/*, Metrics/*), named
  as the rsl_rl Logger names it for the PPO arm so both learners' curves line up."""
  log = extras.get("log") if isinstance(extras, dict) else None
  if not log:
    return
  for key, v in log.items():
    if isinstance(v, torch.Tensor) and v.numel() > 0 and v.is_floating_point():
      stats.add(key if "/" in key else f"Episode/{key}", v)
    elif isinstance(v, (int, float)) and not isinstance(v, bool):
      stats.add(key if "/" in key else f"Episode/{key}", float(v))


def _t(v: Any, device: torch.device, reduce: bool = True) -> torch.Tensor:
  if isinstance(v, torch.Tensor):
    v = v.detach().to(device, torch.float32)
    return v.mean() if reduce else v
  return torch.tensor(float(v), device=device)


def _configure_backends(cfg: SuperMimicRunnerCfg) -> None:
  # Both ways, so tf32=False really means fp32 (train_track already turned TF32 on via mjlab).
  # TF32 only reaches matmuls (the MLPs); FK / quaternion / reward math is element-wise fp32.
  tf32 = bool(cfg.memory.tf32)
  matmul = torch.backends.cuda.matmul
  if hasattr(matmul, "fp32_precision"):  # torch >= 2.9 API, the one mjlab sets
    matmul.fp32_precision = "tf32" if tf32 else "ieee"
  else:
    matmul.allow_tf32 = tf32
