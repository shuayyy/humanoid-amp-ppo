"""Per-env sequence replay for SuperMimic (design_learner.md §3.5; Table A.5: FIFO, 1000/env).

mjlab auto-resets inside step(), so the buffer stores the pre-reset s_{t+1} next to s_t: k-step
windows and the termination head then see real terminal states. Time is indexed by age (oldest
first), so windows and rebuilt histories never straddle the ring's write pointer.

Per-slot meaning (slot t of env e): s = s_t (post-reset if a reset happened at t−1), a_prev =
a_{t−1} (0 right after a reset), a_exec = executed (noisy, unclipped) a_t, s_next = pre-reset
s_{t+1}, r / r_terms / term / trunc = outcome of a_t, ref_idx = frame r_{t+1} is scored against,
ep_step = steps since the reset that produced s_t, xi = DR context of the episode.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Mapping

import torch

FieldSpec = tuple[tuple[int, ...], torch.dtype]
RefValid = Callable[[torch.Tensor], torch.Tensor] | int | None

_REQUIRED = (
  "s", "s_next", "a_exec", "a_prev", "xi", "r", "r_terms", "term", "trunc", "ep_step", "ref_idx"
)


@dataclass(frozen=True)
class ReplaySpec:
  state_dim: int  # 71 (walk) / 84 (lift)
  action_dim: int  # 29
  xi_dim: int  # 33 / 35
  num_reward_terms: int = 0
  # Additional per-step fields: (name, per-step shape, dtype).
  extra_fields: tuple[tuple[str, tuple[int, ...], torch.dtype], ...] = ()

  def fields(self) -> dict[str, FieldSpec]:
    f: dict[str, FieldSpec] = {
      "s": ((self.state_dim,), torch.float32),
      "s_next": ((self.state_dim,), torch.float32),
      "a_exec": ((self.action_dim,), torch.float32),
      "a_prev": ((self.action_dim,), torch.float32),
      "xi": ((self.xi_dim,), torch.float32),
      "r": ((), torch.float32),
      "r_terms": ((self.num_reward_terms,), torch.float32),
      "term": ((), torch.bool),
      "trunc": ((), torch.bool),
      "ep_step": ((), torch.int16),
      "ref_idx": ((), torch.int32),
    }
    for name, shape, dtype in self.extra_fields:
      f[name] = (tuple(shape), dtype)
    return f


@dataclass
class Window:
  """B windows of k consecutive steps of one env; step j uses a[:, j] and predicts target[:, j]."""

  s0: torch.Tensor  # [B,S] state at the window start
  target: torch.Tensor  # [B,k,S] pre-reset s_{t0+j+1}
  a: torch.Tensor  # [B,k,A] executed actions
  a_prev: torch.Tensor  # [B,k,A]
  xi: torch.Tensor  # [B,D] raw DR context of the start slot (other episodes' steps are masked)
  r: torch.Tensor  # [B,k]
  r_terms: torch.Tensor  # [B,k,R]
  term: torch.Tensor  # [B,k] bool, true terminations only
  trunc: torch.Tensor  # [B,k] bool, time-outs / clip end
  mask: torch.Tensor  # [B,k] float32: 1 while no term/trunc happened at an earlier window step
  ref_idx: torch.Tensor  # [B,k] long
  ep_step: torch.Tensor  # [B,k] long
  env_ids: torch.Tensor  # [B] long
  slots: torch.Tensor  # [B,k] long, physical ring slots
  extras: dict[str, torch.Tensor] = field(default_factory=dict)  # spec extra fields, [B,k,...]


@dataclass
class Start:
  """Imagination start states with the clean (noise-free) inputs to rebuild the obs history."""

  s: torch.Tensor  # [B,S]
  a_prev: torch.Tensor  # [B,A]
  xi: torch.Tensor  # [B,D] raw
  ref_idx: torch.Tensor  # [B] long
  ep_step: torch.Tensor  # [B] long
  env_ids: torch.Tensor  # [B] long
  slot: torch.Tensor  # [B] long
  # History frames ordered OLDEST -> NEWEST (newest = the start slot itself), padded with the
  # episode's first frame like the sim's reset padding: frame i is slot t − min(H−1−i, ep_step).
  hist_slots: torch.Tensor  # [B,H] long
  hist_s: torch.Tensor  # [B,H,S]
  hist_a_prev: torch.Tensor  # [B,H,A]
  # Clean obs frames obs_frame_fn(hist_s, hist_a_prev, xi) [B,H,F], when an obs_frame_fn is set.
  hist: torch.Tensor | None = None


ObsFrameFn = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]
"""(s [*,S], a_prev [*,A], xi raw [*,D]) -> clean obs frame [*,F] (features.obs_frame, wrapped)."""


class SequenceReplay:
  """Preallocated ring [N, T] per field; all envs are written together at one shared pointer.

  `obs_frame_fn`, `hist_len` and `ref_valid` are defaults for `sample_starts`, so callers that
  only pass the batch size still get `Start.hist` and the clip-end filter.
  """

  def __init__(
    self,
    num_envs: int,
    capacity: int,
    spec: ReplaySpec | Mapping[str, FieldSpec],
    device: str | torch.device = "cpu",
    *,
    obs_frame_fn: ObsFrameFn | None = None,
    hist_len: int = 10,
    ref_valid: RefValid = None,
  ):
    self.num_envs = num_envs
    self.capacity = capacity
    self.device = torch.device(device)
    self.obs_frame_fn = obs_frame_fn
    self.hist_len = hist_len
    self.ref_valid = ref_valid
    if isinstance(spec, ReplaySpec):
      self.field_specs = spec.fields()
    else:
      self.field_specs = {k: (tuple(v[0]), v[1]) for k, v in spec.items()}
    missing = [k for k in _REQUIRED if k not in self.field_specs]
    if missing:
      raise ValueError(f"replay spec is missing required fields {missing}")
    self.data = {
      name: torch.zeros((num_envs, capacity, *shape), dtype=dtype, device=self.device)
      for name, (shape, dtype) in self.field_specs.items()
    }
    self.ptr = 0
    self.size = 0
    self._ep_counter = torch.zeros(num_envs, dtype=torch.long, device=self.device)
    # [N,T]: s, a_prev and xi of this slot and of every earlier slot of its episode are finite.
    # A physics blow-up stores NaN states (no env term fires on NaN); one NaN start would make
    # the whole actor gradient non-finite, so such slots (and their histories) are never starts.
    self._finite = torch.zeros(num_envs, capacity, dtype=torch.bool, device=self.device)
    self._ep_finite = torch.ones(num_envs, dtype=torch.bool, device=self.device)

  # ---- bookkeeping -------------------------------------------------------------------------------

  def __len__(self) -> int:
    return self.num_envs * self.size

  @property
  def nbytes(self) -> int:
    return sum(t.numel() * t.element_size() for t in self.data.values())

  def clear(self) -> None:
    self.ptr = 0
    self.size = 0
    self._ep_counter.zero_()
    self._finite.zero_()
    self._ep_finite.fill_(True)

  def slot_of_age(self, age: torch.Tensor | int) -> torch.Tensor | int:
    """Physical slot of logical age (0 = oldest stored step)."""
    return (age + self.ptr - self.size) % self.capacity

  def age_of_slot(self) -> torch.Tensor:
    """[T] age of each physical slot; slots with age >= size are unwritten."""
    slots = torch.arange(self.capacity, device=self.device)
    return (slots - (self.ptr - self.size)) % self.capacity

  # ---- writing -----------------------------------------------------------------------------------

  @torch.no_grad()
  def add(self, data: Mapping[str, torch.Tensor]) -> None:
    """Append one vectorized step: every spec field as [N, *shape]. `ep_step` is optional (an
    internal counter driven by term|trunc is used when absent). Extra keys are ignored."""
    N = self.num_envs
    if "ep_step" in data:
      ep_step = torch.as_tensor(data["ep_step"], device=self.device).reshape(N).long()
    else:
      ep_step = self._ep_counter
    for name, buf in self.data.items():
      if name == "ep_step":
        v = ep_step.clamp(max=torch.iinfo(buf.dtype).max)
      elif name in data:
        v = data[name]
      elif buf[0, 0].numel() == 0:
        continue  # zero-width field (e.g. r_terms with no terms)
      else:
        raise KeyError(f"SequenceReplay.add: missing field {name!r}")
      v = torch.as_tensor(v, device=self.device).detach()
      buf[:, self.ptr] = v.reshape(N, *buf.shape[2:]).to(buf.dtype)
    fin = torch.ones(N, dtype=torch.bool, device=self.device)
    for name in ("s", "a_prev", "xi"):
      fin &= torch.isfinite(self.data[name][:, self.ptr].reshape(N, -1)).all(-1)
    stored_ep = self.data["ep_step"][:, self.ptr]
    fin &= (stored_ep == 0) | self._ep_finite  # an episode start clears an earlier blow-up
    self._finite[:, self.ptr] = fin
    self._ep_finite = fin
    done = self.data["term"][:, self.ptr] | self.data["trunc"][:, self.ptr]
    self._ep_counter = torch.where(done, torch.zeros_like(ep_step), ep_step + 1)
    self.ptr = (self.ptr + 1) % self.capacity
    self.size = min(self.size + 1, self.capacity)

  # ---- sampling ----------------------------------------------------------------------------------

  def _randint(
    self, high: int, size: tuple[int, ...], generator: torch.Generator | None
  ) -> torch.Tensor:
    dev = generator.device if generator is not None else self.device
    return torch.randint(0, high, size, generator=generator, device=dev).to(self.device)

  @torch.no_grad()
  def sample_windows(self, batch: int, k: int, generator: torch.Generator | None = None) -> Window:
    """Uniform over (env, start age) with all k steps stored; mask cuts at the first episode end.

    mask[:, 0] = 1; mask[:, j] = Π_{i<j} (1 − done_i) with done = term | trunc. A terminal step
    itself is unmasked and its target is the pre-reset s_next.
    """
    if self.size < k:
      raise ValueError(f"replay holds {self.size} steps per env, need >= {k}")
    env = self._randint(self.num_envs, (batch,), generator)
    age0 = self._randint(self.size - k + 1, (batch,), generator)
    slots = self.slot_of_age(age0[:, None] + torch.arange(k, device=self.device))
    e = env[:, None]

    def g(name: str) -> torch.Tensor:
      return self.data[name][e, slots]

    term, trunc = g("term"), g("trunc")
    alive = (~(term | trunc)).to(torch.float32)
    mask = torch.ones(batch, k, device=self.device)
    mask[:, 1:] = torch.cumprod(alive[:, :-1], dim=1)
    extras = {name: g(name) for name in self.data if name not in _REQUIRED}
    return Window(
      s0=self.data["s"][env, slots[:, 0]],
      target=g("s_next"),
      a=g("a_exec"),
      a_prev=g("a_prev"),
      xi=self.data["xi"][env, slots[:, 0]],
      r=g("r"),
      r_terms=g("r_terms"),
      term=term,
      trunc=trunc,
      mask=mask,
      ref_idx=g("ref_idx").long(),
      ep_step=g("ep_step").long(),
      env_ids=env,
      slots=slots,
      extras=extras,
    )

  @torch.no_grad()
  def start_mask(self, hist_len: int | None = None, ref_valid: RefValid = None) -> torch.Tensor:
    """[N,T] bool: slots usable as imagination starts.

    Written, not followed by a true termination (term = 0), reference still inside the clip
    (`ref_valid`: callable on ref_idx, or the last valid frame index; default from the
    constructor), the clean history of min(hist_len−1, ep_step) earlier steps still stored
    (not overwritten by the ring), and no non-finite s / a_prev / ξ so far in the episode.
    """
    hist_len = self.hist_len if hist_len is None else hist_len
    ref_valid = self.ref_valid if ref_valid is None else ref_valid
    age = self.age_of_slot()[None]  # [1,T]
    ep = self.data["ep_step"].long()
    ok = (age < self.size) & ~self.data["term"] & (age >= torch.clamp(ep, max=hist_len - 1))
    ok &= self._finite
    if ref_valid is not None:
      ref = self.data["ref_idx"].long()
      ok &= ref_valid(ref) if callable(ref_valid) else ref <= int(ref_valid)
    return ok

  @torch.no_grad()
  def sample_starts(
    self,
    batch: int,
    hist_len: int | None = None,
    ref_valid: RefValid = None,
    generator: torch.Generator | None = None,
    obs_frame_fn: ObsFrameFn | None = None,
  ) -> Start:
    """Uniform over `start_mask` slots, with the clean history inputs (s, a_prev per frame) and,
    when an obs_frame_fn is available, the clean obs frames `hist` [B,H,F] (oldest -> newest)."""
    hist_len = self.hist_len if hist_len is None else hist_len
    obs_frame_fn = self.obs_frame_fn if obs_frame_fn is None else obs_frame_fn
    valid = self.start_mask(hist_len, ref_valid).flatten().nonzero().squeeze(1)
    if valid.numel() == 0:
      raise ValueError("SequenceReplay.sample_starts: no valid start slots")
    pick = valid[self._randint(valid.numel(), (batch,), generator)]
    env = pick // self.capacity
    slot = pick % self.capacity
    ep = self.data["ep_step"][env, slot].long()
    back = torch.arange(hist_len - 1, -1, -1, device=self.device)  # oldest -> newest
    lookback = torch.minimum(back[None], ep[:, None])
    hist_slots = (slot[:, None] - lookback) % self.capacity
    e = env[:, None]
    xi = self.data["xi"][env, slot]
    hist_s = self.data["s"][e, hist_slots]
    hist_a_prev = self.data["a_prev"][e, hist_slots]
    hist = None
    if obs_frame_fn is not None:
      hist = obs_frame_fn(hist_s, hist_a_prev, xi[:, None].expand(-1, hist_len, -1))
    return Start(
      s=self.data["s"][env, slot],
      a_prev=self.data["a_prev"][env, slot],
      xi=xi,
      ref_idx=self.data["ref_idx"][env, slot].long(),
      ep_step=ep,
      env_ids=env,
      slot=slot,
      hist_slots=hist_slots,
      hist_s=hist_s,
      hist_a_prev=hist_a_prev,
      hist=hist,
    )

  # ---- persistence (optional `save_replay`) ------------------------------------------------------

  def state_dict(self) -> dict:
    return {
      "data": self.data,
      "ptr": self.ptr,
      "size": self.size,
      "ep_counter": self._ep_counter,
      "finite": self._finite,
      "ep_finite": self._ep_finite,
    }

  def load_state_dict(self, state: dict) -> None:
    for name, buf in self.data.items():
      buf.copy_(state["data"][name])
    self.ptr = int(state["ptr"])
    self.size = int(state["size"])
    self._ep_counter.copy_(state["ep_counter"])
    if "finite" in state:
      self._finite.copy_(state["finite"])
      self._ep_finite.copy_(state["ep_finite"])
    else:  # older state: per-slot check only (no episode history)
      self._finite.fill_(True)
      for name in ("s", "a_prev", "xi"):
        self._finite &= torch.isfinite(self.data[name].flatten(2)).all(-1)
      self._ep_finite.copy_(self._finite[:, (self.ptr - 1) % self.capacity])
