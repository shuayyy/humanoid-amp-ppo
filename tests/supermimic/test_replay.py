"""SequenceReplay: window sampling never crosses resets or the ring pointer; history rebuild pads
with the episode's first frame (design_learner.md §3.5, §8)."""

from __future__ import annotations

import torch

from mjlab_g1.rl.supermimic.replay import ReplaySpec, SequenceReplay

# Synthetic state: [env, episode id, step in episode, global time, is_s_next flag, 0].
S, A, D, R = 6, 2, 1, 3
REF_OFFSET = 3


def _fill(
  rb: SequenceReplay,
  steps: int,
  seed: int,
  p_term: float,
  p_trunc: float,
  max_len: int = 1000,
  give_ep_step: bool = True,
) -> None:
  """Write `steps` vectorized steps of synthetic episodes whose content encodes their origin."""
  N = rb.num_envs
  g = torch.Generator().manual_seed(seed)
  env = torch.arange(N, dtype=torch.float32)
  ep_id = torch.arange(N, dtype=torch.float32) + 1.0
  next_id = float(N + 1)
  step = torch.zeros(N)
  for t in range(steps):
    term = torch.rand(N, generator=g) < p_term
    trunc = ((torch.rand(N, generator=g) < p_trunc) | (step + 1 >= max_len)) & ~term
    zeros = torch.zeros(N)
    s = torch.stack((env, ep_id, step, zeros + t, zeros, zeros), -1)
    s_next = torch.stack((env, ep_id, step + 1, zeros + t + 1, zeros + 1, zeros), -1)
    a_exec = torch.stack((ep_id, step + 0.5), -1)
    prev = torch.stack((ep_id, step - 0.5), -1)
    a_prev = torch.where((step > 0)[:, None], prev, torch.zeros(N, 2))
    data = dict(
      s=s,
      s_next=s_next,
      a_exec=a_exec,
      a_prev=a_prev,
      xi=ep_id[:, None],
      r=zeros + t,
      r_terms=torch.stack((step, step, step), -1),
      term=term,
      trunc=trunc,
      ref_idx=(step + REF_OFFSET).long(),
    )
    if give_ep_step:
      data["ep_step"] = step.long()
    rb.add(data)
    done = term | trunc
    n_done = int(done.sum())
    ep_id = ep_id.clone()
    ep_id[done] = torch.arange(n_done, dtype=torch.float32) + next_id
    next_id += n_done
    step = torch.where(done, torch.zeros_like(step), step + 1)


def _spec() -> ReplaySpec:
  return ReplaySpec(state_dim=S, action_dim=A, xi_dim=D, num_reward_terms=R)


def test_add_layout_and_memory():
  N, T = 5, 16
  rb = SequenceReplay(N, T, _spec())
  assert rb.data["s"].shape == (N, T, S) and rb.data["s"].dtype == torch.float32
  assert rb.data["term"].dtype == torch.bool and rb.data["ep_step"].dtype == torch.int16
  assert rb.data["ref_idx"].dtype == torch.int32 and rb.data["r_terms"].shape == (N, T, R)
  floats = S + S + A + A + D + 1 + R
  assert rb.nbytes == N * T * (4 * floats + 1 + 1 + 2 + 4)
  _fill(rb, 3, seed=0, p_term=0.0, p_trunc=0.0)
  assert len(rb) == 3 * N and rb.size == 3 and rb.ptr == 3
  _fill(rb, 20, seed=0, p_term=0.0, p_trunc=0.0)
  assert rb.size == T and rb.ptr == (23 % T)


def test_window_sampler_no_cross_reset():
  N, T, steps, k, B = 8, 40, 113, 8, 4000
  rb = SequenceReplay(N, T, _spec())
  _fill(rb, steps, seed=1, p_term=0.06, p_trunc=0.04)
  w = rb.sample_windows(B, k, generator=torch.Generator().manual_seed(2))
  assert w.s0.shape == (B, S) and w.target.shape == (B, k, S) and w.a.shape == (B, k, A)
  assert w.mask.shape == (B, k) and w.xi.shape == (B, D) and w.r_terms.shape == (B, k, R)

  # Windows live entirely in the stored (not overwritten, already written) range.
  t0 = w.s0[:, 3]
  assert bool((t0 >= steps - T).all()) and bool((t0 + k - 1 <= steps - 1).all())

  # The physical slots hold consecutive global times (catches ring-wrap bugs).
  s_at = rb.data["s"][w.env_ids[:, None], w.slots]  # [B,k,S]
  assert torch.equal(s_at[..., 3], t0[:, None] + torch.arange(k))
  assert torch.equal(w.s0, s_at[:, 0])

  # Expected mask: step j is valid iff its start state is still in the start's episode.
  same_ep = s_at[..., 1] == w.s0[:, 1:2]
  assert torch.equal(w.mask, same_ep.float())
  assert bool((w.mask[:, 0] == 1).all())
  assert bool((w.mask == 0).any()), "test data should contain windows cut by a reset"
  # Mask is monotone (once 0, stays 0).
  assert bool((w.mask[:, 1:] <= w.mask[:, :-1]).all())

  # Valid targets are the pre-reset s_next of the same episode, exactly j+1 steps after s0.
  m = w.mask > 0
  j = torch.arange(k).expand(B, k)
  assert bool((w.target[..., 4][m] == 1).all())
  assert bool((w.target[..., 1][m] == w.s0[:, 1:2].expand(B, k)[m]).all())
  assert bool((w.target[..., 2][m] == (w.s0[:, 2:3] + j + 1)[m]).all())
  assert bool((w.target[..., 3][m] == (t0[:, None] + j + 1)[m]).all())

  # At terminal / truncated steps the target is the pre-reset state, not the next episode's start.
  done = (w.term | w.trunc) & m
  assert int(done.sum()) > 0
  assert bool((w.target[..., 1][done] == w.s0[:, 1:2].expand(B, k)[done]).all())
  assert bool((w.target[..., 2][done] > 0).all())

  # Per-step fields come from the same slots.
  assert torch.equal(w.a[..., 0], s_at[..., 1]) and torch.equal(w.a[..., 1], s_at[..., 2] + 0.5)
  assert torch.equal(w.ref_idx, s_at[..., 2].long() + REF_OFFSET)
  assert torch.equal(w.ep_step, s_at[..., 2].long())
  assert torch.equal(w.xi[:, 0], w.s0[:, 1])


def test_internal_ep_counter_matches_episode_steps():
  N, T = 6, 30
  rb = SequenceReplay(N, T, _spec())
  _fill(rb, 75, seed=3, p_term=0.05, p_trunc=0.05, give_ep_step=False)
  assert torch.equal(rb.data["ep_step"].long(), rb.data["s"][..., 2].long())


def test_sample_starts_history_padding():
  N, T, steps, B, H = 8, 40, 97, 3000, 10
  max_ref = REF_OFFSET + 14
  rb = SequenceReplay(N, T, _spec())
  _fill(rb, steps, seed=4, p_term=0.05, p_trunc=0.03, max_len=25)
  gen = torch.Generator().manual_seed(5)
  st = rb.sample_starts(B, hist_len=H, ref_valid=max_ref, generator=gen)
  assert st.hist_s.shape == (B, H, S) and st.hist_a_prev.shape == (B, H, A)
  assert st.hist_slots.shape == (B, H)

  # Start filter: no true termination after the start, reference inside the clip.
  assert not bool(rb.data["term"][st.env_ids, st.slot].any())
  assert bool((st.ref_idx <= max_ref).all())
  assert torch.equal(st.ep_step, st.s[:, 2].long())
  assert torch.equal(st.a_prev, rb.data["a_prev"][st.env_ids, st.slot])
  assert torch.equal(st.xi[:, 0], st.s[:, 1])

  # Coverage: both padded (short history) and full histories are sampled.
  assert bool((st.ep_step < H - 1).any()) and bool((st.ep_step >= H - 1).any())

  # Frame i (oldest -> newest) is the state min(H-1-i, ep_step) steps back in the same episode.
  back = torch.arange(H - 1, -1, -1)
  lookback = torch.minimum(back[None], st.ep_step[:, None]).float()
  assert torch.equal(st.hist_s[:, -1], st.s)
  assert bool((st.hist_s[..., 1] == st.s[:, 1:2]).all())
  assert torch.equal(st.hist_s[..., 2], st.s[:, 2:3] - lookback)
  assert torch.equal(st.hist_s[..., 3], st.s[:, 3:4] - lookback)
  # Padding repeats the episode's first frame (step 0, a_prev = 0).
  pad = back[None] >= st.ep_step[:, None]
  assert bool((st.hist_s[..., 2][pad] == 0).all())
  assert bool((st.hist_a_prev[pad] == 0).all())
  steps_h = st.hist_s[..., 2]
  prev = torch.stack((st.s[:, 1:2].expand(B, H), steps_h - 0.5), -1)
  expect_a_prev = torch.where((steps_h > 0)[..., None], prev, 0.0)
  assert torch.equal(st.hist_a_prev, expect_a_prev)

  # History frames are still stored (not overwritten by the ring).
  assert bool((st.hist_s[..., 3] >= steps - T).all())


def test_sample_starts_callable_filter_and_exhaustion():
  N, T = 4, 20
  rb = SequenceReplay(N, T, _spec())
  _fill(rb, 30, seed=6, p_term=0.0, p_trunc=0.0, max_len=8)
  st = rb.sample_starts(500, hist_len=4, ref_valid=lambda ref: ref % 2 == 0)
  assert bool((st.ref_idx % 2 == 0).all())
  try:
    rb.sample_starts(10, ref_valid=-1)
  except ValueError:
    pass
  else:
    raise AssertionError("expected ValueError when no start is valid")


def test_state_dict_roundtrip():
  N, T = 3, 12
  rb = SequenceReplay(N, T, _spec())
  _fill(rb, 17, seed=7, p_term=0.1, p_trunc=0.1)
  rb2 = SequenceReplay(N, T, _spec())
  rb2.load_state_dict(rb.state_dict())
  g1, g2 = torch.Generator().manual_seed(0), torch.Generator().manual_seed(0)
  w1, w2 = rb.sample_windows(64, 4, generator=g1), rb2.sample_windows(64, 4, generator=g2)
  assert torch.equal(w1.target, w2.target) and torch.equal(w1.mask, w2.mask)


def test_sample_starts_obs_frames_and_constructor_defaults():
  """With an obs_frame_fn, Start.hist holds clean frames built from the rebuilt (s, a_prev, ξ)."""
  N, T, H = 6, 30, 5

  def obs_frame(s, a_prev, xi):
    return torch.cat((s[..., 2:4], a_prev, xi), -1)  # [*, 2 + A + D]

  rb = SequenceReplay(N, T, _spec(), obs_frame_fn=obs_frame, hist_len=H, ref_valid=REF_OFFSET + 6)
  _fill(rb, 70, seed=8, p_term=0.05, p_trunc=0.05, max_len=12)
  st = rb.sample_starts(800, generator=torch.Generator().manual_seed(9))
  assert st.hist_s.shape == (800, H, S) and st.hist.shape == (800, H, 2 + A + D)
  xi_h = st.xi[:, None].expand(-1, H, -1)
  assert torch.equal(st.hist, obs_frame(st.hist_s, st.hist_a_prev, xi_h))
  assert bool((st.ref_idx <= REF_OFFSET + 6).all())
  assert torch.equal(st.hist[:, -1, :2], st.s[:, 2:4])
  # Without an obs_frame_fn the field stays empty.
  rb.obs_frame_fn = None
  assert rb.sample_starts(4).hist is None


def test_sample_starts_skip_blown_up_episodes():
  """A NaN state (physics blow-up, which no env termination catches) must never become an
  imagination start, nor sit in a start's rebuilt history; the next episode is usable again."""
  N, T, H = 3, 40, 4
  rb = SequenceReplay(N, T, _spec(), hist_len=H)
  g = torch.Generator().manual_seed(10)
  ep_step = torch.zeros(N, dtype=torch.long)
  for t in range(30):
    s = torch.stack([torch.full((N,), float(t)), ep_step.float(), *torch.zeros(4, N)], -1)
    if 12 <= t < 20:
      s[0, 4] = float("nan")  # env 0 blows up at t=12 and stays NaN until its episode ends
    trunc = torch.zeros(N, dtype=torch.bool)
    trunc[0] = t == 19  # env 0's episode ends at t=19; t=20 starts a fresh one
    a_prev = torch.randn(N, A, generator=g)
    a_prev[ep_step == 0] = 0.0
    rb.add(dict(
      s=s, s_next=s + 1, a_exec=torch.randn(N, A, generator=g), a_prev=a_prev,
      xi=torch.ones(N, D), r=torch.zeros(N), r_terms=torch.zeros(N, R),
      term=torch.zeros(N, dtype=torch.bool), trunc=trunc, ref_idx=ep_step, ep_step=ep_step,
    ))  # fmt: skip
    ep_step = torch.where(trunc, 0, ep_step + 1)
  # A NaN a_prev in env 1 at t=25 poisons that slot and every later slot of its episode.
  rb.data["a_prev"][1, 25, 0] = float("nan")
  rb._finite[1, 25:30] = False  # what add() records for such a step (written directly here)

  ok = rb.start_mask()
  assert bool(ok[0, :12].any()) and not bool(ok[0, 12:20].any()) and bool(ok[0, 20:30].all())
  assert bool(ok[1, :25].all()) and not bool(ok[1, 25:30].any()) and bool(ok[2, :30].all())
  st = rb.sample_starts(2000, generator=torch.Generator().manual_seed(11))
  assert bool(torch.isfinite(st.s).all()) and bool(torch.isfinite(st.hist_s).all())
  assert bool(torch.isfinite(st.a_prev).all()) and bool(torch.isfinite(st.hist_a_prev).all())
  assert bool(((st.env_ids == 0) & (st.slot >= 20)).any())  # env 0 recovered after its reset

  # The flag is recomputed from add(), not only set by hand: replay the poisoned a_prev step.
  rb2 = SequenceReplay(1, 8, _spec(), hist_len=2)
  for t in range(6):
    a_prev = torch.full((1, A), float("nan") if t == 2 else 0.0)
    rb2.add(dict(
      s=torch.zeros(1, S), s_next=torch.zeros(1, S), a_exec=torch.zeros(1, A), a_prev=a_prev,
      xi=torch.ones(1, D), r=torch.zeros(1), r_terms=torch.zeros(1, R),
      term=torch.zeros(1, dtype=torch.bool), trunc=torch.tensor([t == 3]),
      ref_idx=torch.zeros(1, dtype=torch.long),
    ))  # fmt: skip
  assert rb2._finite[0, :6].tolist() == [True, True, False, False, True, True]
  # state_dict carries the flag.
  rb3 = SequenceReplay(1, 8, _spec(), hist_len=2)
  rb3.load_state_dict(rb2.state_dict())
  assert torch.equal(rb3._finite, rb2._finite)
