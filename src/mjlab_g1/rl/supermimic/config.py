"""SuperMimic runner/algorithm configuration.

Paper values cite Table A.4 / A.5 of SuperMimic (arXiv 2610.28480); `# ours` marks fills for
items the paper leaves unspecified (see docs/supermimic/design_learner.md and the brief §4).
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field
from typing import Any, Literal

from mjlab_g1.rl.config import RslRlBaseRunnerCfg


@dataclass
class SuperMimicWMCfg:
  """World model f_ω + learned termination head (design §3)."""

  hidden_dims: tuple[int, ...] = (2048, 2048, 2048)  # Table A.4
  activation: str = "silu_ln"  # ours: Linear -> LayerNorm -> SiLU
  term_hidden_dims: tuple[int, ...] = (256, 128)  # ours (BCE head)
  window_k: int = 8  # §3.2, Eq. 1
  num_batches: int = 10  # Table A.5 "10 batches of 2048 windows" = 10 Adam steps (ours)
  batch_windows: int = 2048  # Table A.5
  learning_rate: float = 2e-4  # Table A.5 (1 GPU)
  betas: tuple[float, float] = (0.9, 0.999)  # ours (paper states β only for the policy)
  eps: float = 1e-8
  max_grad_norm: float = 1.0  # Table A.5
  # Critique M2 (ours, opt-in): stop updating φ / σ_out / σ_nom once this many iterations are
  # done, so σ_out no longer changes what the trained output layer means. The prefill always
  # updates them (0 = freeze right after the prefill). None = cumulative statistics for the
  # whole run (design §3.4).
  freeze_normalizers_after: int | None = None


@dataclass
class SuperMimicPolicyCfg:
  """Actor (motion encoder + decoder) and BPTT imagination (design §4.1)."""

  horizon: int = 32  # Table A.5
  num_imagined: int = 4096  # Table A.5
  gamma: float = 0.99  # Table A.5
  learning_rate: float = 2e-4  # = α_ω (Table A.5)
  betas: tuple[float, float] = (0.9, 0.95)  # Table A.5
  eps: float = 1e-8
  max_grad_norm: float = 1.0  # Table A.5, applied first
  elementwise_clip: float = 0.1  # Table A.5, applied after the global clip
  encoder_dims: tuple[int, ...] = (2048, 1024, 512, 256)  # Table A.4
  decoder_dims: tuple[int, ...] = (2048, 1024, 512, 256)  # Table A.4
  latent_dim: int = 32  # Table A.4 (tanh)
  activation: str = "elu"  # ours
  future_frames: int = 10  # K, App. A
  obs_history: int = 10  # App. A
  out_init_scale: float = 0.01  # ours: start near q0


@dataclass
class SuperMimicCriticCfg:
  """Critic V(φ(ŝ), g, ξ, t_left) fit on imagined λ-returns (Eq. 3)."""

  hidden_dims: tuple[int, ...] = (256, 128)  # Table A.4
  activation: str = "elu"  # ours
  lam: float = 0.95  # Table A.5
  num_epochs: int = 5  # Table A.5
  num_mini_batches: int = 4  # Table A.5
  learning_rate: float = 2e-4  # = α_ω
  betas: tuple[float, float] = (0.9, 0.999)  # ours (PyTorch default)
  eps: float = 1e-8
  max_grad_norm: float = 1.0  # Table A.5
  # ours: t_left = clamp(clip_last - k, 0, h) / h. None = h is the clip length: with h = 64 the
  # input saturated for 2/3 (walk) to 7/8 (lift) of the clip, although the value of 64 vs 500
  # steps left differs ~2x at γ = 0.99 (critique M7).
  t_left_horizon: int | None = None


@dataclass
class SuperMimicMemCfg:
  """Memory / speed knobs (design §7.3). None of these change the math except bf16 (absent)."""

  imag_microbatches: int = 1  # exact gradient accumulation over imagined micro-batches
  checkpoint_steps: bool = False  # reserved (design §7.3 fallback 2); not wired into imagination
  compile: bool = True  # torch.compile WM/actor/critic MLPs + FK in place (user preference, 2026-10-22)
  tf32: bool = True  # Fig. 5 caption: TF32 training


@dataclass
class SuperMimicRunnerCfg(RslRlBaseRunnerCfg):
  class_name: str = "SuperMimicRunner"
  experiment_name: str = "smtrack"
  logger: Literal["wandb", "tensorboard"] = "tensorboard"
  max_iterations: int = 320_000  # Table A.5
  num_steps_per_env: int = 3  # informational only (ceil of env_steps_per_iter); train.py reads it
  save_interval: int = 5000  # ours: ~220 MB per checkpoint at paper widths
  log_interval: int = 50  # ours: metrics.jsonl / tensorboard cadence (averaged over the interval)

  # Data collection.
  env_steps_per_iter: float = 2.5  # UTD 0.4, reading (a): fractional accumulator 2,3,2,3,...
  prefill_steps: int = 32  # ours: vectorized steps with the init actor before iteration 0
  action_noise_std: float = 0.1  # Table A.5, ε ~ N(0, 0.1²); the executed action is stored
  replay_steps_per_env: int = 1000  # Table A.5
  clip_actions: float | None = None  # must stay None: clipping breaks reachability (integration §4.4)
  save_replay: bool = False  # ours: not implemented in v1 (replay is re-collected on resume)

  # Shared learning-rate schedule (all three optimizers).
  lr_final_frac: float = 0.03  # Table A.5
  lr_cosine_frac: float = 0.78  # brief §4: 250k / 320k of the run, keeps the paper's shape
  lr_cosine_iters: int | None = None  # explicit override of lr_cosine_frac * max_iterations

  # Reference pairing and clip end.
  reward_ref_offset: int = 0  # mjlab pairing (brief §1)
  clip_end: Literal["stop", "bootstrap"] = "stop"  # ours (design D4)

  # Startup parity checks (brief §4).
  parity_check: bool = True
  parity_transitions: int = 200  # per-env-step transitions checked before iteration 0
  parity_reward_tol: float = 1e-3
  # Analytic vs env `terminated`: abort beyond max(1, (1 - acc) x #terminations) mismatches.
  # Thresholds come from the env's termination cfg (reward.TrackingRewardCfg.from_env).
  parity_term_min_acc: float = 0.99

  wm: SuperMimicWMCfg = field(default_factory=SuperMimicWMCfg)
  policy: SuperMimicPolicyCfg = field(default_factory=SuperMimicPolicyCfg)
  critic: SuperMimicCriticCfg = field(default_factory=SuperMimicCriticCfg)
  memory: SuperMimicMemCfg = field(default_factory=SuperMimicMemCfg)
  algorithm: dict = field(
    default_factory=lambda: {
      "class_name": "mjlab_g1.rl.supermimic.algorithm:SuperMimic",
      "rnd_cfg": None,  # rsl_rl Logger reads this key
    }
  )

  def cosine_iters(self) -> int:
    """Length of the cosine segment; computed at use time (tyro sets max_iterations late)."""
    if self.lr_cosine_iters is not None:
      return max(1, int(self.lr_cosine_iters))
    return max(1, round(self.lr_cosine_frac * self.max_iterations))

  def lr_factor(self, it: int) -> float:
    """Cosine 1 -> lr_final_frac over cosine_iters(), then constant (Table A.5)."""
    T = self.cosine_iters()
    x = min(max(it, 0), T) / T
    f = self.lr_final_frac
    return f + (1.0 - f) * 0.5 * (1.0 + math.cos(math.pi * x))

  @classmethod
  def smoke(cls, **overrides: Any) -> "SuperMimicRunnerCfg":
    """CPU smoke config: widths /16, H=4, k=2, batch 16, n_imag 16, 3 iterations (brief §4)."""
    cfg = cls(
      max_iterations=3,
      save_interval=1,
      log_interval=1,
      prefill_steps=16,
      replay_steps_per_env=64,
      parity_transitions=16,
      wm=SuperMimicWMCfg(
        hidden_dims=(128, 128, 128),
        term_hidden_dims=(16, 8),
        window_k=2,
        num_batches=2,
        batch_windows=16,
      ),
      policy=SuperMimicPolicyCfg(
        horizon=4,
        num_imagined=16,
        encoder_dims=(128, 64, 32, 16),
        decoder_dims=(128, 64, 32, 16),
      ),
      critic=SuperMimicCriticCfg(hidden_dims=(16, 8), num_epochs=2, num_mini_batches=2),
      memory=SuperMimicMemCfg(compile=False),
    )
    return dataclasses.replace(cfg, **overrides) if overrides else cfg

  @classmethod
  def from_dict(cls, d: dict) -> "SuperMimicRunnerCfg":
    """Rebuild from `dataclasses.asdict` output (train.py / play.py pass dicts)."""
    return _from_dict(cls, d)

  def to_dict(self) -> dict:
    return dataclasses.asdict(self)


def as_runner_cfg(cfg: "SuperMimicRunnerCfg | dict") -> SuperMimicRunnerCfg:
  return cfg if isinstance(cfg, SuperMimicRunnerCfg) else SuperMimicRunnerCfg.from_dict(cfg)


def _from_dict(cls: type, d: dict) -> Any:
  # Unknown keys (e.g. extra script fields) are ignored; tuples come back as lists from yaml.
  kwargs = {}
  for f in dataclasses.fields(cls):
    if f.name not in d:
      continue
    v = d[f.name]
    default = f.default_factory() if f.default_factory is not dataclasses.MISSING else f.default
    if dataclasses.is_dataclass(default) and isinstance(v, dict):
      v = _from_dict(type(default), v)
    elif isinstance(default, tuple) and isinstance(v, list):
      v = tuple(v)
    kwargs[f.name] = v
  return cls(**kwargs)
