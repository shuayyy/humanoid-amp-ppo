"""SuperMimic actor (motion encoder + decoder) and critic (Table A.4, design §4).

Actor: a = dec(tanh(enc(norm g)), norm o_hist). Deterministic, linear unclipped output, last layer
scaled x0.01 so the initial policy sits near the default pose q0 (spec §9 item 24).
Critic: V(norm φ(ŝ), norm g, enc ξ, t_left), zero-initialized output.

The actor takes the history already flattened, in one fixed layout for sim, imagination and
deployment: features.PolicyObsHistory.get() / flatten_history (mjlab term-major by default, the
same as the env's actor_hist group), which imagination reproduces through ImagFns.flatten_hist.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import torch
from torch import nn

_ACTIVATIONS = {"elu": nn.ELU, "silu": nn.SiLU, "relu": nn.ReLU, "tanh": nn.Tanh}


def mlp(in_dim: int, hidden: Sequence[int], out_dim: int, activation: str = "elu") -> nn.Sequential:
  act = _ACTIVATIONS[activation]
  layers: list[nn.Module] = []
  for h in hidden:
    layers += [nn.Linear(in_dim, h), act()]
    in_dim = h
  layers.append(nn.Linear(in_dim, out_dim))
  return nn.Sequential(*layers)


class RunningNorm(nn.Module):
  """Running mean/std normalizer: (x - μ) / (σ + eps), same math as rsl_rl EmpiricalNormalization.

  Unlike rsl_rl's, `update` ignores train/eval mode: the learner calls it explicitly on sim data,
  and forward never touches the stats (frozen inside every update, gradients pass through).
  """

  def __init__(self, dim: int, eps: float = 1e-2):
    super().__init__()
    self.eps = eps
    self.register_buffer("mean", torch.zeros(dim))
    self.register_buffer("var", torch.ones(dim))
    self.register_buffer("count", torch.zeros((), dtype=torch.float64))

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return (x - self.mean) / (self.var.sqrt() + self.eps)

  @torch.no_grad()
  def update(self, x: torch.Tensor) -> None:
    """Fold a batch into the stats. Rows with any non-finite entry are skipped: one NaN state
    from a physics blow-up would otherwise poison the stats (and every later action) for good.
    Masked sums instead of boolean indexing keep it free of host syncs on GPU."""
    x = x.reshape(-1, self.mean.shape[0]).to(self.mean)
    if x.shape[0] == 0:
      return
    ok = torch.isfinite(x).all(-1, keepdim=True)
    w = ok.to(x.dtype)
    x = torch.where(ok, x, torch.zeros_like(x))
    n = w.sum()
    n_safe = n.clamp(min=1.0)
    mean_x = (w * x).sum(0) / n_safe
    var_x = (w * (x - mean_x).square()).sum(0) / n_safe
    self.count += n.to(self.count.dtype)
    rate = (n.to(self.count.dtype) / self.count.clamp(min=1.0)).to(self.mean)  # 0 if all skipped
    delta = mean_x - self.mean
    self.mean += rate * delta
    self.var += rate * (var_x - self.var + delta * (mean_x - self.mean))


class MotionEncoder(nn.Module):
  """K future reference frames -> z (32, tanh-bounded)."""

  def __init__(
    self,
    goal_dim: int,
    hidden: Sequence[int] = (2048, 1024, 512, 256),
    latent_dim: int = 32,
    activation: str = "elu",
  ):
    super().__init__()
    self.net = mlp(goal_dim, hidden, latent_dim, activation)

  def forward(self, goal: torch.Tensor) -> torch.Tensor:
    return torch.tanh(self.net(goal))


class Decoder(nn.Module):
  """(z, 10-frame observation history) -> action (linear, unclipped)."""

  def __init__(
    self,
    latent_dim: int,
    hist_dim: int,
    action_dim: int = 29,
    hidden: Sequence[int] = (2048, 1024, 512, 256),
    activation: str = "elu",
    out_init_scale: float = 0.01,
  ):
    super().__init__()
    self.net = mlp(latent_dim + hist_dim, hidden, action_dim, activation)
    last = self.net[-1]
    assert isinstance(last, nn.Linear)
    with torch.no_grad():
      last.weight.mul_(out_init_scale)
      last.bias.mul_(out_init_scale)

  def forward(self, z: torch.Tensor, hist: torch.Tensor) -> torch.Tensor:
    return self.net(torch.cat((z, hist), dim=-1))


class Actor(nn.Module):
  """π(o_hist, g): deterministic; input normalizers are part of the module (and its checkpoint)."""

  def __init__(
    self,
    hist_dim: int,
    goal_dim: int,
    action_dim: int = 29,
    encoder_dims: Sequence[int] = (2048, 1024, 512, 256),
    decoder_dims: Sequence[int] = (2048, 1024, 512, 256),
    latent_dim: int = 32,
    activation: str = "elu",
    out_init_scale: float = 0.01,
  ):
    super().__init__()
    self.hist_dim, self.goal_dim, self.action_dim = hist_dim, goal_dim, action_dim
    self.hist_norm = RunningNorm(hist_dim)
    self.goal_norm = RunningNorm(goal_dim)
    self.encoder = MotionEncoder(goal_dim, encoder_dims, latent_dim, activation)
    self.decoder = Decoder(latent_dim, hist_dim, action_dim, decoder_dims, activation, out_init_scale)

  @classmethod
  def from_cfg(cls, policy_cfg: Any, hist_dim: int, goal_dim: int, action_dim: int = 29) -> "Actor":
    """Build from SuperMimicPolicyCfg (config.py field names)."""
    return cls(
      hist_dim,
      goal_dim,
      action_dim,
      encoder_dims=tuple(policy_cfg.encoder_dims),
      decoder_dims=tuple(policy_cfg.decoder_dims),
      latent_dim=policy_cfg.latent_dim,
      activation=policy_cfg.activation,
      out_init_scale=policy_cfg.out_init_scale,
    )

  def forward(self, hist: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
    """hist [*, hist_dim] (flattened history), goal [*, goal_dim] -> action [*, action_dim]."""
    if hist.shape[-1] != self.hist_dim or goal.shape[-1] != self.goal_dim:
      raise ValueError(
        f"actor expects hist [*, {self.hist_dim}] and goal [*, {self.goal_dim}], "
        f"got {tuple(hist.shape)} and {tuple(goal.shape)}"
      )
    z = self.encoder(self.goal_norm(goal))
    return self.decoder(z, self.hist_norm(hist))

  def latent(self, goal: torch.Tensor) -> torch.Tensor:
    return self.encoder(self.goal_norm(goal))

  def update_normalizers(self, hist: torch.Tensor, goal: torch.Tensor) -> None:
    """Feed sim observations (never imagined ones) to the running stats."""
    self.hist_norm.update(hist.reshape(-1, self.hist_dim))
    self.goal_norm.update(goal.reshape(-1, self.goal_dim))


class Critic(nn.Module):
  """V(φ(ŝ), g, enc ξ, t_left) -> [*]; MLP (256, 128) ELU, zero-init output."""

  def __init__(
    self,
    phi_dim: int,
    goal_dim: int,
    xi_dim: int,
    hidden: Sequence[int] = (256, 128),
    activation: str = "elu",
  ):
    super().__init__()
    self.phi_dim, self.goal_dim, self.xi_dim = phi_dim, goal_dim, xi_dim
    self.phi_norm = RunningNorm(phi_dim)
    self.goal_norm = RunningNorm(goal_dim)
    self.net = mlp(phi_dim + goal_dim + xi_dim + 1, hidden, 1, activation)
    last = self.net[-1]
    assert isinstance(last, nn.Linear)
    nn.init.zeros_(last.weight)
    nn.init.zeros_(last.bias)

  @classmethod
  def from_cfg(cls, critic_cfg: Any, phi_dim: int, goal_dim: int, xi_dim: int) -> "Critic":
    return cls(phi_dim, goal_dim, xi_dim, tuple(critic_cfg.hidden_dims), critic_cfg.activation)

  def forward(
    self, phi: torch.Tensor, goal: torch.Tensor, xi_enc: torch.Tensor, t_left: torch.Tensor
  ) -> torch.Tensor:
    lead = phi.shape[:-1]
    xi_enc = xi_enc.expand(*lead, self.xi_dim)  # one ξ per rollout, shared across time
    t_left = t_left.to(phi).expand(lead)[..., None]
    x = torch.cat((self.phi_norm(phi), self.goal_norm(goal), xi_enc, t_left), dim=-1)
    return self.net(x).squeeze(-1)

  def update_normalizers(self, phi: torch.Tensor, goal: torch.Tensor) -> None:
    self.phi_norm.update(phi.reshape(-1, self.phi_dim))
    self.goal_norm.update(goal.reshape(-1, self.goal_dim))


class ActorInference(nn.Module):
  """Play/eval/ONNX wrapper: forward(hist, goal), or forward(obs) with obs a dict/TensorDict
  holding the env's actor_hist / actor_goal groups (same layout the actor was trained on)."""

  def __init__(self, actor: Actor, hist_key: str = "actor_hist", goal_key: str = "actor_goal"):
    super().__init__()
    self.actor = actor
    self.hist_key, self.goal_key = hist_key, goal_key

  def forward(self, hist: Any, goal: torch.Tensor | None = None) -> torch.Tensor:
    if goal is None:
      obs: Mapping[str, torch.Tensor] = hist
      hist, goal = obs[self.hist_key], obs[self.goal_key]
    return self.actor(hist, goal)

  @torch.no_grad()
  def act(self, obs_hist: Any, goal: torch.Tensor | None = None) -> torch.Tensor:
    return self.forward(obs_hist, goal)
