"""RSL-RL actor with SuperMimic's motion-encoder / decoder layout.

The paper's PPO-Sim baseline shares the actor architecture with SuperMimic
(Table A.4): a motion encoder over the K future reference frames, squashed to a
tanh-bounded latent, and a decoder over (latent, observation history). This
model gives rsl_rl's PPO that architecture while keeping MLPModel's
normalization, distribution and export plumbing.

Observation groups must be ordered (history, goal): the last ``goal_group``
columns of the concatenated input are the goal.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from tensordict import TensorDict

from rsl_rl.models.mlp_model import MLPModel
from rsl_rl.modules import MLP


class _EncDecNet(nn.Module):
  def __init__(
    self,
    hist_dim: int,
    goal_dim: int,
    output_dim: int,
    encoder_dims: tuple[int, ...],
    decoder_dims: tuple[int, ...],
    latent_dim: int,
    activation: str,
  ) -> None:
    super().__init__()
    self.hist_dim = hist_dim
    self.goal_dim = goal_dim
    self.encoder = MLP(goal_dim, latent_dim, encoder_dims, activation)
    self.decoder = MLP(latent_dim + hist_dim, output_dim, decoder_dims, activation)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    hist, goal = x[..., : self.hist_dim], x[..., self.hist_dim :]
    z = torch.tanh(self.encoder(goal))
    return self.decoder(torch.cat((z, hist), dim=-1))

  def __getitem__(self, idx: int) -> nn.Module:
    # Distribution weight init addresses the output layers as mlp[-k].
    return self.decoder[idx]


class EncDecModel(MLPModel):
  """MLPModel whose network is SuperMimic's encoder / decoder."""

  def __init__(
    self,
    obs: TensorDict,
    obs_groups: dict[str, list[str]],
    obs_set: str,
    output_dim: int,
    hidden_dims: tuple[int, ...] | list[int] = (2048, 1024, 512, 256),
    activation: str = "elu",
    obs_normalization: bool = False,
    distribution_cfg: dict | None = None,
    encoder_dims: tuple[int, ...] | list[int] | None = None,
    decoder_dims: tuple[int, ...] | list[int] | None = None,
    latent_dim: int = 32,
    goal_group: str = "actor_goal",
  ) -> None:
    # MLPModel's hidden_dims set both widths unless given separately (Table A.4 uses
    # the same (2048, 1024, 512, 256) for encoder and decoder).
    encoder_dims = tuple(encoder_dims or hidden_dims)
    decoder_dims = tuple(decoder_dims or hidden_dims)
    groups = list(obs_groups[obs_set])
    if groups[-1] != goal_group:
      raise ValueError(
        f"EncDecModel needs '{goal_group}' as the last observation group of '{obs_set}', got {groups}."
      )
    super().__init__(
      obs,
      obs_groups,
      obs_set,
      output_dim,
      hidden_dims=(8,),
      activation=activation,
      obs_normalization=obs_normalization,
      distribution_cfg=distribution_cfg,
    )
    goal_dim = obs[goal_group].shape[-1]
    mlp_output_dim = self.distribution.input_dim if self.distribution is not None else output_dim
    self.mlp = _EncDecNet(
      hist_dim=self.obs_dim - goal_dim,
      goal_dim=goal_dim,
      output_dim=mlp_output_dim,
      encoder_dims=encoder_dims,
      decoder_dims=decoder_dims,
      latent_dim=latent_dim,
      activation=activation,
    )
    if self.distribution is not None:
      self.distribution.init_mlp_weights(self.mlp)
