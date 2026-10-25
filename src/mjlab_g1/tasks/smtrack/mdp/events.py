from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.managers.event_manager import RecomputeLevel, requires_model_fields
from mjlab.utils.lab_api.math import sample_uniform

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.managers.scene_entity_config import SceneEntityCfg


@requires_model_fields("body_mass", "body_inertia", recompute=RecomputeLevel.set_const)
def body_mass_scale(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  scale_range: tuple[float, float],
  asset_cfg: SceneEntityCfg,
) -> None:
  """Scale mass and inertia together by s ~ U(lo, hi) from the defaults (uniform density).

  dr.body_mass leaves the inertia unchanged; dr.pseudo_inertia samples log-scale, so
  neither gives the brief's U(0.8, 1.2) x nominal mass for a homogeneous box.
  """
  asset = env.scene[asset_cfg.name]
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device)
  body_ids = asset.indexing.body_ids[asset_cfg.body_ids]
  if body_ids.ndim == 0:
    body_ids = body_ids[None]
  scale = sample_uniform(
    torch.tensor(scale_range[0], device=env.device),
    torch.tensor(scale_range[1], device=env.device),
    (len(env_ids), 1),
    env.device,
  )
  env_grid, body_grid = torch.meshgrid(env_ids, body_ids.long(), indexing="ij")
  mass0 = env.sim.get_default_field("body_mass")[body_ids.long()]
  inertia0 = env.sim.get_default_field("body_inertia")[body_ids.long()]
  env.sim.model.body_mass[env_grid, body_grid] = mass0[None] * scale
  env.sim.model.body_inertia[env_grid, body_grid] = inertia0[None] * scale[..., None]
