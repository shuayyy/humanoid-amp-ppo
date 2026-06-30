"""Project event helpers missing from MJLab 1.4."""

from __future__ import annotations

from collections.abc import Mapping

import torch

from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.event_manager import RecomputeLevel, requires_model_fields
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import sample_uniform


def _resolve_global_ids(
    asset: Entity,
    asset_cfg: SceneEntityCfg,
    field: str,
) -> torch.Tensor:
    if field.startswith("body_"):
        local_ids = asset_cfg.body_ids
        global_ids = asset.indexing.body_ids
    elif field.startswith("geom_"):
        local_ids = asset_cfg.geom_ids
        global_ids = asset.indexing.geom_ids
    else:
        raise ValueError(f"Unsupported randomized model field: {field!r}")

    if isinstance(local_ids, slice):
        return global_ids[local_ids].long()
    return global_ids[torch.tensor(local_ids, device=global_ids.device)].long()


def _select_defaults(
    env: ManagerBasedRlEnv,
    field: str,
    env_ids: torch.Tensor,
    global_ids: torch.Tensor,
) -> torch.Tensor:
    defaults = env.sim.get_default_field(field)
    if defaults.dim() == 2:
        return defaults[global_ids].unsqueeze(0).repeat(len(env_ids), 1, 1)
    if defaults.dim() == 3:
        return defaults[env_ids[:, None], global_ids]
    raise ValueError(f"Unsupported default field rank for {field!r}: {defaults.shape}")


def _sample_offsets(
    ranges: tuple[float, float] | Mapping[int, tuple[float, float]],
    shape: torch.Size,
    device: str,
) -> torch.Tensor:
    if isinstance(ranges, Mapping):
        samples = torch.zeros(shape, device=device)
        for dim, value_range in ranges.items():
            samples[..., int(dim)] = sample_uniform(
                value_range[0],
                value_range[1],
                shape[:-1],
                device=device,
            )
        return samples

    return sample_uniform(ranges[0], ranges[1], shape, device=device)


@requires_model_fields(
    "body_ipos",
    "geom_friction",
    recompute=RecomputeLevel.set_const,
)
def randomize_field(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    operation: str,
    field: str,
    ranges: tuple[float, float] | Mapping[int, tuple[float, float]],
) -> None:
    """Randomize a MuJoCo model field using the old project config contract."""
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    else:
        env_ids = env_ids.long()

    asset: Entity = env.scene[asset_cfg.name]
    global_ids = _resolve_global_ids(asset, asset_cfg, field)
    defaults = _select_defaults(env, field, env_ids, global_ids)
    samples = _sample_offsets(ranges, defaults.shape, env.device)

    if operation == "add":
        values = defaults + samples
    elif operation == "scale":
        values = defaults * samples
    elif operation == "abs":
        values = samples
    else:
        raise ValueError(f"Unsupported randomize_field operation: {operation!r}")

    model_field = getattr(env.sim.model, field)
    model_field[env_ids[:, None], global_ids] = values
