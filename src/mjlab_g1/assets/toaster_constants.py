"""Toaster asset constants."""

from pathlib import Path
import os

import mujoco

from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg

TOASTER_XML: Path = Path(
  os.path.join(os.path.dirname(__file__), "Toaster003", "model.xml")
)
assert TOASTER_XML.exists()


def get_toaster_assets() -> dict[str, bytes]:
  assets: dict[str, bytes] = {}
  for asset_path in TOASTER_XML.parent.rglob("*"):
    if asset_path.is_file():
      assets[str(asset_path.relative_to(TOASTER_XML.parent))] = asset_path.read_bytes()
  return assets


def get_toaster_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(str(TOASTER_XML))
  spec.assets = get_toaster_assets()
  return spec


TOASTER_INIT_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0.375, 0.1, 0.125),
  rot=(0.70710678, 0.0, 0.0, 0.70710678),
  joint_pos={".*": 0.0},
  joint_vel={".*": 0.0},
)


FULL_COLLISION_TOASTER = CollisionCfg(
  geom_names_expr=(".*",),
)


TOASTER_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(),
  soft_joint_pos_limit_factor=1.0,
)


def get_toaster_cfg() -> EntityCfg:
  """Get a fresh toaster configuration instance."""
  return EntityCfg(
    init_state=TOASTER_INIT_KEYFRAME,
    collisions=(FULL_COLLISION_TOASTER,),
    spec_fn=get_toaster_spec,
    articulation=TOASTER_ARTICULATION,
  )
