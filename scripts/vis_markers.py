from __future__ import annotations

import mujoco
import mujoco.viewer

from mjlab.entity import EntityCfg
from mjlab_husky.asset_zoo.robots.skateboard.g1_constants import (
  get_g1_29dof_robot_cfg,
  get_toaster_cfg,
)
from mjlab.scene import Scene, SceneCfg
from mjlab.terrains import TerrainImporterCfg


def _print_foot_sites(model: mujoco.MjModel) -> None:
  foot_site_names = [
    "left_foot",
    "left_foot_1",
    "left_foot_2",
    "left_foot_3",
    "left_foot_4",
    "right_foot",
    "right_foot_1",
    "right_foot_2",
    "right_foot_3",
    "right_foot_4",
  ]
  print("Foot sites:")
  for site_name in foot_site_names:
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    if site_id < 0:
      print(f"  {site_name}: not found")
      continue
    rgba = tuple(float(v) for v in model.site_rgba[site_id])
    pos = tuple(float(v) for v in model.site_pos[site_id])
    print(f"  {site_name}: id={site_id}, pos={pos}, rgba={rgba}")


def _print_grasp_mapping(model: mujoco.MjModel, data: mujoco.MjData) -> None:
  site_names = [
    "robot/left_palm",
    "robot/right_palm",
    "toaster/left_grasp_marker",
    "toaster/right_grasp_marker",
  ]
  print("Grasp sites:")
  site_ids: dict[str, int] = {}
  for site_name in site_names:
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    site_ids[site_name] = site_id
    if site_id < 0:
      print(f"  {site_name}: not found")
      continue
    pos = tuple(float(v) for v in data.site_xpos[site_id])
    rgba = tuple(float(v) for v in model.site_rgba[site_id])
    print(f"  {site_name}: id={site_id}, world_pos={pos}, rgba={rgba}")

  pairs = [
    ("robot/left_palm", "toaster/left_grasp_marker"),
    ("robot/left_palm", "toaster/right_grasp_marker"),
    ("robot/right_palm", "toaster/left_grasp_marker"),
    ("robot/right_palm", "toaster/right_grasp_marker"),
  ]
  print("Hand-to-marker distances:")
  for site_a, site_b in pairs:
    site_a_id = site_ids[site_a]
    site_b_id = site_ids[site_b]
    if site_a_id < 0 or site_b_id < 0:
      print(f"  {site_a} -> {site_b}: unavailable")
      continue
    distance = float(
      ((data.site_xpos[site_a_id] - data.site_xpos[site_b_id]) ** 2).sum() ** 0.5
    )
    print(f"  {site_a} -> {site_b}: {distance:.4f}")


def _apply_freejoint_initial_state(
  model: mujoco.MjModel,
  data: mujoco.MjData,
  joint_name: str,
  init_state: EntityCfg.InitialStateCfg,
) -> None:
  joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
  if joint_id < 0:
    raise ValueError(f"Joint '{joint_name}' not found in compiled scene.")

  qpos_adr = model.jnt_qposadr[joint_id]
  data.qpos[qpos_adr : qpos_adr + 3] = init_state.pos
  data.qpos[qpos_adr + 3 : qpos_adr + 7] = init_state.rot


def main() -> None:
  robot_cfg = get_g1_29dof_robot_cfg()
  toaster_cfg = get_toaster_cfg()
  scene_cfg = SceneCfg(
    terrain=TerrainImporterCfg(terrain_type="plane"),
    entities={
      "robot": robot_cfg,
      "toaster": toaster_cfg,
    },
    num_envs=1,
    extent=2.0,
  )
  scene = Scene(scene_cfg, device="cpu")
  model = scene.compile()
  data = mujoco.MjData(model)
  _apply_freejoint_initial_state(
    model,
    data,
    "robot/floating_base_joint",
    robot_cfg.init_state,
  )
  _apply_freejoint_initial_state(
    model,
    data,
    "toaster/object_freejoint",
    toaster_cfg.init_state,
  )
  mujoco.mj_forward(model, data)
  _print_foot_sites(model)
  _print_grasp_mapping(model, data)

  with mujoco.viewer.launch_passive(model, data) as viewer:
    viewer.opt.sitegroup[:] = 1
    while viewer.is_running():
      viewer.sync()


if __name__ == "__main__":
  main()
