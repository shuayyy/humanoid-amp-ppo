"""Unitree G1 whole-body toaster grasp environment configurations."""

from mjlab_husky.asset_zoo.robots.skateboard.g1_constants import (
  G1_29Dof_ACTION_SCALE,
  get_g1_29dof_robot_cfg,
  get_toaster_cfg,
)
from mjlab_husky.envs.g1_wb_grasp_rl_env import G1GraspManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab_husky.tasks.wb_grasp import mdp
from mjlab_husky.tasks.wb_grasp.wb_grasp_env_cfg import make_g1_wb_grasp_env_cfg


def unitree_g1_wb_grasp_env_cfg(play: bool = False) -> G1GraspManagerBasedRlEnvCfg:
  cfg = make_g1_wb_grasp_env_cfg()
  cfg.scene.num_envs = 1 if play else 1024
  cfg.sim.njmax = 1024
  cfg.sim.mujoco.ccd_iterations = 50
  cfg.sim.contact_sensor_maxmatch = 64
  cfg.sim.nconmax = 256

  cfg.scene.entities = {
    "robot": get_g1_29dof_robot_cfg(),
    # "toaster": get_toaster_cfg(),
  }
  #########################################################
  ##### terrain #####
  #########################################################
  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None

  site_names = ("left_foot", "right_foot")
  geom_names = tuple(
    f"{side}_foot{i}_collision" for side in ("left", "right") for i in range(1, 8)
  )

  #########################################################
  ##### contact sensors #####
  #########################################################

  self_collision_cfg = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    fields=("found","force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )

  feet_ground_cfg = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(
      mode="subtree",
      pattern=r"^(left_ankle_roll_link|right_ankle_roll_link)$",
      entity="robot",
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  )

  left_feet_ground_cfg = ContactSensorCfg(
    name="left_feet_ground_contact",
    primary=ContactMatch(
      mode="subtree",
      pattern=r"^(left_ankle_roll_link)$",
      entity="robot",
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  )

  right_feet_ground_cfg = ContactSensorCfg(
    name="right_feet_ground_contact",
    primary=ContactMatch(
      mode="subtree",
      pattern=r"^(right_ankle_roll_link)$",
      entity="robot",
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  )

  toaster_contact_cfg = ContactSensorCfg(
    name="toaster_contact",
    primary=ContactMatch(
      mode="geom",
      pattern=r"^(torso|left_shoulder_yaw|left_elbow_yaw|left_wrist|left_hand|right_shoulder_yaw|right_elbow_yaw|right_wrist|right_hand)_collision$",
      entity="robot",
    ),
    secondary=ContactMatch(
      mode="subtree",
      pattern="object",
      entity="toaster",
    ),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
  )

  illegal_toaster_contact_cfg = ContactSensorCfg(
    name="illegal_toaster_contact",
    primary=ContactMatch(
      mode="geom",
      pattern=r"^(left|right)_(hip|thigh|shin|linkage_brace|foot[1-7])_collision$",
      entity="robot",
    ),
    secondary=ContactMatch(
      mode="subtree",
      pattern="object",
      entity="toaster",
    ),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
  )

  left_hand_toaster_cfg = ContactSensorCfg(
    name="left_hand_toaster_contact",
    primary=ContactMatch(
      mode="geom",
      pattern=r"^(left_hand_collision|left_wrist_collision)$",
      entity="robot",
    ),
    secondary=ContactMatch(
      mode="geom",
      pattern=r"^left_grasp_marker_collision$",
      entity="toaster",
    ),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
  )
  
  right_hand_toaster_cfg = ContactSensorCfg(
    name="right_hand_toaster_contact",
    primary=ContactMatch(
      mode="geom",
      pattern=r"^(right_hand_collision|right_wrist_collision)$",
      entity="robot",
    ),
    secondary=ContactMatch(
      mode="geom",
      pattern=r"^right_grasp_marker_collision$",
      entity="toaster",
    ),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
  )

  illegal_ground_contact_cfg = ContactSensorCfg(
    name="illegal_ground_contact",
    primary=ContactMatch(
      mode="body",
      pattern=r"^(pelvis|torso_link|.*hip.*|.*knee.*|.*shoulder.*|.*elbow.*)$",
      entity="robot",
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found",),
    reduce="none",
    num_slots=1,
  )
  
  cfg.scene.sensors = (
    self_collision_cfg,
    feet_ground_cfg,
    left_feet_ground_cfg,
    right_feet_ground_cfg,
    # toaster_contact_cfg,
    # left_hand_toaster_cfg,
    # right_hand_toaster_cfg,
    # illegal_toaster_contact_cfg,
    illegal_ground_contact_cfg,
  )

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = G1_29Dof_ACTION_SCALE

  cfg.viewer.body_name = "torso_link"

  cfg.phase_ratios = [0.0, 0.5, 1.0]

  # Apply play mode overrides.
  if play:
    # Effectively infinite episode length.
    cfg.episode_length_s = int(60.0)
    cfg.eval_mode = True
    cfg.observations["policy"].enable_corruption = False
    cfg.terminations = {
      "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
    }
    cfg.events.pop("push_robot", None)
    # cfg.commands["skate"].ranges.lin_vel_x = (1.0, 1.0)  # pyright: ignore[reportAttributeAccessIssue]
    # cfg.commands["skate"].ranges.heading = (0.7, 0.7)  # pyright: ignore[reportAttributeAccessIssue]
  return cfg
