"""Unitree G1 locomotion environment configurations."""

from mjlab_g1.assets.g1_constants import (
  G1_29Dof_ACTION_SCALE,
  get_g1_29dof_robot_cfg,
)
from mjlab_g1.envs.g1_locomotion_rl_env import G1LocomotionManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab_g1.tasks.locomotion import mdp
from mjlab_g1.tasks.locomotion.locomotion_env_cfg import (
  make_g1_locomotion_env_cfg,
)


def unitree_g1_locomotion_env_cfg(play: bool = False) -> G1LocomotionManagerBasedRlEnvCfg:
  cfg = make_g1_locomotion_env_cfg()
  cfg.scene.num_envs = 1 if play else 1024
  cfg.sim.njmax = 1024
  cfg.sim.mujoco.ccd_iterations = 50
  cfg.sim.contact_sensor_maxmatch = 64
  cfg.sim.nconmax = 256

  cfg.scene.entities = {
    "robot": get_g1_29dof_robot_cfg(),
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
    illegal_ground_contact_cfg,
  )

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = G1_29Dof_ACTION_SCALE

  cfg.viewer.body_name = "torso_link"

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
    # Viser velocity sliders require a non-zero max value.
    cfg.commands["gostraight"].ranges.lin_vel_y = (-0.1, 0.1)
    cfg.commands["gostraight"].ranges.ang_vel_z = (-0.1, 0.1)

  return cfg
