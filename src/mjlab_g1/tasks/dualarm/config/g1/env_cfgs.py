"""Unitree G1 dual-arm manipulation environment configurations."""

from mjlab_g1.assets.g1_constants import (
  G1_29Dof_ACTION_SCALE,
  get_g1_29dof_robot_cfg,
)
from mjlab_g1.assets.toaster_constants import get_toaster_cfg
from mjlab_g1.envs.g1_dualarm_rl_env import G1DualarmManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab_g1.tasks.dualarm import mdp
from mjlab_g1.tasks.dualarm.dual_arm_env_cfg import (
  make_g1_dualarm_env_cfg,
)


# Dual-arm action scale, sized from the mocap clips instead of actuator limits.
G1_29Dof_DUALARM_ACTION_SCALE: dict[str, float] = {
  **{k: v for k, v in G1_29Dof_ACTION_SCALE.items() if k != r".*_wrist_.*_joint"},
  # Raised: required > actuator bound.
  r".*_knee_joint": 1.79,  # was 0.351, mocap needs 1.49
  r".*_hip_pitch_joint": 1.75,  # was 0.548, mocap needs 1.46
  r".*_shoulder_pitch_joint": 1.27,  # was 0.439, mocap needs 1.06
  r".*_elbow_joint": 1.19,  # was 0.439, mocap needs 0.99
  r".*_shoulder_roll_joint": 0.70,  # was 0.439, mocap needs 0.59
  r".*_shoulder_yaw_joint": 0.62,  # was 0.439, mocap needs 0.51
  r".*_ankle_pitch_joint": 0.61,  # was 0.439, mocap needs 0.51
  # Wrists are split per axis because each axis needs a different range.
  r".*_wrist_roll_joint": 1.19,  # mocap needs 0.99
  r".*_wrist_yaw_joint": 0.85,  # mocap needs 0.71
  r".*_wrist_pitch_joint": 0.48,  # mocap needs 0.40
}


def unitree_g1_dualarm_env_cfg(play: bool = False) -> G1DualarmManagerBasedRlEnvCfg:
  cfg = make_g1_dualarm_env_cfg()
  # The released policy (v16 R3, model_4999) was trained at 8192 envs; 4096 is
  # the default here so the task fits smaller GPUs. Batch size changes the
  # gradient, so a 4096-env run tracks R3's config but not its checkpoint —
  # pass --env.scene.num_envs 8192 to reproduce the released run's setup.
  cfg.scene.num_envs = 1 if play else 4096
  cfg.sim.njmax = 1024
  cfg.sim.mujoco.ccd_iterations = 50
  cfg.sim.contact_sensor_maxmatch = 64
  cfg.sim.nconmax = 256

  cfg.scene.entities = {
    "robot": get_g1_29dof_robot_cfg(),
    "toaster": get_toaster_cfg(),
  }

  #########################################################
  ##### terrain #####
  #########################################################
  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None

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
      pattern=r"^(left_shoulder_yaw|left_elbow_yaw|left_wrist|left_hand|right_shoulder_yaw|right_elbow_yaw|right_wrist|right_hand)_collision$",
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
      # Only legs and feet are illegal toaster contacts here. Torso/body
      # contact is intentionally not included while training the lift.
      # Previous broader pattern included torso/body contact:
      # pattern=r"^(torso|left|right)_(hip|thigh|shin|linkage_brace|foot[1-7])_collision$|^torso_collision$",
      pattern=r"^(left|right)_(hip_collision|thigh|thigh_collision|shin|shin_collision|foot[1-7]_collision)$",
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
    toaster_contact_cfg,
    left_hand_toaster_cfg,
    right_hand_toaster_cfg,
    illegal_toaster_contact_cfg,
    illegal_ground_contact_cfg,
  )

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = G1_29Dof_DUALARM_ACTION_SCALE

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
    # No training => no curriculum: disable the virtual PD outright and drop
    # the adaptive curricula (their recovery branch would otherwise re-raise
    # assistance during long play sessions). The env starts at final
    # difficulty via eval_mode (zero assistance, ground spawn).
    cfg.virtual_pd_cfg.enabled = False
    cfg.curriculum = {}

  return cfg
