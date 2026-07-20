from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.sensor import ContactSensor
from mjlab.utils.lab_api.math import (
  axis_angle_from_quat,
  quat_apply_inverse,
  quat_conjugate,
  quat_mul,
)
if TYPE_CHECKING:
  from mjlab_g1.envs.g1_dualarm_rl_env import G1DualarmManagerBasedRlEnv


def _near_object(
  env: G1DualarmManagerBasedRlEnv, radius_m: float
) -> torch.Tensor:
  """Whether the pelvis is within ``radius_m`` of the object in the XY plane.

  Posture terms that act on the reach share this gate: near the object there
  is no gait to protect, so a descent posture can be priced regardless of
  support state, while the approach walk further out stays exempt.
  """
  return (
    torch.linalg.vector_norm(
      env.robot.data.root_link_pos_w[:, :2]
      - env.toaster.data.root_link_pos_w[:, :2],
      dim=-1,
    )
    < radius_m
  )


def _knee_joint_ids(env: G1DualarmManagerBasedRlEnv) -> torch.Tensor:
  """Cached left/right knee joint indices."""
  if not hasattr(env, "_squat_knee_joint_ids"):
    ids, _ = env.robot.find_joints(
      ("left_knee_joint", "right_knee_joint"), preserve_order=True
    )
    env._squat_knee_joint_ids = torch.as_tensor(
      ids, device=env.device, dtype=torch.long
    )
  return env._squat_knee_joint_ids


def _episode_peak(
  env: G1DualarmManagerBasedRlEnv,
  name: str,
  value: torch.Tensor,
  tracking: torch.Tensor,
) -> torch.Tensor:
  """Accumulate ``value``'s episode maximum over ``tracking`` steps, and
  release it as a one-shot charge on the step the lift starts.

  Per-timestep posture penalties are speed-discounted: crossing a bad
  posture quickly costs almost nothing, so the policy learns to hurry rather
  than to avoid it. Charging the episode peak exactly once makes a brief
  excursion and a sustained one cost the same.

  The result is divided by ``step_dt`` so the reward manager's dt scaling
  cancels and the configured weight reads as cost per unit of ``value``.
  """
  peak_attr, prev_attr = f"_peak_{name}", f"_peak_{name}_prev_started"
  if not hasattr(env, peak_attr):
    setattr(env, peak_attr, torch.zeros(env.num_envs, device=env.device))
    setattr(
      env,
      prev_attr,
      torch.zeros(env.num_envs, device=env.device, dtype=torch.bool),
    )
  peak, prev_started = getattr(env, peak_attr), getattr(env, prev_attr)

  # Rewards are computed before resets, so a fresh episode arrives here with
  # episode_length_buf == 1.
  new_episode = env.episode_length_buf <= 1
  peak[new_episode] = 0.0
  prev_started[new_episode] = False

  started, _ = env._lift_progress()
  peak = torch.maximum(peak, value * tracking.float())
  fire = started & ~prev_started
  setattr(env, peak_attr, peak)
  setattr(env, prev_attr, started.clone())
  return peak * fire.float() / env.step_dt


def get_object_pose(env: G1DualarmManagerBasedRlEnv) -> torch.Tensor:
    obj_pos_w = env.toaster.data.root_link_pos_w[:, :3]
    obj_quat_w = env.toaster.data.root_link_quat_w

    return torch.cat([obj_pos_w, obj_quat_w], dim=-1)
#### dual-arm manipulation rewards ####

def hand_to_toaster(
  env: G1DualarmManagerBasedRlEnv, d_scale: float = 1.5
) -> torch.Tensor:
  dis = env._get_hand_toaster_dis()
  dist = torch.norm(dis, dim=-1)  # [num_envs, 2]

  reward_per_hand = torch.exp(-dist / d_scale)
  return reward_per_hand.mean(dim=-1)


def hands_near_markers(
  env: G1DualarmManagerBasedRlEnv, d_scale: float = 0.15
) -> torch.Tensor:
  """Sharp, ungated bonus for getting BOTH palms onto the grasp markers.

  Unlike ``hand_to_toaster`` (mean over hands, broad guidance), this is the
  PRODUCT of the two per-hand proximities, so it only pays off when both hands
  are simultaneously near their marker -- the precondition for the bilateral
  contact that gates the main lift rewards. This breaks the exploration deadlock
  where contact is never discovered.
  """
  dis = env._get_hand_toaster_dis()
  dist = torch.norm(dis, dim=-1)  # [num_envs, 2]
  prox = torch.exp(-dist / d_scale)  # [num_envs, 2]
  return prox[:, 0] * prox[:, 1]


def grasp_approach(
  env: G1DualarmManagerBasedRlEnv, d_scale: float = 0.06
) -> torch.Tensor:
  """Very sharp bilateral proximity that pulls both palms the final centimeters
  ONTO the markers.

  ``hands_near_markers`` (d_scale=0.15) saturates once the hands are roughly
  near, leaving little gradient for the last few cm into contact. This term uses
  a much smaller length scale so it only lights up right at the marker surface,
  supplying the missing pull across the near->contact gap. Ungated.
  """
  dis = env._get_hand_toaster_dis()
  dist = torch.norm(dis, dim=-1)  # [num_envs, 2]
  prox = torch.exp(-dist / d_scale)  # [num_envs, 2]
  return prox[:, 0] * prox[:, 1]


def upright(
  env: G1DualarmManagerBasedRlEnv,
  gate_on_lift: bool = False,
) -> torch.Tensor:
  """Reward for keeping the torso vertical (projected gravity z ~= -1 upright).

  With ``gate_on_lift=True`` the reward only applies once the lift has
  started: paying for torso verticality DURING the reach made bending at the
  waist costly, so the policy learned to lower itself by splaying the legs
  (splits/lunge) while keeping the torso vertical. Gating frees the reach to
  use a human-like hip hinge; the carry itself is still rewarded upright.
  """
  proj_grav_z = env.robot.data.projected_gravity_b[:, 2]
  reward = torch.clamp(-proj_grav_z, 0.0, 1.0)
  if gate_on_lift:
    started, _ = env._lift_progress()
    reward = reward * started.float()
  return reward


def torso_upright(
  env: G1DualarmManagerBasedRlEnv,
  gate_on_lift: bool = False,
) -> torch.Tensor:
  """Reward keeping the TORSO LINK vertical, not just the pelvis.

  ``upright`` reads the root (pelvis) frame, and v4 exploited that blind
  spot: pelvis level at exactly the ``hold_posture`` target height while the
  whole upper body folded forward at the waist joints (hunched carry). This
  term projects gravity into torso_link, which sits above the waist, so the
  fold itself loses reward. Same lift gating rationale as ``upright``: the
  reach phase legitimately pitches the torso (the mocap reach hits ~50deg).
  """
  torso_ids = torch.as_tensor(
    env.torso_body_id, device=env.device, dtype=torch.long
  )
  torso_quat_w = env.robot.data.body_link_quat_w[:, torso_ids[0]]
  gravity_w = torch.zeros_like(env.robot.data.root_link_pos_w)
  gravity_w[:, 2] = -1.0
  proj_grav_z = quat_apply_inverse(torso_quat_w, gravity_w)[:, 2]
  reward = torch.clamp(-proj_grav_z, 0.0, 1.0)
  if gate_on_lift:
    started, _ = env._lift_progress()
    reward = reward * started.float()
  return reward


def waist_deviation_penalty(
  env: G1DualarmManagerBasedRlEnv,
  deadband_rad: float = 0.3,
  gate_on_lift: bool = True,
) -> torch.Tensor:
  """L1 waist deviation from default beyond a deadband, after lift start.

  Joint-space backstop for ``torso_upright``: there is no kinematic trick
  that hunches the torso without bending these joints. The mocap hold keeps
  total waist deviation within ~0.2 rad of default (a slight natural forward
  lean), so the deadband covers the prior's lean and only the fold is taxed.
  Ungated during the reach, where the prior itself uses ~0.5 rad of pitch.
  """
  waist_ids = torch.as_tensor(
    env.waist_joint_ids, device=env.device, dtype=torch.long
  )
  deviation = torch.sum(
    torch.abs(
      env.robot.data.joint_pos[:, waist_ids]
      - env.robot.data.default_joint_pos[:, waist_ids]
    ),
    dim=-1,
  )
  penalty = torch.clamp(deviation - deadband_rad, min=0.0)
  if gate_on_lift:
    started, _ = env._lift_progress()
    penalty = penalty * started.float()
  return penalty


def leg_symmetry_penalty(
  env: G1DualarmManagerBasedRlEnv,
  deadband_rad: float = 0.25,
  near_radius_m: float = 0.7,
  left_sensor_name: str = "left_feet_ground_contact",
  right_sensor_name: str = "right_feet_ground_contact",
) -> torch.Tensor:
  """L1 left/right mismatch of hip-pitch and knee angles beyond a deadband,
  applied in double support OR within ``near_radius_m`` of the object.

  The mocap descends with a symmetric squat; v5 descended with a one-leg-back
  lunge and held with a fore-aft stagger — maximally asymmetric in exactly
  these joints (sides share sign conventions: the mocap squat bottoms out at
  |L-R| < 0.12 rad, so no mirror flip is needed). A pure double-support gate
  exempted stepping (mid-swing mismatch ~1.4 rad), but v6 exploited it by
  descending single-support (kneel, mismatch ~3.5 rad, one foot unweighted).
  The proximity term closes that hole: near the object there is no gait to
  protect, so asymmetry is taxed regardless of support; far from it the
  approach walk stays exempt. Targets asymmetry, not depth — a deep squat is
  free at any distance, so the v2/v3 failure cannot recur.
  """
  left_sensor: ContactSensor = env.scene[left_sensor_name]
  right_sensor: ContactSensor = env.scene[right_sensor_name]
  assert left_sensor.data.found is not None
  assert right_sensor.data.found is not None
  double_support = torch.any(left_sensor.data.found > 0, dim=-1) & torch.any(
    right_sensor.data.found > 0, dim=-1
  )
  gate = double_support | _near_object(env, near_radius_m)

  left_ids = torch.as_tensor(
    env.left_leg_sym_joint_ids, device=env.device, dtype=torch.long
  )
  right_ids = torch.as_tensor(
    env.right_leg_sym_joint_ids, device=env.device, dtype=torch.long
  )
  mismatch = torch.sum(
    torch.abs(
      env.robot.data.joint_pos[:, left_ids]
      - env.robot.data.joint_pos[:, right_ids]
    ),
    dim=-1,
  )
  return torch.clamp(mismatch - deadband_rad, min=0.0) * gate.float()


def prelift_peak_asymmetry_penalty(
  env: G1DualarmManagerBasedRlEnv,
  deadband_rad: float = 0.25,
  near_radius_m: float = 0.7,
) -> torch.Tensor:
  """One-shot charge at lift start for the PEAK leg asymmetry of the reach.

  ``leg_symmetry`` is per-timestep, so v7/v8 learned to dive through the
  lunge in ~0.5 s: a 3 rad mismatch integrated to ~-0.2/episode. Charging
  the episode maximum once, when the lift triggers, removes the speed
  discount — the dive costs the same at any velocity. Peak tracking uses
  the proximity gate only (no double-support term): far from the object the
  approach gait may swing freely; near it there is no gait to protect.
  """
  started, _ = env._lift_progress()
  left_ids = torch.as_tensor(
    env.left_leg_sym_joint_ids, device=env.device, dtype=torch.long
  )
  right_ids = torch.as_tensor(
    env.right_leg_sym_joint_ids, device=env.device, dtype=torch.long
  )
  mismatch = torch.sum(
    torch.abs(
      env.robot.data.joint_pos[:, left_ids]
      - env.robot.data.joint_pos[:, right_ids]
    ),
    dim=-1,
  )
  excess = torch.clamp(mismatch - deadband_rad, min=0.0)
  return _episode_peak(
    env,
    "prelift_asymmetry",
    excess,
    (~started) & _near_object(env, near_radius_m),
  )


def descent_speed_penalty(
  env: G1DualarmManagerBasedRlEnv,
  max_descent_speed: float = 0.5,
  near_radius_m: float = 0.7,
) -> torch.Tensor:
  """Squared excess of pelvis DOWNWARD speed near the object, pre-lift.

  The v7/v8 reach is a ballistic snatch: the pelvis drops >1.5 m/s so every
  per-timestep posture penalty only bites for a handful of frames. Capping
  descent speed forces the drop to stretch over many timesteps, which gives
  ``leg_symmetry`` (and the AMP discriminator) enough samples to shape it.
  The mocap squat descends well under ``max_descent_speed``, so the target
  motion is free; only the dive pays.
  """
  started, _ = env._lift_progress()
  descent = torch.clamp(-env.robot.data.root_link_lin_vel_w[:, 2], min=0.0)
  excess = torch.clamp(descent - max_descent_speed, min=0.0)
  gate = (~started) & _near_object(env, near_radius_m)
  return torch.square(excess) * gate.float()


def low_straight_knee_penalty(
  env: G1DualarmManagerBasedRlEnv,
  knee_min_rad: float = 0.8,
  pelvis_below_m: float = 0.62,
  near_radius_m: float = 0.7,
) -> torch.Tensor:
  """One-shot charge at lift start for the episode-peak of straight-kneed
  lowness during the reach.

  Measured on the v10 policy (model_34550): the pike drops the pelvis to
  ~0.40 m (p10) with the knees at -0.09 rad — dead straight — while the
  mocap at the same depth has knees at 1.9-2.2 rad. Knee flexion separates
  the strategies by 2 full radians, torso tilt by only ~20 deg, so this is
  the axis to price. The v12 per-step version of this term was answered
  with time compression (the dip below ``pelvis_below_m`` shrank to ~4
  frames, tilt got MORE extreme); charging the episode maximum once at
  lift start makes a 4-frame dip and a 4-second dip cost the same — the
  pattern that already beat the asymmetric dive in v9. A squat of any
  depth is free at any speed.
  """
  started, _ = env._lift_progress()
  low = env.robot.data.root_link_pos_w[:, 2] < pelvis_below_m
  knees = env.robot.data.joint_pos[:, _knee_joint_ids(env)].mean(dim=-1)
  straightness = torch.clamp(knee_min_rad - knees, min=0.0)
  # v13 dodged the height gate: 85% of episodes grab with the pelvis held
  # just ABOVE pelvis_below_m (folding harder to reach) and only drop after
  # the lift latches. The contact-settle window is the un-dodgeable gate:
  # every lift must hold both marker contacts for lift_trigger_contact_steps
  # before `started` latches, and in the mocap the knees are bent 1.9+ rad
  # whenever the hands are on the box — at any pelvis height. Straight knees
  # during settle IS the pike grab, wherever the pelvis sits.
  settle = getattr(env, "contact_settle_buf", None)
  grabbing = (
    (settle > 0) if settle is not None else torch.zeros_like(low)
  )
  low_and_near = low & _near_object(env, near_radius_m)
  return _episode_peak(
    env,
    "low_straight_knee",
    straightness,
    (low_and_near | grabbing) & (~started),
  )


def prelift_torso_pitch_penalty(
  env: G1DualarmManagerBasedRlEnv,
  # 0.9 (was 1.0): measured pike tilt is mean 0.90 / p90 1.23 rad, so the
  # 1.0 deadband sat above the mean and collected ~nothing (-0.02/ep).
  deadband_rad: float = 0.9,
) -> torch.Tensor:
  """Torso tilt beyond a deadband BEFORE the lift starts.

  v10's ``squat_shaping`` carrot never fired: the policy stayed in the v9
  pike (torso ~horizontal, knees straight) and the exp tail gives no usable
  gradient from there. This prices the pike on every current episode, which
  the existing terms cannot: ``torso_upright``/``waist_deviation`` are
  lift-gated, and ``descent_speed`` watches the pelvis, which a pike never
  drops. The deadband clears the mocap reach hinge (~0.9 rad max tilt)
  while the pike (~1.5+ rad) pays. Speed cannot dodge it: the lift trigger
  requires contacts held for ``lift_trigger_contact_steps`` while grasping
  at box height, so a straight-legged policy sustains the fold for that
  whole window no matter how fast it descends.
  """
  torso_ids = torch.as_tensor(
    env.torso_body_id, device=env.device, dtype=torch.long
  )
  torso_quat_w = env.robot.data.body_link_quat_w[:, torso_ids[0]]
  gravity_w = torch.zeros_like(env.robot.data.root_link_pos_w)
  gravity_w[:, 2] = -1.0
  proj_grav_z = quat_apply_inverse(torso_quat_w, gravity_w)[:, 2]
  tilt = torch.acos(torch.clamp(-proj_grav_z, -1.0, 1.0))
  started, _ = env._lift_progress()
  return torch.clamp(tilt - deadband_rad, min=0.0) * (~started).float()


def squat_shaping(
  env: G1DualarmManagerBasedRlEnv,
  near_radius_m: float = 0.7,
  knee_target_rad: float = 2.0,
  # 1.0 (was 0.6): with the policy starting knees-straight, the narrower
  # scale left ~exp(-2.8) of gradient at the pike — too flat to climb.
  knee_scale: float = 1.0,
  pelvis_high_m: float = 0.65,
  pelvis_low_m: float = 0.45,
) -> torch.Tensor:
  """Positive shaping for a mocap-style squat during the reach.

  v9 satisfied every posture PENALTY with a symmetric pike: legs straight
  and parallel, waist folded ~2 rad, pelvis never dropping. Nothing pays
  for the posture the prior actually uses, and penalties only teach what
  to avoid. This term pays for the squat's signature that a pike cannot
  fake: deep knee flexion WITH a low pelvis (mocap bottom: pelvis 0.39-0.46,
  knees 1.9-2.2 rad). The lowness ramp (zero above ``pelvis_high_m``, full
  below ``pelvis_low_m``) keeps upright walking/holding unaffected, and a
  bent-knee kneel cannot farm it: knee-ground contact terminates. Gated on
  near-object pre-lift, and bounded, so lifting still dominates hovering.
  """
  started, _ = env._lift_progress()
  knees = env.robot.data.joint_pos[:, _knee_joint_ids(env)].mean(dim=-1)
  knee_reward = torch.exp(
    -torch.abs(knees - knee_target_rad) / max(knee_scale, 1.0e-6)
  )
  pelvis_z = env.robot.data.root_link_pos_w[:, 2]
  lowness = torch.clamp(
    (pelvis_high_m - pelvis_z) / max(pelvis_high_m - pelvis_low_m, 1.0e-6),
    0.0,
    1.0,
  )
  gate = (~started) & _near_object(env, near_radius_m)
  return knee_reward * lowness * gate.float()


def hold_foreaft_stagger_penalty(
  env: G1DualarmManagerBasedRlEnv,
  deadband_m: float = 0.2,
) -> torch.Tensor:
  """Fore-aft foot offset (pelvis frame) beyond a deadband, during the hold.

  ``stance_width`` prices total separation and ``leg_symmetry`` is L1 on
  hip-pitch/knee, which a mirrored offset can dodge — v7/v8 both hold with
  one foot planted well ahead of the other. Measured along the pelvis x-axis
  so lateral stance width stays free; gated on the hold so gait and the
  (already-priced) reach are untouched.
  """
  holding = (env.success_hold_buf > 0).float()
  feet_ids = torch.as_tensor(
    env.feet_body_ids, device=env.device, dtype=torch.long
  )
  root_pos = env.robot.data.root_link_pos_w
  root_quat = env.robot.data.root_link_quat_w
  feet_pos = env.robot.data.body_link_pos_w[:, feet_ids]
  rel_left = quat_apply_inverse(root_quat, feet_pos[:, 0] - root_pos)
  rel_right = quat_apply_inverse(root_quat, feet_pos[:, 1] - root_pos)
  stagger = torch.abs(rel_left[:, 0] - rel_right[:, 0])
  return torch.clamp(stagger - deadband_m, min=0.0) * holding


def object_centered(
  env: G1DualarmManagerBasedRlEnv,
  target_forward: float = 0.35,
  xy_scale: float = 0.2,
) -> torch.Tensor:
  """Reward holding the object centered ahead of the base (mocap carry pose).

  No other term constrains WHERE around the body the object is held — v4
  carried it beside the right hip with the torso twisted toward it. Gated on
  the hold like ``hold_posture``, this pays for the object sitting straight
  ahead of the pelvis at the prior's hands-in-front offset.
  """
  holding = (env.success_hold_buf > 0).float()
  rel_w = (
    env.toaster.data.root_link_pos_w - env.robot.data.root_link_pos_w
  )
  rel_b = quat_apply_inverse(env.robot.data.root_link_quat_w, rel_w)
  target = torch.tensor(
    [target_forward, 0.0], device=env.device, dtype=rel_b.dtype
  )
  err = torch.linalg.vector_norm(rel_b[:, :2] - target, dim=-1)
  return holding * torch.exp(-err / max(xy_scale, 1.0e-6))


def stance_width_penalty(
  env: G1DualarmManagerBasedRlEnv,
  max_separation: float = 0.65,
) -> torch.Tensor:
  """Penalty on foot separation beyond a normal stance, active at all times.

  The splits/lunge descent puts the feet 1 m+ apart; a soft always-on cost
  makes narrow-stance strategies (squat, hip hinge) preferable everywhere,
  not just during the hold.
  """
  feet_ids = torch.as_tensor(
    env.feet_body_ids, device=env.device, dtype=torch.long
  )
  feet_pos_xy = env.robot.data.body_link_pos_w[:, feet_ids, :2]
  separation = torch.linalg.vector_norm(
    feet_pos_xy[:, 0] - feet_pos_xy[:, 1], dim=-1
  )
  return torch.clamp(separation - max_separation, min=0.0)


def dist_to_toaster(
  env: G1DualarmManagerBasedRlEnv, d_scale: float = 1.5
) -> torch.Tensor:
  root_pos = env.robot.data.root_link_pos_w[:, :3]
  toaster_pos = env.toaster.data.root_link_pos_w[:, :3]

  dist = torch.norm(root_pos - toaster_pos, dim=-1)
  return torch.exp(-dist / d_scale)


def hands_contact(
  env: G1DualarmManagerBasedRlEnv,
  sensor_name: str,
  min_reward_time_s: float = 2.0,
) -> torch.Tensor:
  contact_sensor: ContactSensor = env.scene[sensor_name]

  assert contact_sensor.data.found is not None

  contact = torch.any(contact_sensor.data.found > 0, dim=-1)
  elapsed_s = env.episode_length_buf.float() * env.step_dt
  reward_enabled = elapsed_s >= min_reward_time_s
  return contact.float() * reward_enabled.float()

def hands_at_markers(
  env: G1DualarmManagerBasedRlEnv,
  left_sensor: str,
  right_sensor: str,
  min_reward_time_s: float = 2.0,
) -> torch.Tensor:
  """Return whether both hand-marker contact sensors are active."""
  sensor1: ContactSensor = env.scene[left_sensor]
  sensor2: ContactSensor = env.scene[right_sensor]
  assert sensor1.data.found is not None
  assert sensor2.data.found is not None

  sensor1_contact = torch.any(sensor1.data.found > 0, dim=-1)
  sensor2_contact = torch.any(sensor2.data.found > 0, dim=-1)
  elapsed_s = env.episode_length_buf.float() * env.step_dt
  reward_enabled = elapsed_s >= min_reward_time_s
  return (sensor1_contact & sensor2_contact).float() * reward_enabled.float()


def marker_force(
  env: G1DualarmManagerBasedRlEnv,
  left_sensor: str,
  right_sensor: str,
  min_reward_time_s: float = 2.0,
  target_force: float = 10.0,
) -> torch.Tensor:
  """Reward applying force at both grasp markers while both contacts are active."""
  left_contact_sensor: ContactSensor = env.scene[left_sensor]
  right_contact_sensor: ContactSensor = env.scene[right_sensor]
  assert left_contact_sensor.data.found is not None
  assert right_contact_sensor.data.found is not None
  assert left_contact_sensor.data.force is not None
  assert right_contact_sensor.data.force is not None

  left_contact = torch.any(left_contact_sensor.data.found > 0, dim=-1)
  right_contact = torch.any(right_contact_sensor.data.found > 0, dim=-1)
  both_contact = left_contact & right_contact

  left_force = torch.norm(left_contact_sensor.data.force, dim=-1)
  right_force = torch.norm(right_contact_sensor.data.force, dim=-1)
  left_force = left_force.reshape(env.num_envs, -1).amax(dim=1)
  right_force = right_force.reshape(env.num_envs, -1).amax(dim=1)
  avg_force = 0.5 * (left_force + right_force)
  force_reward = torch.clamp(avg_force / max(target_force, 1.0e-6), 0.0, 1.0)

  elapsed_s = env.episode_length_buf.float() * env.step_dt
  reward_enabled = elapsed_s >= min_reward_time_s
  return force_reward * both_contact.float() * reward_enabled.float()


def object_trajectory_tracking(
  env: G1DualarmManagerBasedRlEnv,
  left_sensor: str,
  right_sensor: str,
  position_tolerance: float = 0.075,
) -> torch.Tensor:
  """
  Reward toaster position tracking along the analytic trajectory.

  The reward is gated by both grasp-marker contacts so the virtual object
  controller cannot create reward by moving the toaster without a real grasp.
  """
  left_contact_sensor: ContactSensor = env.scene[left_sensor]
  right_contact_sensor: ContactSensor = env.scene[right_sensor]

  assert left_contact_sensor.data.found is not None
  assert right_contact_sensor.data.found is not None

  left_contact = torch.any(left_contact_sensor.data.found > 0, dim=-1)
  right_contact = torch.any(right_contact_sensor.data.found > 0, dim=-1)
  both_contact = left_contact & right_contact

  reference_pos_w, _ = env.get_object_trajectory_reference()
  object_pos_w = env.toaster.data.root_link_pos_w[:, :3]

  position_error = torch.linalg.vector_norm(
    object_pos_w - reference_pos_w,
    dim=-1,
  )

  tracking_reward = torch.exp(
    -position_error / max(position_tolerance, 1.0e-6)
  )

  return tracking_reward * both_contact.float()


def object_orientation_tracking(
  env: G1DualarmManagerBasedRlEnv,
  left_sensor: str,
  right_sensor: str,
  tolerance_rad: float = 0.35,
) -> torch.Tensor:
  """Reward keeping the object LEVEL (at its spawn orientation) while held.

  Position tracking alone let the policy carry the object tilted 30-40deg —
  during bootstrap the virtual PD's orientation terms kept it level for free,
  so the policy never learned orientation control. Reference is the per-env
  spawn orientation (includes the randomized yaw), same gating as position
  tracking so ungrasped motion cannot earn it.
  """
  left_contact_sensor: ContactSensor = env.scene[left_sensor]
  right_contact_sensor: ContactSensor = env.scene[right_sensor]
  assert left_contact_sensor.data.found is not None
  assert right_contact_sensor.data.found is not None
  both_contact = (
    torch.any(left_contact_sensor.data.found > 0, dim=-1)
    & torch.any(right_contact_sensor.data.found > 0, dim=-1)
  )

  q_obj = env.toaster.data.root_link_quat_w
  q_ref = env.virtual_pd_controller.reference_quat_w
  q_err = quat_mul(q_ref, quat_conjugate(q_obj))
  angle = torch.linalg.vector_norm(axis_angle_from_quat(q_err), dim=-1)

  return torch.exp(-angle / max(tolerance_rad, 1.0e-6)) * both_contact.float()


def hold_posture(
  env: G1DualarmManagerBasedRlEnv,
  target_base_height: float = 0.75,
  height_scale: float = 0.1,
  feet_max_separation: float = 0.55,
  feet_scale: float = 0.3,
) -> torch.Tensor:
  """Reward a NATURAL carry posture while actually holding at goal.

  Success only checks object height + contacts, so v2 converged to full
  splits and v3 to a wide fencing lunge — stable but undeployable. This term
  pays, only during the hold (success_hold_buf > 0), for keeping the pelvis
  near standing height and the feet within a normal stance width.
  """
  holding = (env.success_hold_buf > 0).float()

  base_height = env.robot.data.root_link_pos_w[:, 2]
  height_reward = torch.exp(
    -torch.abs(base_height - target_base_height) / max(height_scale, 1.0e-6)
  )

  feet_ids = torch.as_tensor(
    env.feet_body_ids, device=env.device, dtype=torch.long
  )
  feet_pos_xy = env.robot.data.body_link_pos_w[:, feet_ids, :2]
  separation = torch.linalg.vector_norm(
    feet_pos_xy[:, 0] - feet_pos_xy[:, 1], dim=-1
  )
  excess = torch.clamp(separation - feet_max_separation, min=0.0)
  feet_reward = torch.exp(-excess / max(feet_scale, 1.0e-6))

  return holding * height_reward * feet_reward


def hold_at_goal(
  env: G1DualarmManagerBasedRlEnv,
) -> torch.Tensor:
  """Reward for SUSTAINING the hold: object at goal height with both marker
  contacts, ramping 0 -> 1 over ``hold_steps`` consecutive steps.

  Success requires a sustained hold, but no other term pays for holding
  specifically (tracking pays the same during transit). This puts gradient
  exactly on the success condition.
  """
  return torch.clamp(
    env.success_hold_buf.float() / max(env.cfg.hold_steps, 1),
    0.0,
    1.0,
  )


def virtual_assistance_force(
  env: G1DualarmManagerBasedRlEnv,
) -> torch.Tensor:
  """Fraction of the virtual-PD force cap used during the last control step.

  Combined with a negative weight (ramped in by
  ``assist_force_penalty_curriculum``), this pays the policy for making the
  virtual object controller unnecessary: the assistance force shrinks exactly
  when the robot itself keeps the object on the reference trajectory.
  """
  return env.assist_force_frac


def missing_grasp_during_lift(
  env: G1DualarmManagerBasedRlEnv,
  left_sensor: str,
  right_sensor: str,
) -> torch.Tensor:
  """Penalty indicator when the trajectory is moving but both marker contacts are absent."""
  left_contact_sensor: ContactSensor = env.scene[left_sensor]
  right_contact_sensor: ContactSensor = env.scene[right_sensor]

  assert left_contact_sensor.data.found is not None
  assert right_contact_sensor.data.found is not None

  left_contact = torch.any(left_contact_sensor.data.found > 0, dim=-1)
  right_contact = torch.any(right_contact_sensor.data.found > 0, dim=-1)
  both_contact = left_contact & right_contact

  # Contact-triggered lift: the penalty applies only while the lift the
  # robot itself initiated is in motion (i.e. it grasped, then let go).
  lift_moving = env.lift_phase_active()

  return ((~both_contact) & lift_moving).float()

#### Stability  rewards ####


def yaw_rate_penalty(
  env: G1DualarmManagerBasedRlEnv,
  threshold: float = 0.075,
) -> torch.Tensor:
  """Penalty for body-frame yaw rate outside a deadzone around zero."""
  yaw_rate = env.robot.data.root_link_ang_vel_b[:, 2]
  excess = torch.clamp(torch.abs(yaw_rate) - threshold, min=0.0)
  return torch.square(excess)


def angular_vel_penalty(
  env: G1DualarmManagerBasedRlEnv,
  threshold: float = 0.05,
) -> torch.Tensor:
  """Penalty for base angular motion outside a small deadzone."""
  ang_vel = env.robot.data.root_link_ang_vel_b
  speed = torch.norm(ang_vel, dim=-1)
  excess = torch.clamp(speed - threshold, min=0.0)
  return torch.square(excess)


def linear_vel_penalty(
  env: G1DualarmManagerBasedRlEnv,
  threshold: float = 0.075,
) -> torch.Tensor:
  """Penalty for base linear motion outside a small deadzone."""
  root_lin_vel = env.robot.data.root_link_lin_vel_b
  speed = torch.norm(root_lin_vel, dim=-1)
  excess = torch.clamp(speed - threshold, min=0.0)
  return torch.square(excess)

def feet_slip(
  env: G1DualarmManagerBasedRlEnv,
  left_sensor_name: str = "left_feet_ground_contact",
  right_sensor_name: str = "right_feet_ground_contact",
  threshold_min: float = 0.0,
) -> torch.Tensor:
  left_sensor: ContactSensor = env.scene[left_sensor_name]
  right_sensor: ContactSensor = env.scene[right_sensor_name]

  left_contact = left_sensor.data.found
  right_contact = right_sensor.data.found
  assert left_contact is not None
  assert right_contact is not None

  contact = torch.cat([left_contact, right_contact], dim=1)

  feet_body_ids = torch.as_tensor(
    env.feet_body_ids, device=env.device, dtype=torch.long
  )
  foot_vel_xy = env.robot.data.body_link_lin_vel_w[:, feet_body_ids, :2]
  foot_speed = torch.norm(foot_vel_xy, dim=-1)

  slip = torch.clamp(foot_speed - threshold_min, min=0.0)
  return torch.sum(slip * (contact > 0).float(), dim=1)

def feet_contact(
  env: G1DualarmManagerBasedRlEnv,
  left_sensor_name: str = "left_feet_ground_contact",
  right_sensor_name: str = "right_feet_ground_contact",
) -> torch.Tensor:
  left_sensor: ContactSensor = env.scene[left_sensor_name]
  right_sensor: ContactSensor = env.scene[right_sensor_name]

  left_contact = left_sensor.data.found
  right_contact = right_sensor.data.found
  assert left_contact is not None
  assert right_contact is not None

  left_in_contact = torch.any(left_contact > 0, dim=-1)
  right_in_contact = torch.any(right_contact > 0, dim=-1)

  return (left_in_contact & right_in_contact).float()

def self_collision_cost(
  env: G1DualarmManagerBasedRlEnv, sensor_name: str
) -> torch.Tensor:
  """Cost that returns whether self-collision was detected by a sensor."""
  sensor: ContactSensor = env.scene[sensor_name]
  assert sensor.data.found is not None
  return sensor.data.found.squeeze(-1).float()

def _illegal_contact_mask(
  env: G1DualarmManagerBasedRlEnv,
  sensor_names: tuple[str, ...],
) -> torch.Tensor:
  illegal = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
  for sensor_name in sensor_names:
    sensor: ContactSensor = env.scene[sensor_name]
    assert sensor.data.found is not None
    illegal |= torch.any(sensor.data.found > 0, dim=-1)
  return illegal

def illegal_contact_penalty(
  env: G1DualarmManagerBasedRlEnv,
  sensor_names: tuple[str, ...],
) -> torch.Tensor:
  return _illegal_contact_mask(env, sensor_names).float()
