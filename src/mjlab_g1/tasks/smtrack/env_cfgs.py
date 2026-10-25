"""SMTrack env configs: mjlab's G1 BeyondMimic tracking cfg with the brief §0 changes.

Shared by PPO and SuperMimic, so both arms see one env (protocol D3). Changes vs mjlab
`unitree_g1_flat_tracking_env_cfg()`:
- joint_limit weight -10 -> -2, no self_collisions reward and no self-collision sensor
  (paper Table A.2);
- no push_robot; base_com / encoder_bias / foot_friction keep mjlab's ranges but resample
  per episode (mode="reset", paper: xi is per episode);
- ee_body_pos termination removed; the clip end is a time-out (`clip_end`), and the
  command never wraps (SMMotionCommand);
- observation groups actor_hist (10-frame history) / actor_goal (K=10 future frames) /
  critic (brief §2);
- lift: box entity, box obs/rewards/termination/DR, box placed at box_ref[k_start].
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Literal

import numpy as np

from mjlab.envs.mdp import dr
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.envs.mdp.terminations import nan_detection
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.tasks.tracking.config.g1.env_cfgs import unitree_g1_flat_tracking_env_cfg
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab_g1.envs.smtrack_env import SMTrackEnvCfg
from mjlab_g1.tasks.smtrack import mdp

TaskName = Literal["walk", "squat", "lift"]

REPO_ROOT = Path(__file__).resolve().parents[4]
MOTION_FILES: dict[str, Path] = {
  "walk": REPO_ROOT / "dataset/tracking/locomotion1.npz",
  # The lift motion without the box: separates "can track the motion" from "can carry".
  "squat": REPO_ROOT / "dataset/tracking/dual_arm1.npz",
  "lift": REPO_ROOT / "dataset/tracking/dual_arm1_box.npz",
}

HISTORY_LEN = 10
# Box DR (brief §0): x nominal, per episode.
BOX_MASS_SCALE_RANGE = (0.8, 1.2)
BOX_FRICTION_SCALE_RANGE = (0.8, 1.2)
# Clip-end truncation must win over the step-count time-out: allow the clip plus 1 s.
EPISODE_MARGIN_S = 1.0


def make_env_cfg(
  task: TaskName, play: bool = False, motion_file: str | Path | None = None
) -> SMTrackEnvCfg:
  base = unitree_g1_flat_tracking_env_cfg()
  cfg = SMTrackEnvCfg(**{f.name: getattr(base, f.name) for f in dataclasses.fields(base)})
  has_box = task == "lift"
  motion_file = Path(motion_file) if motion_file is not None else MOTION_FILES[task]
  num_frames = int(np.load(motion_file)["joint_pos"].shape[0])

  _scene(cfg, has_box)
  _command(cfg, motion_file, has_box)
  _rewards(cfg, has_box)
  _terminations(cfg, has_box)
  _events(cfg, has_box)
  _observations(cfg, has_box)
  cfg.xi_lo, cfg.xi_hi = _xi_bounds(cfg, has_box)
  # The clip end (F-1 rewarded frames from k=0) always comes before this time-out.
  cfg.episode_length_s = num_frames * cfg.sim.mujoco.timestep * cfg.decimation + EPISODE_MARGIN_S
  if play:
    _apply_play(cfg, has_box)
  return cfg


def _scene(cfg: SMTrackEnvCfg, has_box: bool) -> None:
  cfg.scene.sensors = ()  # self-collision sensor only fed the removed reward
  if has_box:
    cfg.scene.entities["box"] = mdp.get_box_cfg()
    # mjlab tracking sizes (35 / 250) are for the robot alone; the box adds floor and
    # hand contacts (condim 3: 4 rows each).
    cfg.sim.nconmax = 128
    cfg.sim.njmax = 600


def _command(cfg: SMTrackEnvCfg, motion_file: Path, has_box: bool) -> None:
  old = cfg.commands["motion"]
  assert isinstance(old, MotionCommandCfg)
  fields = {f.name: getattr(old, f.name) for f in dataclasses.fields(old) if f.init}
  fields["motion_file"] = str(motion_file)
  cfg.commands["motion"] = mdp.SMMotionCommandCfg(
    **fields, box_entity_name="box" if has_box else None
  )


def _rewards(cfg: SMTrackEnvCfg, has_box: bool) -> None:
  del cfg.rewards["self_collisions"]
  cfg.rewards["joint_limit"].weight = -2.0
  if has_box:
    # Ours (the paper has no object terms); std 0.1 m like a hand-sized tolerance.
    cfg.rewards["box_pos"] = RewardTermCfg(
      func=mdp.box_position_error_exp, weight=1.0, params={"command_name": "motion", "std": 0.1}
    )
    cfg.rewards["box_ori"] = RewardTermCfg(
      func=mdp.box_orientation_error_exp, weight=0.5, params={"command_name": "motion", "std": 0.4}
    )


def _terminations(cfg: SMTrackEnvCfg, has_box: bool) -> None:
  del cfg.terminations["ee_body_pos"]
  cfg.terminations["clip_end"] = TerminationTermCfg(
    func=mdp.motion_clip_end, params={"command_name": "motion"}, time_out=True
  )
  # A physics blow-up otherwise runs NaN until the clip ends (the anchor checks compare
  # against thresholds, which NaN never trips). Flagged as a time-out: the state carries
  # no learnable failure signal, and the analytic reward has no NaN criterion to mirror.
  cfg.terminations["nan"] = TerminationTermCfg(func=nan_detection, time_out=True)
  if has_box:
    cfg.terminations["box_pos"] = TerminationTermCfg(
      func=mdp.bad_box_pos, params={"command_name": "motion", "threshold": 0.3}
    )


def _events(cfg: SMTrackEnvCfg, has_box: bool) -> None:
  del cfg.events["push_robot"]
  for name in ("base_com", "encoder_bias", "foot_friction"):
    cfg.events[name].mode = "reset"
  if has_box:
    cfg.events["box_mass"] = EventTermCfg(
      mode="reset",
      func=mdp.body_mass_scale,
      params={
        "asset_cfg": SceneEntityCfg("box", body_names=(mdp.BOX_BODY_NAME,)),
        "scale_range": BOX_MASS_SCALE_RANGE,
      },
    )
    cfg.events["box_friction"] = EventTermCfg(
      mode="reset",
      func=dr.geom_friction,
      params={
        "asset_cfg": SceneEntityCfg("box", geom_names=(mdp.BOX_GEOM_NAME,)),
        "operation": "scale",
        "ranges": BOX_FRICTION_SCALE_RANGE,
      },
    )


def _noise(half_width: float) -> Unoise:
  return Unoise(n_min=-half_width, n_max=half_width)


def _observations(cfg: SMTrackEnvCfg, has_box: bool) -> None:
  # Order and noise follow brief §2 / features.OBS_TERMS_*: the learner's history layout
  # is mjlab's per-term one ([term1 h0..h9 | term2 h0..h9 | ...]).
  hist = {
    "joint_pos": ObservationTermCfg(
      func=mdp.joint_pos_rel, params={"biased": True}, noise=_noise(0.01)
    ),
    "joint_vel": ObservationTermCfg(func=mdp.joint_vel_rel, noise=_noise(0.5)),
    "base_ang_vel": ObservationTermCfg(
      func=mdp.builtin_sensor, params={"sensor_name": "robot/imu_ang_vel"}, noise=_noise(0.2)
    ),
    "projected_gravity": ObservationTermCfg(func=mdp.projected_gravity, noise=_noise(0.05)),
    "actions": ObservationTermCfg(func=mdp.last_action),
  }
  if has_box:
    hist["box_pos_b"] = ObservationTermCfg(func=mdp.box_pos_b, noise=_noise(0.01))
    hist["box_rot6d_b"] = ObservationTermCfg(func=mdp.box_rot6d_b, noise=_noise(0.02))
  goal = {"goal": ObservationTermCfg(func=mdp.goal_frames)}

  critic = dict(cfg.observations["critic"].terms)  # mjlab critic terms, 286
  critic["goal"] = ObservationTermCfg(func=mdp.goal_frames)
  if has_box:
    critic["box_state"] = ObservationTermCfg(func=mdp.box_state_b)

  cfg.observations = {
    "actor_hist": ObservationGroupCfg(
      terms=hist,
      concatenate_terms=True,
      enable_corruption=True,
      history_length=HISTORY_LEN,
      flatten_history_dim=True,
    ),
    "actor_goal": ObservationGroupCfg(
      terms=goal, concatenate_terms=True, enable_corruption=False
    ),
    "critic": ObservationGroupCfg(
      terms=critic, concatenate_terms=True, enable_corruption=False
    ),
  }


def _xi_bounds(cfg: SMTrackEnvCfg, has_box: bool) -> tuple[tuple[float, ...], tuple[float, ...]]:
  """xi = [foot friction 1, encoder bias 29, torso CoM offset 3 (, box mass, box friction)]."""
  fr = cfg.events["foot_friction"].params["ranges"]
  bias = cfg.events["encoder_bias"].params["bias_range"]
  com = cfg.events["base_com"].params["ranges"]
  lo = [fr[0], *([bias[0]] * 29), *(com[i][0] for i in range(3))]
  hi = [fr[1], *([bias[1]] * 29), *(com[i][1] for i in range(3))]
  if has_box:
    lo += [BOX_MASS_SCALE_RANGE[0], BOX_FRICTION_SCALE_RANGE[0]]
    hi += [BOX_MASS_SCALE_RANGE[1], BOX_FRICTION_SCALE_RANGE[1]]
  return tuple(float(x) for x in lo), tuple(float(x) for x in hi)


def _apply_play(cfg: SMTrackEnvCfg, has_box: bool) -> None:
  """Nominal evaluation: no obs noise, xi nominal, no reset noise, start at frame 0."""
  cfg.observations["actor_hist"].enable_corruption = False
  # DR events stay (fields stay per-env, get_xi works) but pin the nominal values.
  cfg.events["foot_friction"].params["ranges"] = (1.0, 1.0)
  cfg.events["encoder_bias"].params["bias_range"] = (0.0, 0.0)
  cfg.events["base_com"].params["ranges"] = {i: (0.0, 0.0) for i in range(3)}
  if has_box:
    cfg.events["box_mass"].params["scale_range"] = (1.0, 1.0)
    cfg.events["box_friction"].params["ranges"] = (1.0, 1.0)
  motion = cfg.commands["motion"]
  assert isinstance(motion, MotionCommandCfg)
  motion.sampling_mode = "start"
  motion.pose_range = {}
  motion.velocity_range = {}
  motion.joint_position_range = (0.0, 0.0)
