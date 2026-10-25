"""Evaluate a PPO or SuperMimic checkpoint on an SMTrack task (brief §5) -> JSON.

  PYTHONPATH=src python src/mjlab_g1/scripts/eval_track.py --task G1-SMTrack-Walk \
    --learner ppo --checkpoint logs/smtrack/.../model_29999.pt [--episodes 64] [--output m.json]

Nominal condition (protocol §2.2): play cfg = every episode from frame 0, no DR (foot
friction 1.0, xi nominal), no observation noise, no reset noise, deterministic actions.
Episodes run in parallel, one per env, and end at the first termination or the clip end.
Nominal episodes are identical, so their success is one rollout, not a rate. --randomize keeps
the frame-0 start and deterministic actions but restores the training cfg's DR, observation
noise and reset noise, so each episode differs (robustness condition).

Metrics (per episode, then mean over episodes):
- success: reached the clip end without a termination; fall_rate: terminated.
- norm_return: undiscounted reward sum / ((F-1) x max per-step reward); steps after an
  episode ends add 0. norm_return_tracking uses only the 6 tracking kernels.
- body_pos_err_{mean,max}_m (global, 14 tracked bodies) and anchor_rel_body_pos_err_*
  (R_anchor^T (p_b - p_anchor) vs the same for the reference), joint_pos_err_rad (mean |dq|),
  peak_joint_speed (max |qd|, with the reference's peak for scale).
- lift: box_success (no termination and box z >= ref z - 0.1 at the clip end),
  box_pos_err_m, peak_box_vz, box_tilt_end_deg, box_rise_5_to_55cm_s.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
from pathlib import Path
from typing import Any

import torch

from mjlab.utils.lab_api.math import quat_apply_inverse

TRACKING_TERMS = (
  "motion_global_root_pos",
  "motion_global_root_ori",
  "motion_body_pos",
  "motion_body_ori",
  "motion_body_lin_vel",
  "motion_body_ang_vel",
)
BOX_SUCCESS_TOL = 0.1
RISE_LOW, RISE_HIGH = 0.05, 0.55


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
  from mjlab_g1.tasks.smtrack import list_tasks

  p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  p.add_argument("--task", choices=list_tasks(), required=True)
  p.add_argument("--learner", choices=("ppo", "supermimic"), required=True)
  p.add_argument("--checkpoint", required=True, type=Path)
  p.add_argument("--episodes", type=int, default=64)
  p.add_argument("--device", default=None)
  p.add_argument("--output", type=Path, default=None, help="JSON path (default: stdout only)")
  p.add_argument("--randomize", action="store_true",
                 help="training DR + obs noise + reset noise (still frame 0, deterministic actions)")
  p.add_argument("--agent-cfg", type=Path, default=None,
                 help="agent.json (default: <checkpoint dir>/params/agent.json)")
  p.add_argument("--sm-clean-obs", action="store_true",
                 help="with --randomize, give SuperMimic clean observations (the pre-2026-10-23 "
                      "behaviour; PPO always gets the env's obs corruption)")
  return p.parse_args(argv)


def obs_noise(args: argparse.Namespace) -> bool:
  """Whether the actor sees sensor noise. PPO reads the env's corrupted actor_hist (on under
  --randomize); SuperMimic's inference policy builds its own frames from the state, so it needs
  the same noise switched on explicitly (identical half-widths, features.OBS_NOISE)."""
  if args.learner == "supermimic" and args.sm_clean_obs:
    return False
  return bool(args.randomize)


def load_agent_cfg(args: argparse.Namespace) -> dict:
  path = args.agent_cfg or args.checkpoint.parent / "params" / "agent.json"
  if path.exists():
    with open(path) as f:
      return json.load(f)
  print(f"[eval_track] {path} not found; using the default {args.learner} cfg")
  if args.learner == "ppo":
    from mjlab_g1.tasks.smtrack import make_ppo_cfg

    return dataclasses.asdict(make_ppo_cfg(args.task))
  from mjlab_g1.rl.supermimic.config import SuperMimicRunnerCfg

  return dataclasses.asdict(SuperMimicRunnerCfg())


def make_policy(args: argparse.Namespace, vec_env: Any, device: str) -> Any:
  from mjlab_g1.scripts.train_track import make_runner

  agent = load_agent_cfg(args)
  if args.learner == "supermimic":
    # A one-clip deterministic rollout gains nothing from torch.compile, and compiling on
    # the CPU (inductor + C++ toolchain) would dominate the eval time.
    agent.setdefault("memory", {})["compile"] = False
  runner = make_runner(args.learner, vec_env, agent, None, device)
  runner.load(str(args.checkpoint), map_location=device)
  policy = runner.get_inference_policy(device)
  if args.learner == "supermimic":
    policy.obs_noise = obs_noise(args)
  return policy


def max_step_reward(env: Any, terms: tuple[str, ...] | None = None) -> float:
  """Sum of positive weights x dt: every exp kernel at 1, penalties at 0."""
  rm = env.reward_manager
  total = sum(
    rm.get_term_cfg(n).weight
    for n in rm.active_terms
    if rm.get_term_cfg(n).weight > 0 and (terms is None or n in terms)
  )
  return total * env.step_dt


class EpisodeStats:
  """Per-env accumulators over the valid (pre-done) steps of the single episode."""

  def __init__(self, n: int, device: str):
    z = lambda: torch.zeros(n, device=device)  # noqa: E731
    self.ret, self.ret_track, self.steps = z(), z(), z()
    self.body_err_sum, self.anchor_err_sum, self.joint_err_sum = z(), z(), z()
    self.body_err_max, self.anchor_err_max, self.qd_max = z(), z(), z()
    self.box_err_sum, self.box_vz_max = z(), z()
    self.t_low = torch.full((n,), math.nan, device=device)
    self.t_high = torch.full((n,), math.nan, device=device)

  @staticmethod
  def add(acc: torch.Tensor, x: torch.Tensor, m: torch.Tensor) -> None:
    acc += torch.where(m, x, torch.zeros_like(x))

  @staticmethod
  def peak(acc: torch.Tensor, x: torch.Tensor, m: torch.Tensor) -> None:
    acc.copy_(torch.where(m, torch.maximum(acc, x), acc))


def tracking_errors(env: Any, k: torch.Tensor) -> dict[str, torch.Tensor]:
  """Errors of the post-step robot vs reference frame k (the frame this step was scored on)."""
  cmd = env.motion_command
  robot = env.scene["robot"]
  m = cmd.motion
  ref_pos = m.body_pos_w[k] + env.scene.env_origins[:, None]
  ref_quat_a = m.body_quat_w[k, cmd.motion_anchor_body_index]
  pos = cmd.robot_body_pos_w
  a = cmd.motion_anchor_body_index
  glob = torch.norm(pos - ref_pos, dim=-1)  # [N,14]
  rel = quat_apply_inverse(cmd.robot_anchor_quat_w[:, None].expand_as(cmd.robot_body_quat_w),
                           pos - pos[:, a : a + 1])
  rel_ref = quat_apply_inverse(ref_quat_a[:, None].expand_as(cmd.robot_body_quat_w),
                               ref_pos - ref_pos[:, a : a + 1])
  anchor = torch.norm(rel - rel_ref, dim=-1)
  joint = (robot.data.joint_pos - m.joint_pos[k]).abs().mean(-1)
  return {
    "body_mean": glob.mean(-1), "body_max": glob.max(-1).values,
    "anchor_mean": anchor.mean(-1), "anchor_max": anchor.max(-1).values,
    "joint": joint, "qd_max": robot.data.joint_vel.abs().max(-1).values,
  }


@torch.no_grad()
def evaluate(args: argparse.Namespace) -> dict[str, Any]:
  from mjlab.rl import RslRlVecEnvWrapper
  from mjlab_g1.envs.smtrack_env import SMTrackEnv
  from mjlab_g1.tasks.smtrack import make_env_cfg

  device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
  cfg = make_env_cfg(args.task, play=not args.randomize)
  if args.randomize:
    cfg.commands["motion"].sampling_mode = "start"
  cfg.scene.num_envs = args.episodes
  cfg.seed = 0
  # Finished envs are frozen (stepped, ignored), never reset: a partial reset() would
  # advance every other env's reference frame (see smtrack_env.py).
  cfg.auto_reset = False
  env = SMTrackEnv(cfg, device=device)
  vec_env = RslRlVecEnvWrapper(env)
  policy = make_policy(args, vec_env, device)

  n, cmd = env.num_envs, env.motion_command
  num_steps = cmd.num_frames - 1  # rewarded frames 1..F-1 from a frame-0 start
  r_max, r_max_track = max_step_reward(env), max_step_reward(env, TRACKING_TERMS)
  track_ids = [env.reward_term_names.index(t) for t in TRACKING_TERMS]
  st = EpisodeStats(n, device)
  active = torch.ones(n, dtype=torch.bool, device=device)
  success = torch.zeros(n, dtype=torch.bool, device=device)
  fell = torch.zeros(n, dtype=torch.bool, device=device)
  box_ok = torch.zeros(n, dtype=torch.bool, device=device)
  box_tilt_end = torch.full((n,), math.nan, device=device)
  box_z0 = env.get_state()[:, 73].clone() if env.has_box else None
  dt = env.step_dt

  obs = vec_env.get_observations()
  for t in range(num_steps + 1):
    k = env.get_ref_index()
    obs, rew, _, extras = vec_env.step(policy(obs).to(device))
    term = env.reset_terminated.clone()
    trunc = extras["time_outs"].to(torch.bool)
    m = active
    st.add(st.ret, rew, m)
    st.add(st.ret_track, env.get_reward_terms()[:, track_ids].sum(-1), m)
    st.add(st.steps, torch.ones_like(rew), m)
    e = tracking_errors(env, k)
    st.add(st.body_err_sum, e["body_mean"], m)
    st.add(st.anchor_err_sum, e["anchor_mean"], m)
    st.add(st.joint_err_sum, e["joint"], m)
    st.peak(st.body_err_max, e["body_max"], m)
    st.peak(st.anchor_err_max, e["anchor_max"], m)
    st.peak(st.qd_max, e["qd_max"], m)
    if env.has_box:
      s = env.get_state()
      box_z = s[:, 73]
      st.add(st.box_err_sum, torch.norm(s[:, 71:74] - cmd.box_ref_pos[k], dim=-1), m)
      st.peak(st.box_vz_max, s[:, 80].abs(), m)
      rise = box_z - box_z0
      tt = (t + 1) * dt
      st.t_low = torch.where(m & st.t_low.isnan() & (rise >= RISE_LOW), tt, st.t_low)
      st.t_high = torch.where(m & st.t_high.isnan() & (rise >= RISE_HIGH), tt, st.t_high)

    done = active & (term | trunc)
    success |= done & trunc & ~term
    fell |= done & term
    if env.has_box and done.any():
      s = env.get_state()
      ok = s[:, 73] >= cmd.box_ref_pos[k, 2] - BOX_SUCCESS_TOL
      box_ok |= done & trunc & ~term & ok
      # Tilt = angle between the box z axis and world z.
      up = quat_apply_inverse(s[:, 74:78], torch.tensor([0.0, 0.0, 1.0], device=device).expand(n, 3))
      tilt = torch.rad2deg(torch.acos(up[:, 2].clamp(-1.0, 1.0)))
      box_tilt_end = torch.where(done, tilt, box_tilt_end)
    active = active & ~done
    env._manual_reset_pending.zero_()  # freeze finished envs instead of resetting them
    if not active.any():
      break

  steps = st.steps.clamp(min=1)
  per_ep = {
    "success": success.float(),
    "fall": fell.float(),
    "norm_return": st.ret / (num_steps * r_max),
    "norm_return_tracking": st.ret_track / (num_steps * r_max_track),
    "episode_steps": st.steps,
    "body_pos_err_mean_m": st.body_err_sum / steps,
    "body_pos_err_max_m": st.body_err_max,
    "anchor_rel_body_pos_err_mean_m": st.anchor_err_sum / steps,
    "anchor_rel_body_pos_err_max_m": st.anchor_err_max,
    "joint_pos_err_rad": st.joint_err_sum / steps,
    "peak_joint_speed": st.qd_max,
  }
  if env.has_box:
    per_ep |= {
      "box_success": box_ok.float(),
      "box_pos_err_m": st.box_err_sum / steps,
      "peak_box_vz": st.box_vz_max,
      "box_tilt_end_deg": box_tilt_end,
      "box_rise_5_to_55cm_s": st.t_high - st.t_low,
    }
  summary = {key: _nanmean(v) for key, v in per_ep.items()}
  summary["fall_rate"] = summary.pop("fall")
  summary["success_rate"] = summary.pop("success")
  summary["ref_peak_joint_speed"] = float(cmd.motion.joint_vel.abs().max())
  vec_env.close()
  return {
    "task": args.task,
    "learner": args.learner,
    "checkpoint": str(args.checkpoint),
    "condition": "randomized" if args.randomize else "nominal",
    "obs_noise": obs_noise(args),
    "episodes": n,
    "clip_steps": num_steps,
    "max_step_reward": r_max,
    "metrics": summary,
    "per_episode": {key: v.tolist() for key, v in per_ep.items()},
  }


def _nanmean(x: torch.Tensor) -> float | None:
  x = x.float()
  ok = ~x.isnan()
  return float(x[ok].mean()) if ok.any() else None


def main(argv: list[str] | None = None) -> dict[str, Any]:
  args = parse_args(argv)
  result = evaluate(args)
  text = json.dumps(result, indent=2)
  if args.output is not None:
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text)
  print(json.dumps({"task": result["task"], "learner": result["learner"], **result["metrics"]}, indent=2))
  return result


if __name__ == "__main__":
  os.environ.setdefault("MUJOCO_GL", "disable")
  main()
