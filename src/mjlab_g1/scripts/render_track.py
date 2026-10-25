"""Render a tracking policy to mp4 without OpenGL (mujoco_warp ray-traced camera).

The Turing cluster has no OpenGL, so mjlab's OffscreenRenderer cannot run. This attaches
a world-mounted camera aimed at the torso (a body-mounted camera inherits the body's
rotation and dips below the floor) and reads RGB from mjlab's camera sensor every step.
Frames are annotated with time, reference frame and tracking error.

  # SMTrack tasks (PPO or SuperMimic, checkpoint from train_track.py):
  PYTHONPATH=src python src/mjlab_g1/scripts/render_track.py --task G1-SMTrack-Lift \
    --learner ppo --checkpoint logs/smtrack/.../model_15000.pt --output models/videos/x.mp4
  # Stock mjlab tracking runs (train_mjlab_stock_tracking.sbatch):
  PYTHONPATH=src python src/mjlab_g1/scripts/render_track.py \
    --task Mjlab-Tracking-Flat-Unitree-G1-No-State-Estimation \
    --motion-file dataset/tracking/locomotion1.npz --checkpoint logs/stock_tracking/.../model_9999.pt
"""

from __future__ import annotations

import argparse
import dataclasses
import os
from pathlib import Path
from typing import Any

import imageio
import mujoco
import numpy as np
import torch
from PIL import Image, ImageDraw

STOCK_LOAD_CFG = {
  "actor": True,
  "critic": False,
  "optimizer": False,
  "iteration": False,
  "rnd": False,
}


def parse_args() -> argparse.Namespace:
  p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  p.add_argument("--task", required=True, help="G1-SMTrack-* or an mjlab tracking task id")
  p.add_argument("--learner", choices=("ppo", "supermimic"), default="ppo")
  p.add_argument("--checkpoint", required=True, type=Path)
  p.add_argument("--motion-file", default=None, help="stock mjlab tasks only")
  p.add_argument("--frames", type=int, default=None, help="default: one pass of the clip")
  p.add_argument("--width", type=int, default=960)
  p.add_argument("--height", type=int, default=540)
  p.add_argument("--cam-pos", type=float, nargs=3, default=None,
                 help="world camera position (default: per task, see default_cam_pos)")
  p.add_argument("--start-frame", type=int, default=0, help="SMTrack: start at this reference frame")
  p.add_argument("--output", type=Path, default=None)
  p.add_argument("--label", default=None, help="first overlay line (default: task + learner)")
  p.add_argument("--device", default=None)
  return p.parse_args()


def default_cam_pos(args: argparse.Namespace) -> tuple[float, float, float]:
  """Side view centred on the walk (root x 0 -> 4 m); a closer front-diagonal view otherwise."""
  motion = f"{args.task} {args.motion_file or ''}".lower()
  if "walk" in motion or "locomotion" in motion:
    return (2.0, -3.0, 1.2)
  return (1.7, -2.3, 1.1)


def add_chase_camera(env_cfg: Any, width: int, height: int, pos: tuple[float, ...]) -> None:
  """World-mounted camera that keeps the torso centred (targetbody mode), plus a key light
  (the default scene leaves the robot dark under the ray tracer)."""
  from mjlab.sensor import CameraSensorCfg

  robot_cfg = env_cfg.scene.entities["robot"]
  spec_fn = robot_cfg.spec_fn

  def spec_with_camera() -> mujoco.MjSpec:
    spec = spec_fn()
    cam = spec.worldbody.add_camera()
    cam.name = "chase"
    cam.pos = pos
    cam.mode = mujoco.mjtCamLight.mjCAMLIGHT_TARGETBODY
    cam.targetbody = "torso_link"
    cam.fovy = 40.0
    light = spec.worldbody.add_light()
    light.name = "render_key"
    light.type = mujoco.mjtLightType.mjLIGHT_DIRECTIONAL
    light.pos = (0.0, 0.0, 4.0)
    light.dir = (0.3, 0.5, -1.0)
    light.diffuse = (0.7, 0.7, 0.7)
    light.ambient = (0.25, 0.25, 0.25)
    return spec

  robot_cfg.spec_fn = spec_with_camera
  env_cfg.scene.sensors = (
    *env_cfg.scene.sensors,
    CameraSensorCfg(
      name="chase_cam",
      camera_name="robot/chase",
      width=width,
      height=height,
      data_types=("rgb",),
      use_textures=True,
      use_shadows=False,
      enabled_geom_groups=(0, 1, 2),
    ),
  )


def build(args: argparse.Namespace, device: str) -> tuple[Any, Any, Any, str]:
  """Returns (unwrapped env, vec env, policy, label)."""
  from mjlab.rl import RslRlVecEnvWrapper

  if args.task.startswith("G1-SMTrack"):
    from mjlab_g1.envs.smtrack_env import SMTrackEnv
    from mjlab_g1.scripts.eval_track import load_agent_cfg
    from mjlab_g1.scripts.train_track import make_runner
    from mjlab_g1.tasks.smtrack import make_env_cfg

    env_cfg = make_env_cfg(args.task, play=True)
    env_cfg.scene.num_envs = 1
    add_chase_camera(env_cfg, args.width, args.height, tuple(args.cam_pos or default_cam_pos(args)))
    env = SMTrackEnv(env_cfg, device=device)
    vec = RslRlVecEnvWrapper(env)
    agent = load_agent_cfg(argparse.Namespace(agent_cfg=None, checkpoint=args.checkpoint, learner=args.learner, task=args.task))
    if args.learner == "supermimic":
      agent.setdefault("memory", {})["compile"] = False  # one rollout; skip the compile cost
    runner = make_runner(args.learner, vec, agent, None, device)
    runner.load(str(args.checkpoint), map_location=device)
    return env, vec, runner.get_inference_policy(device), f"{args.task} {args.learner}"

  import mjlab.tasks  # noqa: F401  (populates the registry)
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import MjlabOnPolicyRunner
  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

  env_cfg = load_env_cfg(args.task, play=True)
  env_cfg.scene.num_envs = 1
  if args.motion_file is None:
    raise SystemExit("--motion-file is required for stock mjlab tracking tasks")
  env_cfg.commands["motion"].motion_file = args.motion_file
  add_chase_camera(env_cfg, args.width, args.height, tuple(args.cam_pos or default_cam_pos(args)))
  agent_cfg = load_rl_cfg(args.task)
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  vec = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  runner = (load_runner_cls(args.task) or MjlabOnPolicyRunner)(vec, dataclasses.asdict(agent_cfg), device=device)
  runner.load(str(args.checkpoint), load_cfg=STOCK_LOAD_CFG, map_location=device)
  return env, vec, runner.get_inference_policy(device=device), f"stock PPO {Path(args.motion_file).stem}"


def annotate(frame: np.ndarray, lines: list[str]) -> np.ndarray:
  im = Image.fromarray(frame)
  d = ImageDraw.Draw(im)
  d.rectangle([0, 0, 330, 16 * len(lines) + 6], fill=(0, 0, 0))
  for i, line in enumerate(lines):
    d.text((6, 4 + 16 * i), line, fill=(255, 255, 0))
  return np.asarray(im)


def main() -> None:
  args = parse_args()
  device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
  env, vec, policy, label = build(args, device)
  label = args.label or label
  cmd = env.command_manager.get_term("motion")
  num_frames = int(cmd.motion.time_step_total)
  if args.start_frame > 0:
    # Exact reference state at the start frame (play cfg: no reset noise), as in
    # phase_map_track.py; histories / last action are cleared by the reset.
    ids = torch.arange(env.num_envs, device=env.device)
    env._reset_idx(ids)
    cmd.reset_to_frame(ids, args.start_frame)
    env.scene.write_data_to_sim()
    env.sim.forward()
    cmd.update_relative_body_poses()
  steps = args.frames or (num_frames - 1 - args.start_frame)
  has_box = getattr(env, "has_box", False)

  frames = []
  obs = vec.get_observations().to(device)
  with torch.inference_mode():
    for t in range(steps):
      obs, _, dones, _ = vec.step(policy(obs).to(device))
      obs = obs.to(device)
      env.sim.sense()  # ray-traced camera render (no GL)
      rgb = env.scene["chase_cam"].data.rgb[0].cpu().numpy()
      body_err = (cmd.body_pos_relative_w[0] - cmd.robot_body_pos_w[0]).norm(dim=-1).mean().item()
      lines = [label, f"t={t * env.step_dt:5.2f}s  frame {int(cmd.time_steps[0])}/{num_frames - 1}",
               f"body pos err {100 * body_err:5.1f} cm",
               f"max |joint vel| {env.scene['robot'].data.joint_vel[0].abs().max().item():5.1f} rad/s"]
      if has_box:
        box_z = env.scene["box"].data.root_link_pos_w[0, 2].item()
        lines.append(f"box z {box_z:.2f} m")
      if bool(dones[0]):
        lines.append("RESET")
      frames.append(annotate(rgb, lines))

  out = args.output or args.checkpoint.parent / "videos" / f"{args.checkpoint.stem}.mp4"
  out.parent.mkdir(parents=True, exist_ok=True)
  fps = int(round(1.0 / env.step_dt))
  imageio.mimsave(str(out), frames, fps=fps)
  print(f"[render_track] wrote {len(frames)} frames ({fps} fps) to {out}")


if __name__ == "__main__":
  os.environ.setdefault("MUJOCO_GL", "disable")
  main()
