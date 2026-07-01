"""Render a fixed-length policy video without opening an interactive viewer."""

from __future__ import annotations

import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import tyro

from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.wrappers import VideoRecorder
from mjlab_g1.envs.g1_dualarm_rl_env import (
  G1DualarmManagerBasedRlEnv,
  G1DualarmManagerBasedRlEnvCfg,
)
from mjlab_g1.envs.g1_locomotion_rl_env import (
  G1LocomotionManagerBasedRlEnv,
  G1LocomotionManagerBasedRlEnvCfg,
)
from mjlab_g1.rl import RslRlVecEnvWrapper
from mjlab_g1.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls
from rsl_rl.runners import OnPolicyRunner


@dataclass(frozen=True)
class RenderVideoConfig:
  checkpoint_file: str | None = None
  num_envs: int = 1
  device: str | None = None
  video_length: int = 500
  video_height: int = 720
  video_width: int = 1280
  output_dir: str | None = None
  name_prefix: str = "policy"


def _checkpoint_iteration(path: Path) -> int:
  match = re.fullmatch(r"model_(\d+)\.pt", path.name)
  return int(match.group(1)) if match else -1


def _find_latest_checkpoint(experiment_name: str) -> Path:
  log_root = Path("logs") / "rsl_rl" / experiment_name
  checkpoints = list(log_root.glob("*/model_*.pt"))
  if not checkpoints:
    raise FileNotFoundError(f"No checkpoints found under {log_root}")
  return max(checkpoints, key=lambda path: (path.stat().st_mtime, _checkpoint_iteration(path)))


def run_render_video(task_id: str, cfg: RenderVideoConfig) -> Path:
  configure_torch_backends()

  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
  env_cfg = load_env_cfg(task_id, play=True)
  agent_cfg = load_rl_cfg(task_id)
  checkpoint_path = (
    Path(cfg.checkpoint_file)
    if cfg.checkpoint_file is not None
    else _find_latest_checkpoint(agent_cfg.experiment_name)
  )
  if not checkpoint_path.exists():
    raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

  env_cfg.scene.num_envs = cfg.num_envs
  env_cfg.viewer.height = cfg.video_height
  env_cfg.viewer.width = cfg.video_width

  if isinstance(env_cfg, G1LocomotionManagerBasedRlEnvCfg):
    env = G1LocomotionManagerBasedRlEnv(
      cfg=env_cfg,
      device=device,
      render_mode="rgb_array",
    )
  elif isinstance(env_cfg, G1DualarmManagerBasedRlEnvCfg):
    env = G1DualarmManagerBasedRlEnv(
      cfg=env_cfg,
      device=device,
      render_mode="rgb_array",
    )
  else:
    raise TypeError(f"Unsupported env cfg type: {type(env_cfg)}")

  video_dir = Path(cfg.output_dir) if cfg.output_dir is not None else checkpoint_path.parent / "videos" / "render"
  env = VideoRecorder(
    env,
    video_folder=video_dir,
    step_trigger=lambda step: step == 0,
    video_length=cfg.video_length,
    name_prefix=cfg.name_prefix,
    disable_logger=True,
  )
  env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

  runner_cls = load_runner_cls(task_id) or OnPolicyRunner
  runner = runner_cls(env, asdict(agent_cfg), device=device)
  runner.load(
    str(checkpoint_path),
    load_cfg={
      "actor": True,
      "critic": False,
      "optimizer": False,
      "iteration": False,
      "rnd": False,
      "discriminator": False,
    },
    map_location=device,
  )
  policy = runner.get_inference_policy(device=device)

  obs = env.get_observations().to(device)
  with torch.inference_mode():
    for _ in range(cfg.video_length):
      actions = policy(obs)
      obs, _, _, _ = env.step(actions.to(env.device))
      obs = obs.to(device)

  env.close()
  print(f"[INFO] Rendered {cfg.video_length} frames from {checkpoint_path}")
  print(f"[INFO] Video directory: {video_dir}")
  return video_dir


def main() -> None:
  all_tasks = list_tasks()
  chosen_task, remaining_args = tyro.cli(
    tyro.extras.literal_type_from_choices(all_tasks),
    add_help=False,
    return_unknown_args=True,
  )
  args = tyro.cli(
    RenderVideoConfig,
    args=remaining_args,
    prog=sys.argv[0] + f" {chosen_task}",
    config=(tyro.conf.AvoidSubcommands, tyro.conf.FlagConversionOff),
  )
  run_render_video(chosen_task, args)


if __name__ == "__main__":
  main()
