"""Train PPO or SuperMimic on an SMTrack task (brief §5).

  PYTHONPATH=src python src/mjlab_g1/scripts/train_track.py G1-SMTrack-Walk --learner ppo
  PYTHONPATH=src python src/mjlab_g1/scripts/train_track.py G1-SMTrack-Lift --learner supermimic \
    --num-envs 4096 --max-iterations 100000 --seed 0 --run-name s0
  ... --smoke --device cpu      # CPU plumbing check: 8 envs, small nets, 3 iterations

Both learners build the same env cfg (only num_envs differs) and log to
logs/smtrack/<task>/<learner>/<date>_<run>/ with params/{env,agent}.{yaml,json}.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

from mjlab.utils.os import dump_yaml

LEARNERS = ("ppo", "supermimic")
DEFAULT_NUM_ENVS = {"ppo": 8192, "supermimic": 4096}  # paper PPO-Sim / Table A.5
SMOKE_NUM_ENVS = 8
SMOKE_ITERS = 3


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
  from mjlab_g1.tasks.smtrack import list_tasks

  p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  p.add_argument("task", choices=list_tasks())
  p.add_argument("--learner", choices=LEARNERS, required=True)
  p.add_argument("--num-envs", type=int, default=None)
  p.add_argument("--max-iterations", type=int, default=None)
  p.add_argument("--seed", type=int, default=0)
  p.add_argument("--run-name", default="")
  p.add_argument("--device", default=None, help="default: cuda:0 if available, else cpu")
  p.add_argument("--log-root", default="logs/smtrack")
  p.add_argument("--smoke", action="store_true", help="8 envs, widths/16, 3 iterations")
  p.add_argument("--save-interval", type=int, default=None, help="checkpoint every N iterations")
  p.add_argument(
    "--start-uniform-mix",
    type=float,
    default=None,
    help="uniform weight of the adaptive start-frame sampler (paper: 0.5; default: env cfg)",
  )
  p.add_argument(
    "--compile",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="torch.compile the networks (default on; --no-compile for eager)",
  )
  return p.parse_args(argv)


def _shrink(dims: tuple[int, ...] | list[int]) -> tuple[int, ...]:
  return tuple(max(4, d // 16) for d in dims)


def build_agent_cfg(args: argparse.Namespace) -> Any:
  from mjlab_g1.tasks.smtrack import make_ppo_cfg

  if args.learner == "ppo":
    cfg = make_ppo_cfg(args.task)
    if args.smoke:
      cfg.actor.hidden_dims = _shrink(cfg.actor.hidden_dims)
      cfg.critic.hidden_dims = _shrink(cfg.critic.hidden_dims)
      cfg.max_iterations = SMOKE_ITERS
      cfg.save_interval = 1
  else:
    from mjlab_g1.rl.supermimic.config import SuperMimicRunnerCfg

    cfg = SuperMimicRunnerCfg.smoke() if args.smoke else SuperMimicRunnerCfg()
    cfg.experiment_name = f"smtrack_{args.task}_supermimic"
  if args.learner == "supermimic":
    cfg.memory.compile = args.compile
  if args.max_iterations is not None:
    cfg.max_iterations = args.max_iterations
  if args.save_interval is not None:
    cfg.save_interval = args.save_interval
  cfg.seed = args.seed
  cfg.run_name = args.run_name
  cfg.logger = "tensorboard"
  return cfg


def make_log_dir(args: argparse.Namespace) -> Path:
  name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
  if args.run_name:
    name += f"_{args.run_name}"
  return Path(args.log_root).resolve() / args.task / args.learner / name


def dump_params(log_dir: Path, env_cfg: Any, agent_cfg: Any) -> None:
  """yaml for people, json for eval_track.py (tuples come back as lists either way)."""
  env_d, agent_d = dataclasses.asdict(env_cfg), dataclasses.asdict(agent_cfg)
  dump_yaml(log_dir / "params" / "env.yaml", env_d)
  dump_yaml(log_dir / "params" / "agent.yaml", agent_d)
  with open(log_dir / "params" / "agent.json", "w") as f:
    json.dump(agent_d, f, indent=2, default=str)


def make_runner(learner: str, env: Any, agent_cfg: dict, log_dir: str | None, device: str) -> Any:
  """Runner for either learner; `agent_cfg` is a dict (runners mutate it in place)."""
  if learner == "ppo":
    from mjlab.rl import MjlabOnPolicyRunner

    # mjlab's runner strips the cnn/rnn keys the vendored MLPModel rejects.
    return MjlabOnPolicyRunner(env, agent_cfg, log_dir, device)
  from mjlab_g1.rl.supermimic.runner import SuperMimicRunner

  return SuperMimicRunner(env, agent_cfg, log_dir, device)


def main(argv: list[str] | None = None) -> Path:
  args = parse_args(argv)
  from mjlab.rl import RslRlVecEnvWrapper
  from mjlab.utils.torch import configure_torch_backends
  from mjlab_g1.envs.smtrack_env import SMTrackEnv
  from mjlab_g1.tasks.smtrack import make_env_cfg

  device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
  configure_torch_backends()
  if args.compile and not device.startswith("cuda"):
    # Inductor's CPU backend needs a C++ toolchain; CPU runs are plumbing checks anyway.
    print("[train_track] --compile ignored on CPU", flush=True)
    args.compile = False

  env_cfg = make_env_cfg(args.task)
  if args.smoke:
    num_envs = SMOKE_NUM_ENVS
  else:
    num_envs = args.num_envs or DEFAULT_NUM_ENVS[args.learner]
  env_cfg.scene.num_envs = num_envs
  env_cfg.seed = args.seed
  if args.start_uniform_mix is not None:
    env_cfg.commands["motion"].adaptive_uniform_mix = args.start_uniform_mix
  if args.learner == "ppo":
    # The clip end is the end of the task, not a time limit: with no `time_outs` in
    # extras rsl_rl's PPO does not bootstrap there, matching SuperMimic's clip_end="stop".
    env_cfg.is_finite_horizon = True
  agent_cfg = build_agent_cfg(args)

  log_dir = make_log_dir(args)
  dump_params(log_dir, env_cfg, agent_cfg)
  print(f"[train_track] {args.task} {args.learner} N={num_envs} iters={agent_cfg.max_iterations} "
        f"seed={args.seed} device={device} -> {log_dir}", flush=True)

  env = SMTrackEnv(env_cfg, device=device)
  vec_env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  agent_dict = dataclasses.asdict(agent_cfg)
  if args.learner == "ppo" and args.compile:
    from mjlab_g1.tasks.smtrack.rl_cfgs import TORCH_COMPILE_MODE

    agent_dict["torch_compile_mode"] = TORCH_COMPILE_MODE
  runner = make_runner(args.learner, vec_env, agent_dict, str(log_dir), device)
  # Start frames come from the adaptive sampler; random episode clocks would only shorten
  # the first episodes (the clip-end truncation is keyed on the reference frame).
  runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=False)
  vec_env.close()
  return log_dir


if __name__ == "__main__":
  os.environ.setdefault("MUJOCO_GL", "disable")
  main()
