#!/usr/bin/env python3
"""Train the dual-arm task on top of the locomotion policy (ResMimic-style).

The locomotion actor is loaded FROZEN inside the environment and the RL policy
learns residual actions on top of it:

    applied_action = locomotion_action + residual_scale[joint] * policy_action

This replaces the earlier warm-start weight copy: the locomotion prior can no
longer be destroyed by early task-reward gradients, and the residual policy
only has to learn task-specific corrections (ResMimic, Xie et al. 2025).

Usage:
    MUJOCO_GL=egl PYTHONPATH=src python train_dualarm_from_locomotion.py

Environment overrides:
    NUM_ENVS=2048           number of parallel environments (default 1024)
    BASE_CHECKPOINT=...     locomotion checkpoint (default models/locomotion.pt)
    WANDB_PROJECT=ppo       wandb project name (default ppo)
    EXP_NAME=...            experiment name / log dir (default g1_dualarm)
    RUN_NAME=...            wandb/log run name suffix (default empty)
    SEED=42                 RNG seed (default from task cfg)
    MAX_ITERS=50000         max learning iterations (default from task cfg)
    RESUME=1                resume latest checkpoint of this experiment

Play/eval needs no extra flags: the residual base checkpoint is part of the
task's default env config, so the env composes actions identically at play
time.
"""

import dataclasses
import os
import sys

# Default to EGL but let the caller override (the cluster's compute nodes need
# MUJOCO_GL=disable; the depth camera renders via mujoco_warp, not OpenGL).
os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, "src")

from mjlab_g1.scripts.train import TrainConfig, launch_training

BASE_CHECKPOINT = os.environ.get("BASE_CHECKPOINT", "models/locomotion.pt")
NUM_ENVS = int(os.environ.get("NUM_ENVS", "1024"))
WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "ppo")
EXP_NAME = os.environ.get("EXP_NAME")
RUN_NAME = os.environ.get("RUN_NAME", "")
SEED = os.environ.get("SEED")
MAX_ITERS = os.environ.get("MAX_ITERS")
RESUME = os.environ.get("RESUME", "0") == "1"


def main() -> None:
    print("=" * 70)
    print("Dual-Arm Training with Frozen Locomotion Base (residual learning)")
    print("=" * 70)

    cfg = TrainConfig.from_task("Mjlab-G1-DualArm")

    # Env/agent cfgs are mutable; TrainConfig itself is frozen (use
    # dataclasses.replace).
    cfg.env.scene.num_envs = NUM_ENVS
    cfg.env.residual_base_checkpoint = BASE_CHECKPOINT

    cfg.agent.logger = "wandb"
    cfg.agent.wandb_project = WANDB_PROJECT
    if EXP_NAME is not None:
        cfg.agent.experiment_name = EXP_NAME
    cfg.agent.run_name = RUN_NAME
    if SEED is not None:
        cfg.agent.seed = int(SEED)
    if MAX_ITERS is not None:
        cfg.agent.max_iterations = int(MAX_ITERS)
    if RESUME:
        cfg.agent.resume = True

    cfg = dataclasses.replace(
        cfg,
        video=False,
        # The episode clock indexes the object lift trajectory; starting
        # episodes at random phases desynchronizes reference and state.
        init_at_random_ep_len=False,
    )

    print("\nConfig:")
    print(f"  Task:                Mjlab-G1-DualArm")
    print(f"  Environments:        {cfg.env.scene.num_envs}")
    print(f"  Frozen base policy:  {cfg.env.residual_base_checkpoint}")
    print(
        "  Residual scale:      "
        f"legs {cfg.env.residual_scale_legs} / "
        f"waist {cfg.env.residual_scale_waist} / "
        f"arms {cfg.env.residual_scale_arms}"
    )
    print(f"  W&B project:         {cfg.agent.wandb_project}")
    print(f"  Seed:                {cfg.agent.seed}")
    print(f"  Max iterations:      {cfg.agent.max_iterations}")
    print(f"  Resume:              {cfg.agent.resume}")
    print(f"  Video:               {cfg.video}")
    print("\nStarting training...\n")

    launch_training(task_id="Mjlab-G1-DualArm", args=cfg)


if __name__ == "__main__":
    main()
