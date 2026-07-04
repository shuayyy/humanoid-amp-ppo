#!/usr/bin/env python3
"""Train dual-arm with locomotion pre-trained weights."""

import os
import sys
import torch
from pathlib import Path

# Setup environment
os.environ["MUJOCO_GL"] = "egl"
os.environ["PYTHONPATH"] = "src"

# Add src to path
sys.path.insert(0, "src")

from mjlab_g1.scripts.train import launch_training, TrainConfig

def main():
    # Load pre-trained locomotion actor into dual-arm
    print("="*70)
    print("Training Dual-Arm with Locomotion Pre-trained Weights")
    print("="*70)

    # Create base config for dual-arm
    cfg = TrainConfig.from_task("Mjlab-G1-DualArm")

    # Reduce environments to fit in GPU memory
    cfg.env.scene.num_envs = 32
    cfg.video = False

    print(f"\n✅ Config:")
    print(f"  Task: Mjlab-G1-DualArm")
    print(f"  Environments: {cfg.env.scene.num_envs}")
    print(f"  GPU: cuda:0")
    print(f"  Video: {cfg.video}")

    print(f"\n⏳ Starting training...")
    print(f"  Environment will load pre-trained weights after initialization")

    # Launch training - will use random init, but we'll load weights after
    launch_training(task_id="Mjlab-G1-DualArm", args=cfg)


if __name__ == "__main__":
    main()
