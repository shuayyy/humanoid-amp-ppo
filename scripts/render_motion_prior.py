"""Kinematically replay the AMP motion-prior clips and render them.

The dual-arm AMP discriminator learns its style signal from the clips in
``dataset/dualarm`` (frames of [root_pos(3), root_quat_wxyz(4), dof_pos(29)]
at 50 fps, MuJoCo joint order). This script writes those frames straight
into the sim state — no physics stepping, no policy — and renders through
the same GL-free mujoco_warp ray-traced camera as render_warp_video.py, so
it runs on Turing's compute nodes.

Run from the repo root on a GPU node:
    PYTHONPATH=src python scripts/render_motion_prior.py \
        --motion-files dataset/dualarm/dual_arm1.npy dataset/dualarm/dual_arm2.npy
"""

from __future__ import annotations

import argparse
from pathlib import Path

import imageio
import numpy as np
import torch

from mjlab.utils.torch import configure_torch_backends
from mjlab_g1.envs.g1_dualarm_rl_env import G1DualarmManagerBasedRlEnv
from mjlab_g1.tasks.registry import load_env_cfg

from render_warp_video import TASK_ID, add_chase_camera

MOTION_FPS = 50.0  # matches time_between_frames=1/50 in amp_ppo.py


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--motion-files",
        nargs="+",
        default=[
            "dataset/dualarm/dual_arm1.npy",
            "dataset/dualarm/dual_arm2.npy",
        ],
    )
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--loops", type=int, default=2)
    parser.add_argument("--output", default="models/videos/motion_priors/priors.mp4")
    args = parser.parse_args()

    configure_torch_backends()
    device = "cuda:0"

    env_cfg = load_env_cfg(TASK_ID, play=True)
    env_cfg.scene.num_envs = 1
    add_chase_camera(env_cfg, args.width, args.height)

    env = G1DualarmManagerBasedRlEnv(cfg=env_cfg, device=device)
    env.reset()

    # Park the toaster out of frame: the clips are robot-only and are not
    # aligned to this env's object spawn, so leaving it in view is misleading.
    toaster_state = env.toaster.data.default_root_state.clone()
    toaster_state[:, :2] = torch.tensor([0.0, 3.0], device=device)
    env.toaster.write_root_state_to_sim(toaster_state)

    num_joints = env.robot.data.default_joint_pos.shape[1]
    zero_vel6 = torch.zeros(1, 6, device=device)
    zero_joint_vel = torch.zeros(1, num_joints, device=device)

    frames = []
    with torch.inference_mode():
        for motion_file in args.motion_files:
            motion = np.load(motion_file)
            assert motion.shape[1] == 7 + num_joints, (
                f"{motion_file}: expected {7 + num_joints} cols, got {motion.shape[1]}"
            )
            print(f"[INFO] {motion_file}: {motion.shape[0]} frames "
                  f"({motion.shape[0] / MOTION_FPS:.2f}s at {MOTION_FPS:.0f} fps)")

            clip = torch.as_tensor(motion, dtype=torch.float32, device=device)
            for _ in range(args.loops):
                for i in range(clip.shape[0]):
                    root_state = torch.cat(
                        [clip[i : i + 1, 0:7], zero_vel6], dim=-1
                    )
                    env.robot.write_root_state_to_sim(root_state)
                    env.robot.write_joint_state_to_sim(
                        clip[i : i + 1, 7:], zero_joint_vel
                    )
                    env.scene.write_data_to_sim()
                    env.sim.forward()
                    env.sim.sense()  # ray-traced camera render (no GL)

                    rgb = env.scene["chase_cam"].data.rgb
                    assert rgb is not None
                    frames.append(rgb[0].cpu().numpy())

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(str(out_path), frames, fps=int(MOTION_FPS))
    print(f"[INFO] Wrote {len(frames)} frames ({MOTION_FPS:.0f} fps) to {out_path}")


if __name__ == "__main__":
    main()
