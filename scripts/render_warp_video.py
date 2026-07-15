"""Render a policy rollout video WITHOUT OpenGL.

Turing's compute nodes ship no libEGL/libOSMesa, so mujoco's offscreen
renderer cannot run there. This script instead renders through mjlab's
mujoco_warp ray-traced camera sensor — the same GL-free CUDA pipeline the
dual-arm task used for depth during training — by attaching a chase camera
to the robot and encoding the per-step RGB frames with imageio.

Run from the repo root on a GPU node:
    PYTHONPATH=src python scripts/render_warp_video.py \
        --checkpoint-file models/snapshots/g1_dualarm_v3_final_model_4500.pt
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import imageio
import mujoco
import torch

from mjlab.sensor import CameraSensorCfg
from mjlab.utils.torch import configure_torch_backends
from mjlab_g1.envs.g1_dualarm_rl_env import G1DualarmManagerBasedRlEnv
from mjlab_g1.rl import RslRlVecEnvWrapper
from mjlab_g1.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

TASK_ID = "Mjlab-G1-DualArm"


def add_chase_camera(env_cfg, width: int, height: int) -> None:
    """Attach a target-tracking third-person camera to the robot spec."""
    robot_cfg = env_cfg.scene.entities["robot"]
    original_spec_fn = robot_cfg.spec_fn

    def spec_with_chase_camera() -> mujoco.MjSpec:
        spec = original_spec_fn()
        cam = spec.worldbody.add_camera()
        cam.name = "chase"
        # World-mounted, aimed at the torso. Mounting on torso_link made the
        # camera inherit the body's rotation: once the policy leaned into its
        # hold posture the camera dipped below the floor plane, which
        # backface-culls from underneath (v4 footage went dark after ~1s).
        # targetbody mode keeps it aimed at the torso without orientation math.
        cam.pos = (1.9, -2.6, 1.4)
        cam.mode = mujoco.mjtCamLight.mjCAMLIGHT_TARGETBODY
        cam.targetbody = "torso_link"
        cam.fovy = 42.0
        return spec

    robot_cfg.spec_fn = spec_with_chase_camera
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-file", required=True)
    parser.add_argument("--video-length", type=int, default=500)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--output", default="models/videos/v3/policy_warp.mp4")
    parser.add_argument(
        "--clock-lift",
        action="store_true",
        help="Legacy clock-triggered lift reference; required for checkpoints "
        "trained before the contact-triggered change (baseline, v2).",
    )
    args = parser.parse_args()

    configure_torch_backends()
    device = "cuda:0"

    env_cfg = load_env_cfg(TASK_ID, play=True)
    env_cfg.scene.num_envs = 1
    if args.clock_lift:
        env_cfg.lift_trigger_contact_steps = 0
    add_chase_camera(env_cfg, args.width, args.height)

    agent_cfg = load_rl_cfg(TASK_ID)
    env = G1DualarmManagerBasedRlEnv(cfg=env_cfg, device=device)
    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    runner_cls = load_runner_cls(TASK_ID)
    runner = runner_cls(wrapped, asdict(agent_cfg), device=device)
    runner.load(
        args.checkpoint_file,
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

    print(
        f"[INFO] lift_trigger_contact_steps={env.cfg.lift_trigger_contact_steps} "
        f"(0 = legacy clock trigger) | assistance={env.virtual_pd_assistance_scale} "
        f"| reset_height_frac={env.object_reset_height_frac}"
    )

    fps = int(round(1.0 / env.step_dt))
    frames = []
    obs = wrapped.get_observations().to(device)
    with torch.inference_mode():
        for step in range(args.video_length):
            actions = policy(obs)
            obs, _, _, _ = wrapped.step(actions.to(wrapped.device))
            obs = obs.to(device)

            env.sim.sense()  # ray-traced camera render (no GL)
            rgb = env.scene["chase_cam"].data.rgb
            assert rgb is not None
            frames.append(rgb[0].cpu().numpy())

            if step % 25 == 0:
                ref_pos, _ = env.get_object_trajectory_reference()
                print(
                    f"t={step * env.step_dt:5.2f}s "
                    f"obj_z={env.toaster.data.root_link_pos_w[0, 2]:.3f} "
                    f"ref_z={ref_pos[0, 2]:.3f} "
                    f"base_z={env.robot.data.root_link_pos_w[0, 2]:.3f} "
                    f"contacts={bool(env._both_marker_contacts()[0])}"
                )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(str(out_path), frames, fps=fps)
    print(f"[INFO] Wrote {len(frames)} frames ({fps} fps) to {out_path}")


if __name__ == "__main__":
    main()
