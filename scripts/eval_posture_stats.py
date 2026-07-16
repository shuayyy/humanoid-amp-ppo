"""Roll out a checkpoint headlessly and print phase-aggregated posture stats.

Complements the rendered video with the numbers that define "natural":
leg left/right mismatch, pelvis height, torso pitch, and object position in
the base frame, split into pre-lift (reach) and post-lift (carry/hold)
phases. Compare directly against the mocap prior (squat: pelvis ~0.4 m,
|L-R| < 0.12 rad; hold: pelvis ~0.78 m, torso pitch 2-13 deg, waist within
~0.2 rad).

Run from the repo root on a GPU node:
    PYTHONPATH=src python scripts/eval_posture_stats.py \
        --checkpoint-file models/snapshots/g1_dualarm_v6_model_12300.pt
"""

from __future__ import annotations

import argparse
from dataclasses import asdict

import torch

from mjlab.utils.lab_api.math import quat_apply_inverse
from mjlab.utils.torch import configure_torch_backends
from mjlab_g1.envs.g1_dualarm_rl_env import G1DualarmManagerBasedRlEnv
from mjlab_g1.rl import RslRlVecEnvWrapper
from mjlab_g1.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

TASK_ID = "Mjlab-G1-DualArm"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-file", required=True)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--steps", type=int, default=400)
    args = parser.parse_args()

    configure_torch_backends()
    device = "cuda:0"

    env_cfg = load_env_cfg(TASK_ID, play=True)
    env_cfg.scene.num_envs = args.num_envs
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

    left_ids = torch.as_tensor(
        env.left_leg_sym_joint_ids, device=device, dtype=torch.long
    )
    right_ids = torch.as_tensor(
        env.right_leg_sym_joint_ids, device=device, dtype=torch.long
    )
    waist_ids = torch.as_tensor(
        env.waist_joint_ids, device=device, dtype=torch.long
    )
    torso_id = int(env.torso_body_id[0])

    records: dict[str, list[torch.Tensor]] = {k: [] for k in (
        "prelift_leg_mismatch", "prelift_pelvis_z",
        "hold_leg_mismatch", "hold_pelvis_z", "hold_torso_pitch_deg",
        "hold_waist_dev", "hold_obj_forward", "hold_obj_lateral",
    )}

    obs = wrapped.get_observations().to(device)
    with torch.inference_mode():
        for _ in range(args.steps):
            actions = policy(obs)
            obs, _, _, _ = wrapped.step(actions.to(wrapped.device))
            obs = obs.to(device)

            jp = env.robot.data.joint_pos
            mismatch = torch.sum(
                torch.abs(jp[:, left_ids] - jp[:, right_ids]), dim=-1
            )
            pelvis_z = env.robot.data.root_link_pos_w[:, 2]
            started, _ = env._lift_progress()
            holding = env.success_hold_buf > 0

            prelift = ~started
            if prelift.any():
                records["prelift_leg_mismatch"].append(mismatch[prelift])
                records["prelift_pelvis_z"].append(pelvis_z[prelift])
            if holding.any():
                torso_quat = env.robot.data.body_link_quat_w[:, torso_id]
                g = torch.zeros(env.num_envs, 3, device=device)
                g[:, 2] = -1.0
                proj = quat_apply_inverse(torso_quat, g)
                pitch_deg = torch.rad2deg(torch.asin(proj[:, 0].clamp(-1, 1)))
                waist_dev = torch.sum(
                    torch.abs(
                        jp[:, waist_ids]
                        - env.robot.data.default_joint_pos[:, waist_ids]
                    ),
                    dim=-1,
                )
                rel_w = (
                    env.toaster.data.root_link_pos_w
                    - env.robot.data.root_link_pos_w
                )
                rel_b = quat_apply_inverse(
                    env.robot.data.root_link_quat_w, rel_w
                )
                records["hold_leg_mismatch"].append(mismatch[holding])
                records["hold_pelvis_z"].append(pelvis_z[holding])
                records["hold_torso_pitch_deg"].append(pitch_deg[holding])
                records["hold_waist_dev"].append(waist_dev[holding])
                records["hold_obj_forward"].append(rel_b[holding, 0])
                records["hold_obj_lateral"].append(rel_b[holding, 1])

    print(f"[STATS] checkpoint={args.checkpoint_file}")
    for name, chunks in records.items():
        if not chunks:
            print(f"  {name:24s} (no samples)")
            continue
        v = torch.cat(chunks)
        print(f"  {name:24s} mean={v.mean():7.3f}  p10={v.quantile(0.1):7.3f} "
              f"p90={v.quantile(0.9):7.3f}  n={v.numel()}")


if __name__ == "__main__":
    main()
