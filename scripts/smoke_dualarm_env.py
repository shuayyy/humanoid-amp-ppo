"""Build the dual-arm env and sanity-check reward terms on a GPU node.

Cheap insurance before a long training job whenever reward terms change:
constructs the env (which resolves every configured term), steps it with
zero actions, and prints the posture-related terms directly so sign and
magnitude errors surface in minutes instead of hours.

Run from the repo root on a GPU node:
    PYTHONPATH=src python scripts/smoke_dualarm_env.py
"""

from __future__ import annotations

import torch

from mjlab.utils.torch import configure_torch_backends
from mjlab_g1.envs.g1_dualarm_rl_env import G1DualarmManagerBasedRlEnv
from mjlab_g1.tasks.dualarm.mdp import rewards as dualarm_rewards
from mjlab_g1.tasks.registry import load_env_cfg


def main() -> None:
    configure_torch_backends()
    device = "cuda:0"

    env_cfg = load_env_cfg("Mjlab-G1-DualArm", play=True)
    env_cfg.scene.num_envs = 16
    env = G1DualarmManagerBasedRlEnv(cfg=env_cfg, device=device)
    env.reset()

    print("[INFO] dualarm reward terms:",
          list(env.dualarm_reward_manager.active_terms))

    # RSI checks belong HERE, before any stepping: the frozen locomotion
    # base pulls initialized squats back toward standing within ~30 steps
    # (that mis-timing produced the false alarm in smoke 2101636).
    if env.cfg.rsi_fraction > 0.0:
        pelvis_z = env.robot.data.root_link_pos_w[:, 2]
        squat0 = pelvis_z < 0.6
        frac = squat0.float().mean().item()
        assert 0.05 < frac < 0.6, (
            f"RSI fraction {env.cfg.rsi_fraction} but {frac:.2f} of envs "
            "start below pelvis 0.6"
        )
        knee_ids, _ = env.robot.find_joints(
            ("left_knee_joint", "right_knee_joint"), preserve_order=True
        )
        knees0 = env.robot.data.joint_pos[:, knee_ids].mean(dim=-1)
        assert knees0[squat0].mean() > 1.5, (
            "RSI envs should start with deeply bent knees (mocap 1.9-2.2), "
            f"got mean {knees0[squat0].mean():.2f}"
        )
        print(
            f"[INFO] RSI at reset: {frac:.2f} of envs squatting, knees mean "
            f"{knees0[squat0].mean():.2f} rad, squat_shaping mean "
            f"{dualarm_rewards.squat_shaping(env)[squat0].mean():.3f}"
        )

    action_dim = env.action_manager.total_action_dim
    with torch.inference_mode():
        for step in range(30):
            actions = torch.zeros(env.num_envs, action_dim, device=device)
            env.step(actions)

        checks = {
            "torso_upright (ungated)": dualarm_rewards.torso_upright(env),
            "torso_upright (lift-gated)": dualarm_rewards.torso_upright(
                env, gate_on_lift=True
            ),
            "waist_deviation_penalty": dualarm_rewards.waist_deviation_penalty(
                env
            ),
            "object_centered": dualarm_rewards.object_centered(env),
            "leg_symmetry_penalty": dualarm_rewards.leg_symmetry_penalty(env),
            "upright (pelvis, ungated)": dualarm_rewards.upright(env),
        }

    for name, val in checks.items():
        print(f"  {name:32s} mean={val.mean():.4f} "
              f"min={val.min():.4f} max={val.max():.4f}")

    # Post-settling posture checks: RSI envs may still be recovering from
    # their squat start (or mid-episode after resets during the step loop),
    # so posture assertions apply to the STANDING subset only.
    standing = env.robot.data.root_link_pos_w[:, 2] > 0.7
    assert standing.any(), "expected some standing envs after settling"
    ungated = dualarm_rewards.torso_upright(env)
    assert ungated[standing].mean() > 0.8, (
        "torso_upright should be ~1 when standing"
    )
    assert dualarm_rewards.waist_deviation_penalty(
        env, gate_on_lift=False
    )[standing].max() < 0.5, "waist deviation should be near zero at default pose"
    # The locomotion base may have envs mid-gait here, so an absolute bound
    # is wrong (that false assumption failed the first smoke run at ~1.4
    # rad). Structural check instead: the penalty must be exactly zero for
    # every env outside its gate (double support OR near the object — the
    # proximity arm closes v6's single-support kneel loophole).
    left = env.scene["left_feet_ground_contact"].data.found
    right = env.scene["right_feet_ground_contact"].data.found
    double_support = (
        torch.any(left > 0, dim=-1) & torch.any(right > 0, dim=-1)
    )
    near_object = (
        torch.linalg.vector_norm(
            env.robot.data.root_link_pos_w[:, :2]
            - env.toaster.data.root_link_pos_w[:, :2],
            dim=-1,
        )
        < 0.7
    )
    gate = double_support | near_object
    sym = dualarm_rewards.leg_symmetry_penalty(env)
    assert torch.all(sym[~gate] == 0.0), (
        "leg_symmetry_penalty must be zero outside its gate"
    )
    print(f"[INFO] double_support={double_support.float().mean():.2f} "
          f"near_object={near_object.float().mean():.2f} of envs, "
          f"leg_symmetry mean={sym.mean():.4f}")

    # Phase-scheduled AMP hook: pre-lift envs get the boosted coef.
    from mjlab_g1.tasks.registry import load_rl_cfg

    agent_cfg = load_rl_cfg("Mjlab-G1-DualArm")
    prelift_coef = agent_cfg.amp_prelift_reward_coef
    started, _ = env._lift_progress()
    print(f"[INFO] amp_prelift_reward_coef={prelift_coef} "
          f"lift_started={started.float().mean():.2f} of envs")
    print("[INFO] smoke test passed")


if __name__ == "__main__":
    main()
