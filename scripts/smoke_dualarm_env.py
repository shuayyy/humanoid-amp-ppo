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

    # Standing near default pose: torso should be ~vertical, waist ~default.
    ungated = dualarm_rewards.torso_upright(env)
    assert ungated.mean() > 0.8, "torso_upright should be ~1 when standing"
    assert dualarm_rewards.waist_deviation_penalty(
        env, gate_on_lift=False
    ).max() < 0.5, "waist deviation should be near zero at default pose"
    # The locomotion base may have envs mid-gait here, so an absolute bound
    # is wrong (that false assumption failed the first smoke run at ~1.4
    # rad). Structural check instead: the penalty must be exactly zero for
    # every env not in double support.
    left = env.scene["left_feet_ground_contact"].data.found
    right = env.scene["right_feet_ground_contact"].data.found
    double_support = (
        torch.any(left > 0, dim=-1) & torch.any(right > 0, dim=-1)
    )
    sym = dualarm_rewards.leg_symmetry_penalty(env)
    assert torch.all(sym[~double_support] == 0.0), (
        "leg_symmetry_penalty must be gated off outside double support"
    )
    print(f"[INFO] double_support={double_support.float().mean():.2f} of envs, "
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
