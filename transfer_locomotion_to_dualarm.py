#!/usr/bin/env python3
"""Transfer locomotion actor weights into a dual-arm checkpoint (partial, column-aware).

Why this is not a plain state_dict copy
---------------------------------------
The locomotion and dual-arm policies have DIFFERENT observation spaces, so a
naive shape-equality copy silently skips the two tensors that actually carry the
locomotion prior: the input projection ``mlp.0.weight`` and the observation
normalizer. This script instead copies them column-by-column.

The layout that makes this correct (verified against the configs):

* Both policy groups use ``history_length=5`` with term-major flattening, i.e.
  ``[termA_t0..termA_t4, termB_t0..termB_t4, ...]``.
* The locomotion policy obs is EXACTLY the 6 proprio terms
  ``base_lin_vel(3), base_ang_vel(3), projected_gravity(3),
  joint_pos(29), joint_vel(29), actions(29)`` -> 96/frame * 5 = 480 dims.
* The dual-arm policy obs begins with those SAME 6 terms in the SAME order, then
  appends object_pose/grasp markers/trajectory/depth -> 304/frame * 5 = 1520.

=> The dual-arm actor's input columns [0:480] are byte-for-byte the locomotion
   obs. So we copy the whole locomotion input weight into dst[:, :480], and ZERO
   dst[:, 480:] so the warm-started policy initially behaves exactly like the
   locomotion policy (object/depth inputs are ignored until training learns
   them). Only the ACTOR is transferred; the critic/optimizer/discriminator are
   left as-is and start fresh.
"""

from pathlib import Path

import torch

# The 6 shared proprio terms are identical between the two policy obs groups, so
# the locomotion obs width equals the width of the dual-arm proprio prefix. We
# derive this at runtime from the source checkpoint rather than hardcoding it.
NORMALIZER_KEYS = ("obs_normalizer._mean", "obs_normalizer._var", "obs_normalizer._std")
INPUT_WEIGHT_KEY = "mlp.0.weight"


def _transfer_actor(src_actor: dict, dst_actor: dict) -> tuple[int, int, list[str]]:
    """Copy locomotion actor weights into the dual-arm actor state dict (in place).

    Returns (n_full_copies, shared_width, skipped_keys).
    """
    src_in = src_actor[INPUT_WEIGHT_KEY]
    dst_in = dst_actor[INPUT_WEIGHT_KEY]
    shared = src_in.shape[1]  # locomotion obs width == dual-arm proprio prefix width

    # Invariants that guarantee the column mapping is a valid prefix copy.
    assert src_in.shape[0] == dst_in.shape[0], (
        f"actor hidden width mismatch: {src_in.shape} vs {dst_in.shape}"
    )
    assert dst_in.shape[1] >= shared, (
        f"dst obs width {dst_in.shape[1]} smaller than src {shared}; "
        "obs layout assumption violated."
    )

    n_full = 0
    skipped: list[str] = []
    for key, src_val in src_actor.items():
        if key not in dst_actor:
            skipped.append(key)
            continue

        if key == INPUT_WEIGHT_KEY:
            # Proprio prefix gets locomotion weights; object/depth columns zeroed
            # so the initial policy == locomotion policy given proprio.
            new_w = torch.zeros_like(dst_actor[key])
            new_w[:, :shared] = src_val
            dst_actor[key] = new_w
        elif key in NORMALIZER_KEYS:
            # Proprio columns get locomotion running stats; the rest are set to
            # identity (mean 0, var/std 1) so raw object/depth pass through and
            # the running normalizer re-estimates them during training.
            new_n = torch.ones_like(dst_actor[key]) if key != "obs_normalizer._mean" else torch.zeros_like(dst_actor[key])
            new_n[:, :shared] = src_val
            dst_actor[key] = new_n
        elif key == "obs_normalizer.count":
            # Reset so the normalizer adapts to the new obs columns from scratch
            # (same robot => proprio stats reconverge quickly).
            dst_actor[key] = torch.zeros_like(dst_actor[key])
        elif src_val.shape == dst_actor[key].shape:
            # Hidden layers, output layer, biases, policy std: identical shapes.
            dst_actor[key] = src_val.clone()
            n_full += 1
        else:
            skipped.append(f"{key} (shape {tuple(src_val.shape)} != {tuple(dst_actor[key].shape)})")

    return n_full, shared, skipped


def transfer_weights(src_checkpoint: str, dst_checkpoint: str, output: str) -> Path:
    src_path, dst_path = Path(src_checkpoint), Path(dst_checkpoint)
    if not src_path.exists():
        raise FileNotFoundError(f"Source checkpoint not found: {src_checkpoint}")
    if not dst_path.exists():
        raise FileNotFoundError(f"Destination checkpoint not found: {dst_checkpoint}")

    print(f"Loading source (locomotion): {src_checkpoint}")
    src_ckpt = torch.load(src_path, map_location="cpu", weights_only=False)
    print(f"Loading destination (dual-arm template): {dst_checkpoint}")
    dst_ckpt = torch.load(dst_path, map_location="cpu", weights_only=False)

    src_actor = src_ckpt.get("actor_state_dict", {})
    dst_actor = dst_ckpt.get("actor_state_dict", {})
    if not src_actor or not dst_actor:
        raise ValueError("Missing actor_state_dict in one of the checkpoints.")

    n_full, shared, skipped = _transfer_actor(src_actor, dst_actor)

    # Write a MINIMAL actor-only warm-start file. We deliberately drop the
    # critic/optimizer/discriminator and — critically — the AMP normalizer
    # stats: AMP_PPO.load restores `amp_normalizer_*` unconditionally, and the
    # template's stale stats (old 29-dim joint_pos AMP mode) would clobber the
    # freshly-built 67-dim rich-AMP normalizer and crash the discriminator.
    # --init_checkpoint only loads the actor, so nothing else is needed.
    warmstart_ckpt = {
        "actor_state_dict": dst_actor,
        "iter": 0,  # warm-start, not a resume
        "infos": dst_ckpt.get("infos", {}),
    }

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(warmstart_ckpt, out_path)

    # Post-checks: prove the copy landed where we intended.
    check = torch.load(out_path, map_location="cpu", weights_only=False)["actor_state_dict"]
    w = check[INPUT_WEIGHT_KEY]
    assert torch.equal(w[:, :shared], src_actor[INPUT_WEIGHT_KEY]), "proprio prefix copy failed"
    assert torch.count_nonzero(w[:, shared:]) == 0, "object/depth columns not zeroed"

    print("\n✅ Transfer complete")
    print(f"  Proprio input columns copied : [0:{shared}] of {w.shape[1]}")
    print(f"  Object/depth columns zeroed  : [{shared}:{w.shape[1]}]")
    print(f"  Shape-matched layers copied  : {n_full} (hidden/output/bias/std)")
    if skipped:
        print(f"  Skipped                      : {skipped}")
    print(f"  Output                       : {out_path}")
    return out_path


if __name__ == "__main__":
    print("=" * 70)
    print("Transfer Learning: Locomotion Actor -> Dual-Arm (column-aware)")
    print("=" * 70)

    output = transfer_weights(
        src_checkpoint="models/locomotion.pt",
        dst_checkpoint="models/dual_arm.pt",
        output="models/dualarm_from_locomotion.pt",
    )

    print("\n🚀 Next: train dual-arm warm-started from the locomotion actor:")
    print(
        "\nMUJOCO_GL=egl PYTHONPATH=src python src/mjlab_g1/scripts/train.py \\\n"
        "  Mjlab-G1-DualArm \\\n"
        f"  --init_checkpoint {output} \\\n"
        "  --env.scene.num_envs 4096"
    )
