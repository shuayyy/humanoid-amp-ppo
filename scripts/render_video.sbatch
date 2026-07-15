#!/bin/bash
#SBATCH --job-name=g1_render
#SBATCH --account=mmaghelih
#SBATCH --partition=short
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --gres=gpu:rtx_pro_6000_b:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err

# Render a policy rollout video on a compute node. Probes GL backends first
# (EGL has been unreliable on this cluster's nodes; osmesa is the software
# fallback). Usage:
#   CHECKPOINT=models/snapshots/g1_dualarm_v3_final_model_4500.pt \
#     sbatch scripts/render_video.sbatch

set -euo pipefail

# `module` is a shell function from /etc/profile.d; it is NOT inherited when
# the submitting shell lacks it (job 2089699 failed exactly here). Source it
# explicitly so the script works regardless of the submitter's environment.
if ! command -v module >/dev/null 2>&1; then
  source /etc/profile.d/modules.sh
fi

module purge
module load python/3.12.3/mftt2ua
module load cuda12.2/toolkit/12.2.2
module load git/2.42.0/pcfyx2d

cd "${SLURM_SUBMIT_DIR:-$PWD}"

export PATH="$HOME/.local/bin:$PATH"
export PYTHONUNBUFFERED=1
export UV_CACHE_DIR="${UV_CACHE_DIR:-$HOME/scratch/.cache/uv}"
export GIT_PYTHON_REFRESH=quiet
export WANDB_MODE=disabled

CHECKPOINT="${CHECKPOINT:-models/snapshots/g1_dualarm_v3_final_model_4500.pt}"
VIDEO_LENGTH="${VIDEO_LENGTH:-500}"
OUT_DIR="${OUT_DIR:-models/videos/v3}"

mkdir -p logs/slurm "$OUT_DIR"

# The compute nodes ship NO libEGL/libOSMesa (verified in job 2089703), so
# OpenGL offscreen rendering is impossible there. Render through mjlab's
# mujoco_warp ray-traced camera instead — pure CUDA, no GL required.
export MUJOCO_GL=disable

# CLOCK_LIFT=1 evaluates under the legacy clock-triggered lift reference
# (required for checkpoints trained before the contact-trigger change).
EXTRA_ARGS=()
[ "${CLOCK_LIFT:-0}" = "1" ] && EXTRA_ARGS+=(--clock-lift)

PYTHONPATH=src uv run --no-sync python scripts/render_warp_video.py \
  --checkpoint-file "$CHECKPOINT" \
  --video-length "$VIDEO_LENGTH" \
  --output "$OUT_DIR/policy_warp.mp4" \
  "${EXTRA_ARGS[@]}"

ls -la "$OUT_DIR"
