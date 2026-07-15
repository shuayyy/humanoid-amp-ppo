#!/bin/bash
#SBATCH --job-name=g1_prior
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

# Kinematic replay of the AMP motion-prior clips (see render_motion_prior.py).
# Usage: sbatch scripts/render_motion_prior.sbatch

set -euo pipefail

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
export MUJOCO_GL=disable

OUT="${OUT:-models/videos/motion_priors/priors.mp4}"
mkdir -p logs/slurm "$(dirname "$OUT")"

PYTHONPATH=src uv run --no-sync python scripts/render_motion_prior.py \
  --output "$OUT"

ls -la "$(dirname "$OUT")"
