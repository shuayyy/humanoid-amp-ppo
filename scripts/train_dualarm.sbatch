#!/bin/bash
#SBATCH --job-name=g1_dualarm
#SBATCH --account=mmaghelih
#SBATCH --partition=short
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --gres=gpu:rtx_pro_6000_b:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=16:00:00
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err

# Dual-arm training with the frozen locomotion base policy (ResMimic-style
# residual learning). Submit from the repo root:
#   sbatch scripts/train_dualarm.sbatch
#
# Env overrides (forwarded to train_dualarm_from_locomotion.py):
#   NUM_ENVS=2048 WANDB_PROJECT=ppo SEED=42 MAX_ITERS=50000 RUN_NAME=myrun \
#     sbatch scripts/train_dualarm.sbatch
#   FRESH=1  force a clean run instead of auto-resuming an old checkpoint.

set -euo pipefail

# `module` is a shell function from /etc/profile.d; it is NOT inherited when
# the submitting shell lacks it. Source it explicitly so submissions work
# from any environment (cron, CI, non-login shells).
if ! command -v module >/dev/null 2>&1; then
  source /etc/profile.d/modules.sh
fi

module purge
# This repo needs Python >=3.12 (pyproject requires-python), unlike the
# unitree_rl_mjlab repo which uses 3.11.
module load python/3.12.3/mftt2ua
module load cuda12.2/toolkit/12.2.2
# Compute nodes have no system git (login nodes do); uv needs it to fetch the
# mjlab git dependency during sync. Jobs 2088096/7 failed on exactly this.
module load git/2.42.0/pcfyx2d

cd "${SLURM_SUBMIT_DIR:-$PWD}"

export PATH="$HOME/.local/bin:$PATH"
export PYTHONUNBUFFERED=1
export UV_CACHE_DIR="${UV_CACHE_DIR:-$HOME/scratch/.cache/uv}"
export UV_PYTHON_DOWNLOADS="${UV_PYTHON_DOWNLOADS:-never}"
# MUJOCO_GL=egl crashes on this cluster's compute nodes (see the note in
# unitree_rl_mjlab/scripts/train_iceskate.sbatch). The dual-arm depth camera
# renders via mujoco_warp (GPU ray-tracing, no OpenGL) and training runs with
# --video False, so GL can be disabled entirely.
export MUJOCO_GL="${MUJOCO_GL:-disable}"
export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
# Compute nodes have no internet: log wandb offline and sync later from the
# login node with `wandb sync logs/rsl_rl/g1_dualarm/<run>/wandb/latest-run`.
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_PROJECT="${WANDB_PROJECT:-ppo}"
# rsl_rl's logger imports GitPython at module load; degrade to a warning if
# `git` isn't on PATH after module purge.
export GIT_PYTHON_REFRESH="${GIT_PYTHON_REFRESH:-quiet}"

# Reduce allocator fragmentation next to the warp sim's large allocations.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

export NUM_ENVS="${NUM_ENVS:-16384}"
export BASE_CHECKPOINT="${BASE_CHECKPOINT:-models/locomotion.pt}"
# Experiment name: also the log/auto-resume directory. Give parallel runs
# (e.g. the v2 robustness config) distinct names so they never resume each
# other's checkpoints.
export EXP_NAME="${EXP_NAME:-g1_dualarm}"

mkdir -p logs/slurm "$UV_CACHE_DIR"

echo "Job ID: ${SLURM_JOB_ID:-none}"
echo "Node: $(hostname)"
echo "Working dir: $(pwd)"
echo "Python: $(command -v python)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "WANDB_PROJECT=$WANDB_PROJECT  WANDB_MODE=$WANDB_MODE"
echo "NUM_ENVS=$NUM_ENVS"
echo "BASE_CHECKPOINT=$BASE_CHECKPOINT"
echo "SEED=${SEED:-<cfg default>}  MAX_ITERS=${MAX_ITERS:-<cfg default>}  RUN_NAME=${RUN_NAME:-<none>}"

nvidia-smi || true

if ! command -v uv >/dev/null 2>&1; then
  echo "uv was not found on PATH."
  echo "Install it once on the login node with:"
  echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
  exit 127
fi

uv sync --python "$(command -v python)"

# Auto-resume if this experiment already has a checkpoint (crashed or
# timed-out previous job). Set FRESH=1 to force a clean run.
LOG_ROOT="$(pwd)/logs/rsl_rl/${EXP_NAME}"
if [[ "${FRESH:-0}" != "1" ]] && [[ -d "$LOG_ROOT" ]] && \
   find "$LOG_ROOT" -mindepth 2 -maxdepth 2 -name "model_*.pt" -print -quit | grep -q .; then
  echo "Found existing checkpoint under $LOG_ROOT — resuming."
  export RESUME=1
else
  echo "Starting fresh (no resume)."
fi

uv run --no-sync python train_dualarm_from_locomotion.py
