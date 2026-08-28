#!/bin/bash
#SBATCH --job-name=g1_train
#SBATCH --account=cchamzas
#SBATCH --partition=quick
#SBATCH --nodes=1
#SBATCH --gres=gpu:rtx_pro_6000_b:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err

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
export UV_PYTHON_DOWNLOADS="${UV_PYTHON_DOWNLOADS:-never}"
export MUJOCO_GL="${MUJOCO_GL:-disable}"
export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_PROJECT="${WANDB_PROJECT:-ppo}"
export GIT_PYTHON_REFRESH="${GIT_PYTHON_REFRESH:-quiet}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

TASK="${TASK:-Mjlab-G1-DualArm}"
NUM_ENVS="${NUM_ENVS:-8192}"
MAX_ITERS="${MAX_ITERS:-5000}"
EXP_NAME="${EXP_NAME:-g1_dualarm}"
RUN_NAME="${RUN_NAME:-slurm}"

mkdir -p logs/slurm "$UV_CACHE_DIR"

echo "Job ID: ${SLURM_JOB_ID:-none}  Node: $(hostname)"
echo "TASK=$TASK  EXP_NAME=$EXP_NAME  RUN_NAME=$RUN_NAME"
echo "NUM_ENVS=$NUM_ENVS  MAX_ITERS=$MAX_ITERS"
nvidia-smi || true

if ! command -v uv >/dev/null 2>&1; then
  echo "uv not on PATH; install uv on the login node first." >&2
  exit 127
fi

uv sync --python "$(command -v python)"

PYTHONPATH=src uv run --no-sync python src/mjlab_g1/scripts/train.py "$TASK" \
  --env.scene.num_envs "$NUM_ENVS" \
  --agent.max_iterations "$MAX_ITERS" \
  --agent.experiment_name "$EXP_NAME" \
  --agent.run_name "$RUN_NAME" \
  --video False \
  --init_at_random_ep_len False
