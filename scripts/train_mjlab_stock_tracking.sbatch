#!/bin/bash
# Stock mjlab BeyondMimic PPO tracking on one of the repo's converted clips.
# Reference point for the SuperMimic side-by-side; uses mjlab's own task and
# train script unchanged (not the paper-matched PPO-Sim baseline).
#SBATCH --job-name=stock_track
#SBATCH --account=mmaghelih
#SBATCH --partition=short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"

export GIT_PYTHON_REFRESH=quiet MUJOCO_GL=disable WANDB_MODE=disabled PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

MOTION="${MOTION:-dataset/tracking/locomotion1.npz}"
NUM_ENVS="${NUM_ENVS:-8192}"
MAX_ITERS="${MAX_ITERS:-10000}"
RUN_NAME="${RUN_NAME:-stock_$(basename "$MOTION" .npz)}"

echo "Job ${SLURM_JOB_ID:-none} on $(hostname): MOTION=$MOTION NUM_ENVS=$NUM_ENVS MAX_ITERS=$MAX_ITERS"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv || true

.venv/bin/python -P -m mjlab.scripts.train Mjlab-Tracking-Flat-Unitree-G1-No-State-Estimation \
  --env.commands.motion.motion_file "$MOTION" \
  --env.scene.num_envs "$NUM_ENVS" \
  --agent.max_iterations "$MAX_ITERS" \
  --agent.logger tensorboard \
  --agent.run_name "$RUN_NAME" \
  --log_root logs/stock_tracking
