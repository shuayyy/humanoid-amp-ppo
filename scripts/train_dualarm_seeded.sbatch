#!/bin/bash
#SBATCH --job-name=g1_dualarm
#SBATCH --account=mmaghelih
#SBATCH --partition=short
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --gres=gpu:rtx_pro_6000_b:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err

# v9 fine-tune: v7 + anti-dive terms (prelift_peak_asymmetry, descent_speed,
# hold_stagger). Seeds the v9 log root from the NEWEST v7 checkpoint at
# runtime, so it can be queued behind the final v7 job with
# --dependency=afterany and pick up whatever iteration v7 actually reached.
# Submit from the repo root:
#   NUM_ENVS=4096 MAX_ITERS=8000 RUN_NAME=squat_v9 \
#     sbatch --dependency=afterany:<v7_jobid> scripts/train_dualarm_v9.sbatch

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"

export EXP_NAME="${EXP_NAME:-g1_dualarm_v9}"
SEED_FROM="${SEED_FROM:-logs/rsl_rl/g1_dualarm_v7}"
LOG_ROOT="logs/rsl_rl/${EXP_NAME}"

if ! find "$LOG_ROOT" -mindepth 2 -maxdepth 2 -name "model_*.pt" -print -quit \
    2>/dev/null | grep -q .; then
  latest=""
  latest_iter=-1
  for f in "$SEED_FROM"/*/model_*.pt; do
    [ -e "$f" ] || continue
    iter="${f##*model_}"
    iter="${iter%.pt}"
    if [ "$iter" -gt "$latest_iter" ]; then
      latest_iter="$iter"
      latest="$f"
    fi
  done
  if [ -z "$latest" ]; then
    echo "No checkpoint found under $SEED_FROM — cannot seed ${EXP_NAME}."
    exit 1
  fi
  # Fake early timestamp so real run dirs (created later the same day) sort
  # after the seed — same convention as v7's seed_v6 dir.
  seed_dir="$LOG_ROOT/$(date +%F)_00-00-00_seed_v7_model${latest_iter}"
  mkdir -p "$seed_dir"
  cp "$latest" "$seed_dir/"
  echo "Seeded $seed_dir with $latest"
fi

source scripts/train_dualarm.sbatch
