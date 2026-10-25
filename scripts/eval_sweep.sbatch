#!/bin/bash
# Evaluate a run's checkpoints (every STRIDE-th, plus the last) with eval_track.py for learning
# curves. Deterministic paper protocol (frame 0, no DR / noise), so few episodes suffice.
#   sbatch --export=ALL,RUN_DIR=logs/smtrack/G1-SMTrack-Walk/ppo/<run>,STRIDE=4 scripts/eval_sweep.sbatch
#SBATCH --job-name=eval_sweep
#SBATCH --account=mmaghelih
#SBATCH --partition=short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
export PYTHONPATH=src GIT_PYTHON_REFRESH=quiet MUJOCO_GL=disable WANDB_MODE=disabled PYTHONUNBUFFERED=1

RUN_DIR="${RUN_DIR:?set RUN_DIR}"
STRIDE="${STRIDE:-1}"
EPISODES="${EPISODES:-4}"
TASK=$(basename "$(dirname "$(dirname "$RUN_DIR")")")
LEARNER=$(basename "$(dirname "$RUN_DIR")")

mapfile -t CKPTS < <(ls "$RUN_DIR"/model_*.pt | sort -V)
LAST="${CKPTS[-1]}"
for i in "${!CKPTS[@]}"; do
  CKPT="${CKPTS[$i]}"
  (( i % STRIDE == 0 )) || [[ "$CKPT" == "$LAST" ]] || continue
  OUT="$RUN_DIR/eval_$(basename "$CKPT" .pt).json"
  [[ -f "$OUT" ]] && continue
  echo "== $CKPT"
  .venv/bin/python src/mjlab_g1/scripts/eval_track.py --task "$TASK" --learner "$LEARNER" \
    --checkpoint "$CKPT" --episodes "$EPISODES" --device cuda:0 --output "$OUT"
done
