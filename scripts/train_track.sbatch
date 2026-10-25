#!/bin/bash
#SBATCH --job-name=smtrack
#SBATCH --account=mmaghelih
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err

# SMTrack training (brief §5): PPO or SuperMimic on G1-SMTrack-Walk / G1-SMTrack-Lift.
#   sbatch --export=ALL,TASK=G1-SMTrack-Walk,LEARNER=ppo,SEED=0 scripts/train_track.sbatch
#   (full path if sbatch is not on PATH: /cm/shared/apps/slurm/current/bin/sbatch)
# Pin a GPU type for comparison runs (protocol: same type for every run of a task), e.g.
#   sbatch --gres=gpu:rtx_pro_6000_b:1 ...   (H100 / rtx_pro_6000_b / A100 / H200 all work)
# No `uv sync` here: the venv already exists and a sync would pull CUDA torch.

set -euo pipefail

if ! command -v module >/dev/null 2>&1; then
  source /etc/profile.d/modules.sh
fi
module load git/2.42.0/pcfyx2d 2>/dev/null || true  # rsl_rl logger stores a git diff

cd "${SLURM_SUBMIT_DIR:-$PWD}"

export PATH="$HOME/.local/bin:$PATH"
export PYTHONUNBUFFERED=1
export PYTHONPATH=src
export GIT_PYTHON_REFRESH=quiet
export MUJOCO_GL=disable
export WANDB_MODE=disabled
export UV_CACHE_DIR="${UV_CACHE_DIR:-$HOME/scratch/.cache/uv}"
export UV_PYTHON_DOWNLOADS=never  # the venv python is the cluster's spack 3.12.3
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

TASK="${TASK:-G1-SMTrack-Walk}"
LEARNER="${LEARNER:-ppo}"
SEED="${SEED:-0}"
NUM_ENVS="${NUM_ENVS:-}"    # empty: 8192 (ppo) / 4096 (supermimic)
MAX_ITERS="${MAX_ITERS:-}"  # empty: cfg default (ppo 30000, supermimic 320000); absolute, not relative
RUN_NAME="${RUN_NAME:-${LEARNER}_s${SEED}}"
TRAIN_TIMEOUT="${TRAIN_TIMEOUT:-23h}"  # leaves time for the eval below inside the 24 h slot
EVAL="${EVAL:-1}"           # 1: evaluate the last checkpoint after training
EVAL_EPISODES="${EVAL_EPISODES:-64}"
SMOKE="${SMOKE:-0}"         # 1: --smoke (8 envs, small nets, 3 its): GPU plumbing check
EXTRA_ARGS="${EXTRA_ARGS:-}"  # appended verbatim to train_track.py

mkdir -p logs/slurm
echo "Job ${SLURM_JOB_ID:-none} on $(hostname): TASK=$TASK LEARNER=$LEARNER SEED=$SEED RUN_NAME=$RUN_NAME"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

ARGS=("$TASK" --learner "$LEARNER" --seed "$SEED" --run-name "$RUN_NAME" --device cuda:0)
[[ -n "$NUM_ENVS" ]] && ARGS+=(--num-envs "$NUM_ENVS")
[[ -n "$MAX_ITERS" ]] && ARGS+=(--max-iterations "$MAX_ITERS")
[[ "$SMOKE" == "1" ]] && ARGS+=(--smoke)
# shellcheck disable=SC2206
[[ -n "$EXTRA_ARGS" ]] && ARGS+=($EXTRA_ARGS)

set +e
timeout --signal=INT "$TRAIN_TIMEOUT" \
  uv run --no-sync python src/mjlab_g1/scripts/train_track.py "${ARGS[@]}"
STATUS=$?
set -e
echo "train_track exit status: $STATUS"

if [[ "$EVAL" == "1" ]]; then
  RUN_DIR=$(ls -dt logs/smtrack/"$TASK"/"$LEARNER"/*_"$RUN_NAME" 2>/dev/null | head -1 || true)
  CKPT=$(ls "$RUN_DIR"/model_*.pt 2>/dev/null | sort -V | tail -1 || true)
  if [[ -n "$CKPT" ]]; then
    echo "Evaluating $CKPT"
    uv run --no-sync python src/mjlab_g1/scripts/eval_track.py --task "$TASK" --learner "$LEARNER" \
      --checkpoint "$CKPT" --episodes "$EVAL_EPISODES" --device cuda:0 \
      --output "$RUN_DIR/eval_$(basename "$CKPT" .pt).json"
  else
    echo "No checkpoint found under ${RUN_DIR:-logs/smtrack/$TASK/$LEARNER}" >&2
  fi
fi
exit "$STATUS"
