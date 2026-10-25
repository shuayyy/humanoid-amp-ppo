#!/bin/bash
# Side-by-side SuperMimic | PPO-Sim videos on a GPU (render_track.py's ray-traced camera; no GL).
# RENDER_LIST names a file with one line per video, using each run's newest checkpoint:
#   <name> <task> <cam x,y,z or -> <supermimic run dir> <ppo run dir>
#   sbatch --export=ALL,RENDER_LIST=renders.txt,OUT_DIR=models/videos/compare scripts/render_compare.sbatch
#SBATCH --job-name=render_compare
#SBATCH --account=mmaghelih
#SBATCH --partition=short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
export PYTHONPATH=src GIT_PYTHON_REFRESH=quiet MUJOCO_GL=disable WANDB_MODE=disabled PYTHONUNBUFFERED=1

RENDER_LIST="${RENDER_LIST:?set RENDER_LIST}"
OUT_DIR="${OUT_DIR:-models/videos/compare}"
mkdir -p "$OUT_DIR"

latest() { ls "$1"/model_*.pt | sort -V | tail -1; }
iter() { basename "$1" .pt | cut -d_ -f2; }

while read -r name task cam sm_dir ppo_dir; do
  [[ -z "$name" || "$name" == \#* ]] && continue
  sm=$(latest "$sm_dir"); ppo=$(latest "$ppo_dir")
  extra=()
  [[ "$cam" != "-" ]] && extra+=(--cam-pos ${cam//,/ })
  for spec in "supermimic $sm SuperMimic" "ppo $ppo PPO-Sim"; do
    set -- $spec
    echo "== $name $1 $(basename "$2")"
    .venv/bin/python -P -m mjlab_g1.scripts.render_track --task "$task" --learner "$1" --checkpoint "$2" \
      --device cuda:0 --label "$name: $3 it $(iter "$2")" --output "$OUT_DIR/${name}_$1_$(iter "$2").mp4" "${extra[@]}"
  done
  .venv/bin/python - "$OUT_DIR/${name}_supermimic_$(iter "$sm").mp4" "$OUT_DIR/${name}_ppo_$(iter "$ppo").mp4" \
    "$OUT_DIR/${name}_sm$(iter "$sm")_vs_ppo$(iter "$ppo").mp4" <<'PY'
import sys
import imageio, imageio.v3 as iio, numpy as np
a_path, b_path, out = sys.argv[1:]
fps = iio.immeta(a_path)["fps"]
w = imageio.get_writer(out, fps=fps, codec="libx264", macro_block_size=8, output_params=["-crf", "28", "-preset", "slow"])
for a, b in zip(iio.imiter(a_path), iio.imiter(b_path)):
  w.append_data(np.concatenate([a, np.full((a.shape[0], 8, 3), 255, np.uint8), b], axis=1))
w.close()
print(f"[render_compare] wrote {out}")
PY
done < "$RENDER_LIST"
