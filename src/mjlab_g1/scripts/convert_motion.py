"""Convert repo mocap clips to mjlab tracking motions (.npz).

The clips are [frames, 36] qpos arrays (root pos, quat wxyz, 29 joints) at 30 fps.
mjlab's csv_to_npz expects a CSV with an xyzw quaternion, resamples to the env
control rate (50 Hz -- the MotionCommand advances one frame per env step and
never reads the fps key) and replays the clip kinematically on mjlab's G1 to
record every body's pose and velocity. The result is only valid for mjlab's G1
(30 bodies); the repo's g1_realsense.xml has an extra body that shifts the arm
body indices.

The capture rate is not stored in the clips. Walking is converted at 30 fps
(~1.0 m/s, a normal pace). The dual-arm clips are treated as 20 fps (decision
2026-10-22): at 30 fps the person grasps after a ~0.3 s dwell and lifts 0.8 m
in 1.2-1.6 s, which the robot's friction-only grasp cannot follow.

Usage (from the repo root):
  WANDB_MODE=disabled uv run python src/mjlab_g1/scripts/convert_motion.py \
    dataset/locomotion/locomotion1.npy dataset/tracking/locomotion1.npz
  ... dataset/dualarm/dual_arm1.npy dataset/tracking/dual_arm1.npz --input-fps 20
"""

import argparse
import functools
import os
import tempfile
from pathlib import Path

import numpy as np


def npy_to_csv(src: Path, dst: Path) -> None:
  q = np.load(src).astype(np.float64)
  assert q.ndim == 2 and q.shape[1] == 36, q.shape
  # wxyz -> xyzw; csv_to_npz swaps it back.
  np.savetxt(dst, q[:, [0, 1, 2, 4, 5, 6, 3, *range(7, 36)]], delimiter=",")


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  parser.add_argument("src", type=Path, help="[frames, 36] qpos .npy clip")
  parser.add_argument("dst", type=Path, help="output tracking .npz")
  parser.add_argument("--input-fps", type=float, default=30.0)
  parser.add_argument("--device", default="cuda:0")
  args = parser.parse_args()
  src, dst = args.src, args.dst
  dst.parent.mkdir(parents=True, exist_ok=True)
  os.environ.setdefault("WANDB_MODE", "disabled")

  import mjlab.scripts.csv_to_npz as csv_to_npz

  # csv_to_npz hard-codes /tmp/motion.npz; redirect the save to dst.
  savez = np.savez
  csv_to_npz.np.savez = lambda _path, **arrays: savez(dst, **arrays)
  # Its pelvis-velocity self-check uses float32 default tolerances (1e-5) and
  # trips on CPU at ~2e-5 m/s; 1e-4 still catches a real FK/velocity mismatch.
  assert_close = csv_to_npz.torch.testing.assert_close
  csv_to_npz.torch.testing.assert_close = functools.partial(
    assert_close, atol=1e-4, rtol=1e-3
  )

  with tempfile.TemporaryDirectory() as tmp:
    csv = Path(tmp) / f"{src.stem}.csv"
    npy_to_csv(src, csv)
    try:
      csv_to_npz.main(
        input_file=str(csv),
        output_name=src.stem,
        input_fps=args.input_fps,
        output_fps=50,
        device=args.device,
      )
    finally:
      csv_to_npz.np.savez = savez
      csv_to_npz.torch.testing.assert_close = assert_close
  print(f"[convert_motion] {src} @ {args.input_fps:g} fps -> {dst}")


if __name__ == "__main__":
  main()
