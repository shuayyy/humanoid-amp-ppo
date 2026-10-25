"""Add a box reference trajectory to a dual-arm tracking clip (brief §3).

The dual-arm mocap has no object, so the box reference is derived from the hands:
before the grasp the box rests on the floor under the palm midpoint, after it the box
rides with the palm midpoint and turns with the pelvis yaw.

  PYTHONPATH=src .venv/bin/python src/mjlab_g1/scripts/make_box_reference.py \
    --input dataset/tracking/dual_arm1.npz [--output dataset/tracking/dual_arm1_box.npz]

CPU only (plain mujoco FK on mjlab's g1.xml); the input npz is not modified.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
import numpy as np

import mjlab

G1_XML = (
  Path(mjlab.__file__).parent / "asset_zoo" / "robots" / "unitree_g1" / "xmls" / "g1.xml"
)

# Box geometry: 0.32 m across the hands. The palm sites sit on the hand-capsule axis
# (radius 0.035), so the 0.34-0.44 m site separation while holding is a 0.27-0.37 m
# hand-surface gap (mean ~0.31 m). 0.36 m tall puts the 0.23 m grasp-time palm midpoint
# on the side faces; the earlier 0.25 m box had the hands pressing on its top face.
BOX_HALF_EXTENTS = np.array([0.15, 0.16, 0.18])
# The box centre sits this far ahead of the grasp-time palm midpoint (along the pelvis
# heading). At 0 the thighs/shins cleared the box by only 1.7 cm at the squat bottom and
# PPO policies collapsed there on knee contact; the palms stay on the 0.30 m-deep side faces.
BOX_FORWARD_M = 0.05
GRASP_Z_TOL = 0.01  # grasp = first frame within 1 cm of the lowest palm midpoint
FPS = 50.0  # tracking npz files are 50 Hz (MotionCommand advances 1 frame per step)


def _yaw_of(quat_wxyz: np.ndarray) -> np.ndarray:
  w, x, y, z = (quat_wxyz[..., i] for i in range(4))
  return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _yaw_quat(yaw: np.ndarray) -> np.ndarray:
  half = 0.5 * yaw
  q = np.zeros(yaw.shape + (4,))
  q[..., 0] = np.cos(half)
  q[..., 3] = np.sin(half)
  return q


def forward_kinematics(data: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
  """Run FK over the clip; returns (body_pos [F,30,3], palm_pos [F,2,3])."""
  model = mujoco.MjModel.from_xml_path(str(G1_XML))
  mj = mujoco.MjData(model)
  assert model.nq == 36 and model.nbody == 31, "expected mjlab's 30-body G1"
  palm_ids = [model.site("left_palm").id, model.site("right_palm").id]
  num_frames = data["joint_pos"].shape[0]
  body_pos = np.zeros((num_frames, model.nbody - 1, 3))
  palm_pos = np.zeros((num_frames, 2, 3))
  for k in range(num_frames):
    mj.qpos[:3] = data["body_pos_w"][k, 0]
    mj.qpos[3:7] = data["body_quat_w"][k, 0]
    mj.qpos[7:] = data["joint_pos"][k]
    mujoco.mj_kinematics(model, mj)
    body_pos[k] = mj.xpos[1:]
    palm_pos[k] = mj.site_xpos[palm_ids]
  return body_pos, palm_pos


def find_grasp_frame(mid_z: np.ndarray) -> int:
  half = max(len(mid_z) // 2, 1)
  z_min = mid_z[:half].min()
  return int(np.nonzero(mid_z[:half] <= z_min + GRASP_Z_TOL)[0][0])


def box_reference(
  palm_pos: np.ndarray, pelvis_quat: np.ndarray, grasp: int
) -> dict[str, np.ndarray]:
  num_frames = palm_pos.shape[0]
  mid = palm_pos.mean(axis=1)
  pelvis_yaw = np.unwrap(_yaw_of(pelvis_quat))

  heading = np.array([np.cos(pelvis_yaw[grasp]), np.sin(pelvis_yaw[grasp])])
  spawn = np.array([*(mid[grasp, :2] + BOX_FORWARD_M * heading), BOX_HALF_EXTENTS[2]])
  pos = np.repeat(spawn[None], num_frames, axis=0)
  # After the grasp the box keeps its grasp-time offset to the palm midpoint, so the
  # trajectory is continuous at the grasp frame.
  pos[grasp:] = mid[grasp:] + (spawn - mid[grasp])
  # The palms keep sinking up to GRASP_Z_TOL after the grasp frame; a box reference
  # below the floor is unreachable, so hold it at rest height instead.
  pos[:, 2] = np.maximum(pos[:, 2], BOX_HALF_EXTENTS[2])
  yaw = np.full(num_frames, pelvis_yaw[grasp])
  yaw[grasp:] = pelvis_yaw[grasp:]
  quat = _yaw_quat(yaw)

  dt = 1.0 / FPS
  lin_vel = np.gradient(pos, dt, axis=0)
  ang_vel = np.zeros_like(pos)
  ang_vel[:, 2] = np.gradient(yaw, dt)  # yaw-only rotation: world z rate
  return {
    "box_pos_w": pos.astype(np.float32),
    "box_quat_w": quat.astype(np.float32),
    "box_lin_vel_w": lin_vel.astype(np.float32),
    "box_ang_vel_w": ang_vel.astype(np.float32),
  }


def _box_clearance(body_pos: np.ndarray, box: dict[str, np.ndarray]) -> np.ndarray:
  """Signed distance [F,30] of each body origin to the box surface (<0 = inside)."""
  yaw = _yaw_of(box["box_quat_w"])
  rel = body_pos - box["box_pos_w"][:, None, :]
  c, s = np.cos(-yaw)[:, None], np.sin(-yaw)[:, None]
  local = np.stack(
    [c * rel[..., 0] - s * rel[..., 1], s * rel[..., 0] + c * rel[..., 1], rel[..., 2]],
    axis=-1,
  )
  q = np.abs(local) - BOX_HALF_EXTENTS
  outside = np.linalg.norm(np.maximum(q, 0.0), axis=-1)
  inside = np.minimum(q.max(axis=-1), 0.0)
  return outside + inside


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  parser.add_argument("--input", required=True, type=Path)
  parser.add_argument("--output", type=Path, default=None)
  args = parser.parse_args()
  out = args.output or args.input.with_name(args.input.stem + "_box.npz")

  src = np.load(args.input)
  data = {k: src[k] for k in src.files}
  num_frames = data["joint_pos"].shape[0]

  body_pos, palm_pos = forward_kinematics(data)
  fk_err = np.abs(body_pos - data["body_pos_w"]).max()
  print(f"[{args.input.name}] frames={num_frames}  FK vs npz body_pos_w max err {fk_err * 1e3:.3f} mm")
  if fk_err > 1e-3:
    raise RuntimeError("npz body order/joint order does not match mjlab g1.xml")

  mid = palm_pos.mean(axis=1)
  grasp = find_grasp_frame(mid[:, 2])
  box = box_reference(palm_pos, data["body_quat_w"][:, 0], grasp)

  sep = np.linalg.norm(palm_pos[:, 0] - palm_pos[:, 1], axis=-1)
  hold = slice(grasp, num_frames)
  last_s = slice(max(num_frames - int(FPS), grasp), num_frames)
  print(f"grasp frame {grasp} (t = {grasp / FPS:.2f} s), palm-mid at grasp {np.round(mid[grasp], 3)}")
  print(f"palm separation at grasp {sep[grasp]:.3f} m; while holding (grasp..end) "
        f"mean {sep[hold].mean():.3f} min {sep[hold].min():.3f} max {sep[hold].max():.3f} m; "
        f"last 1 s mean {sep[last_s].mean():.3f} m  (box width {2 * BOX_HALF_EXTENTS[1]:.2f} m)")
  bz = box["box_pos_w"][:, 2]
  print(f"box ref height range [{bz.min():.3f}, {bz.max():.3f}] m, final {bz[-1]:.3f} m; "
        f"peak |v_z| {np.abs(box['box_lin_vel_w'][:, 2]).max():.2f} m/s")
  pelvis_x = data["body_pos_w"][grasp, 0, :2]
  print(f"box centre ahead of pelvis at grasp {np.linalg.norm(mid[grasp, :2] - pelvis_x):.3f} m")

  # Sanity: non-hand bodies should stay outside the box. Hands (wrist links) are
  # expected to squeeze it, since the palm sites sit on the hand-capsule axis.
  clear = _box_clearance(body_pos, box)
  names = [mujoco.MjModel.from_xml_path(str(G1_XML)).body(i + 1).name for i in range(30)]
  non_hand = [i for i, n in enumerate(names) if "wrist" not in n]
  worst = non_hand[int(np.argmin(clear[:, non_hand].min(axis=0)))]
  print(f"min clearance of non-hand body origins to box: {clear[:, non_hand].min():.3f} m ({names[worst]})")

  data.update(box)
  data["grasp_frame"] = np.array(grasp, dtype=np.int64)
  data["box_half_extents"] = BOX_HALF_EXTENTS.astype(np.float32)
  data["box_forward_offset"] = np.float32(BOX_FORWARD_M)
  np.savez(out, **data)
  print(f"wrote {out}")


if __name__ == "__main__":
  main()
