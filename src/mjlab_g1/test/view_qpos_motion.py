from __future__ import annotations

import argparse
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np


def _default_model_xml() -> Path:
  return (
    Path(__file__).resolve().parents[1]
    / "assets"
    / "g1"
    / "g1_29dof_realsense.xml"
  )


def _build_model(model_xml: Path) -> mujoco.MjModel:
  xml_text = model_xml.read_text()
  # The 29-DoF XML in this repo points at ./meshes/, but the checked-in assets
  # live under g1/assets/.
  if model_xml.name == "g1_29dof_realsense.xml":
    meshdir = (model_xml.parent / "assets").as_posix()
    xml_text = xml_text.replace('meshdir="./meshes/"', f'meshdir="{meshdir}"')
  spec = mujoco.MjSpec.from_string(xml_text, assets={})
  spec.worldbody.add_geom(
    type=mujoco.mjtGeom.mjGEOM_PLANE,
    # The 29-DoF XML declares explicit foot-floor contact pairs, so the
    # added plane must be named "floor" for the model to compile.
    name="floor",
    size=[10.0, 10.0, 0.1],
    pos=[0.0, 0.0, 0.0],
    rgba=[0.3, 0.35, 0.4, 1.0],
  )
  return spec.compile()


def _load_motion_as_qpos(motion_file: Path, model: mujoco.MjModel) -> np.ndarray:
  frames = np.load(motion_file, allow_pickle=True)
  if frames.ndim != 2:
    raise ValueError(f"{motion_file} must be a 2D array, got shape {frames.shape}")

  nq = model.nq
  if frames.shape[1] == nq:
    qpos = frames.astype(np.float32, copy=False)
    print(f"[INFO] {motion_file.name}: using direct qpos format {qpos.shape}")
    return qpos

  # Legacy fallback for the 23-DoF robot model. The AMP motion files in
  # this repo are 36D direct qpos for the 29-DoF model above, so this path
  # should not be used for human_push_*.npy or dataset*_g1_qpos.npy.
  if frames.shape[1] == 36 and nq == 30:
    qpos = np.zeros((frames.shape[0], nq), dtype=np.float32)
    qpos[:, :7] = frames[:, :7]
    qpos[:, 7:] = frames[:, 13:36]
    print(
      f"[INFO] {motion_file.name}: converted AMP-style 36D frames to qpos {qpos.shape}"
    )
    return qpos

  raise ValueError(
    f"{motion_file} shape {frames.shape} is not compatible with model nq={nq}. "
    "Expected [T, nq] direct qpos or [T, 36] AMP-style frames."
  )


def _play_motions(
  model: mujoco.MjModel,
  motions: list[tuple[str, np.ndarray]],
  fps: float,
  loop: bool,
) -> None:
  data = mujoco.MjData(model)
  frame_dt = 1.0 / fps

  clip_idx = 0
  frame_idx = 0
  paused = False
  last_tick = time.perf_counter()
  viewer_ref: list[mujoco.viewer.Handle | None] = [None]

  print("[INFO] Controls:")
  print("  Space: pause/resume")
  print("  Right arrow: next frame")
  print("  Left arrow: previous frame")
  print("  N: next clip")
  print("  P: previous clip")

  def _update_camera() -> None:
    viewer = viewer_ref[0]
    if viewer is None:
      return
    root = data.qpos[:3]
    viewer.cam.lookat[:] = [root[0], root[1], max(0.7, root[2] - 0.1)]
    viewer.cam.distance = 3.5
    viewer.cam.azimuth = 140.0
    viewer.cam.elevation = -20.0

  def _set_frame(new_clip_idx: int, new_frame_idx: int) -> None:
    nonlocal clip_idx, frame_idx
    clip_idx = new_clip_idx % len(motions)
    clip_name, clip_frames = motions[clip_idx]
    frame_idx = int(np.clip(new_frame_idx, 0, clip_frames.shape[0] - 1))
    data.qpos[:] = clip_frames[frame_idx]
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    _update_camera()
    print(
      f"[INFO] clip={clip_name} frame={frame_idx + 1}/{clip_frames.shape[0]}",
      end="\r",
      flush=True,
    )

  def _key_callback(keycode: int) -> None:
    nonlocal paused
    if keycode == 32:  # space
      paused = not paused
      return
    if keycode == 262:  # right
      _set_frame(clip_idx, frame_idx + 1)
      return
    if keycode == 263:  # left
      _set_frame(clip_idx, frame_idx - 1)
      return
    if keycode in (78, 110):  # n/N
      _set_frame(clip_idx + 1, 0)
      return
    if keycode in (80, 112):  # p/P
      _set_frame(clip_idx - 1, 0)
      return

  _set_frame(0, 0)

  with mujoco.viewer.launch_passive(model, data, key_callback=_key_callback) as viewer:
    viewer_ref[0] = viewer
    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = False
    _update_camera()
    while viewer.is_running():
      now = time.perf_counter()
      if not paused and now - last_tick >= frame_dt:
        last_tick = now
        clip_name, clip_frames = motions[clip_idx]
        next_frame = frame_idx + 1
        if next_frame >= clip_frames.shape[0]:
          if clip_idx + 1 < len(motions):
            _set_frame(clip_idx + 1, 0)
          elif loop:
            _set_frame(0, 0)
          else:
            paused = True
        else:
          _set_frame(clip_idx, next_frame)

      viewer.sync()
      time.sleep(0.001)


def main() -> None:
  parser = argparse.ArgumentParser(
    description="Kinematic MuJoCo viewer for qpos/AMP-style motion files."
  )
  parser.add_argument(
    "motion_files",
    nargs="+",
    help="One or more .npy motion files to play.",
  )
  parser.add_argument(
    "--model-xml",
    type=Path,
    default=_default_model_xml(),
    help="Robot XML used for visualization.",
  )
  parser.add_argument(
    "--fps",
    type=float,
    default=50.0,
    help="Playback frame rate.",
  )
  parser.add_argument(
    "--loop",
    action="store_true",
    help="Loop back to the first clip after the last clip ends.",
  )
  args = parser.parse_args()

  model = _build_model(args.model_xml)
  motions = []
  for motion_path_str in args.motion_files:
    motion_path = Path(motion_path_str).expanduser().resolve()
    motions.append((motion_path.name, _load_motion_as_qpos(motion_path, model)))

  _play_motions(model, motions, fps=args.fps, loop=args.loop)


if __name__ == "__main__":
  main()
