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
    / "asset_zoo"
    / "robots"
    / "skateboard"
    / "xmls"
    / "g1_29dof_realsense.xml"
  )


def _build_model(model_xml: Path) -> mujoco.MjModel:
  xml_text = model_xml.read_text()
  meshdir = (model_xml.parent / "assets").as_posix()
  xml_text = xml_text.replace('meshdir="./meshes/"', f'meshdir="{meshdir}"')

  spec = mujoco.MjSpec.from_string(xml_text, assets={})
  spec.worldbody.add_geom(
    type=mujoco.mjtGeom.mjGEOM_PLANE,
    name="floor",
    size=[10.0, 10.0, 0.1],
    pos=[0.0, 0.0, 0.0],
    rgba=[0.3, 0.35, 0.4, 1.0],
  )
  return spec.compile()


def _load_qpos_29dof(motion_file: Path, model: mujoco.MjModel) -> np.ndarray:
  frames = np.load(motion_file, allow_pickle=True)
  if frames.ndim != 2:
    raise ValueError(f"{motion_file} must be 2D, got {frames.shape}")
  if frames.shape[1] != model.nq:
    raise ValueError(
      f"{motion_file} shape {frames.shape} does not match {model.nq} qpos. "
      "Expected 36 = 7 root + 29 joint qpos."
    )

  qpos = frames.astype(np.float32, copy=False)
  print(f"[INFO] {motion_file.name}: direct 29-DoF qpos {qpos.shape}")
  return qpos


def _play(model: mujoco.MjModel, clips: list[tuple[str, np.ndarray]], fps: float, loop: bool) -> None:
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
    viewer.cam.lookat[:] = [root[0], root[1], max(0.75, root[2] - 0.05)]
    viewer.cam.distance = 3.2
    viewer.cam.azimuth = 145.0
    viewer.cam.elevation = -18.0

  def _set_frame(new_clip_idx: int, new_frame_idx: int) -> None:
    nonlocal clip_idx, frame_idx
    clip_idx = new_clip_idx % len(clips)
    clip_name, frames = clips[clip_idx]
    frame_idx = int(np.clip(new_frame_idx, 0, frames.shape[0] - 1))
    data.qpos[:] = frames[frame_idx]
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    _update_camera()
    print(
      f"[INFO] clip={clip_name} frame={frame_idx + 1}/{frames.shape[0]}",
      end="\r",
      flush=True,
    )

  def _key_callback(keycode: int) -> None:
    nonlocal paused
    if keycode == 32:
      paused = not paused
      return
    if keycode == 262:
      _set_frame(clip_idx, frame_idx + 1)
      return
    if keycode == 263:
      _set_frame(clip_idx, frame_idx - 1)
      return
    if keycode in (78, 110):
      _set_frame(clip_idx + 1, 0)
      return
    if keycode in (80, 112):
      _set_frame(clip_idx - 1, 0)
      return

  _set_frame(0, 0)

  with mujoco.viewer.launch_passive(model, data, key_callback=_key_callback) as viewer:
    viewer_ref[0] = viewer
    _update_camera()
    while viewer.is_running():
      now = time.perf_counter()
      if not paused and now - last_tick >= frame_dt:
        last_tick = now
        _, frames = clips[clip_idx]
        next_frame = frame_idx + 1
        if next_frame >= frames.shape[0]:
          if clip_idx + 1 < len(clips):
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
    description="View 29-DoF G1 qpos motion files: 36 = 7 root + 29 joint qpos."
  )
  parser.add_argument("motion_files", nargs="+", help="One or more .npy motion files.")
  parser.add_argument("--fps", type=float, default=50.0, help="Playback frame rate.")
  parser.add_argument("--loop", action="store_true", help="Loop clips.")
  args = parser.parse_args()

  model = _build_model(_default_model_xml())
  if model.nq != 36:
    raise ValueError(f"Expected 29-DoF model with nq=36, got nq={model.nq}")

  clips = []
  for motion_path_str in args.motion_files:
    motion_path = Path(motion_path_str).expanduser().resolve()
    clips.append((motion_path.name, _load_qpos_29dof(motion_path, model)))

  _play(model, clips, fps=args.fps, loop=args.loop)


if __name__ == "__main__":
  main()
