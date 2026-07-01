from __future__ import annotations

"""
Replay AMP dataset poses on a plane with no physics stepping.

Locomotion dataset:

    python test/play_dataset.py --dataset locomotion

Dual-arm dataset:

    python test/play_dataset.py --dataset dualarm
"""

import argparse
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
VENV_SITE_PACKAGES = (
    REPO_ROOT
    / ".venv"
    / "lib"
    / f"python{sys.version_info.major}.{sys.version_info.minor}"
    / "site-packages"
)
for path in (SRC_ROOT, VENV_SITE_PACKAGES):
    if path.exists():
        sys.path.insert(0, str(path))

import mujoco
import mujoco.viewer
import numpy as np
from mjlab.scene import Scene, SceneCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab_g1.assets.g1_constants import get_g1_29dof_robot_cfg


DATASET_DIRS = {
    "locomotion": REPO_ROOT / "dataset" / "locomotion",
    "dualarm": REPO_ROOT / "dataset" / "dualarm",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay locomotion or dual-arm AMP dataset frames at 45 FPS."
    )
    parser.add_argument(
        "--dataset",
        choices=("locomotion", "dualarm", "all"),
        required=True,
        help="Dataset folder to replay.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=45.0,
        help="Playback FPS. Defaults to 45.",
    )
    parser.add_argument(
        "--motion",
        type=Path,
        default=None,
        help="Optional specific .npy motion file to replay.",
    )
    parser.add_argument(
        "--no-viewer",
        action="store_true",
        help="Load frames and call mj_forward without opening the viewer.",
    )
    return parser.parse_args()


def build_plane_robot_model() -> mujoco.MjModel:
    scene_cfg = SceneCfg(
        terrain=TerrainEntityCfg(
            terrain_type="plane",
            terrain_generator=None,
        ),
        entities={"robot": get_g1_29dof_robot_cfg()},
        num_envs=1,
        extent=2.0,
    )
    scene = Scene(scene_cfg, device="cpu")
    model = scene.compile()
    model.opt.timestep = 1.0 / 45.0
    return model


def dataset_paths(dataset: str, motion: Path | None) -> list[Path]:
    if motion is not None:
        path = motion if motion.is_absolute() else REPO_ROOT / motion
        if not path.exists():
            raise FileNotFoundError(path)
        return [path]

    names = DATASET_DIRS.keys() if dataset == "all" else (dataset,)
    paths: list[Path] = []
    for name in names:
        paths.extend(sorted(DATASET_DIRS[name].glob("*.npy")))
    if not paths:
        raise FileNotFoundError(f"No .npy files found for dataset {dataset!r}.")
    return paths


def load_motion(path: Path) -> np.ndarray:
    frames = np.load(path, allow_pickle=True)
    if frames.ndim != 2 or frames.shape[1] < 36:
        raise ValueError(f"Expected {path} to have shape [frames, >=36], got {frames.shape}.")
    return np.asarray(frames[:, :36], dtype=np.float64)


def set_pose(model: mujoco.MjModel, data: mujoco.MjData, frame: np.ndarray) -> None:
    if model.nq != 36:
        raise ValueError(f"Expected model.nq == 36 for G1 29-DoF poses, got {model.nq}.")
    data.qpos[:] = frame
    data.qvel[:] = 0.0
    data.ctrl[:] = 0.0
    mujoco.mj_forward(model, data)


def print_motion_stats(path: Path, frames: np.ndarray, fps: float) -> None:
    num_frames = frames.shape[0]
    duration_s = (num_frames - 1) / fps if num_frames > 1 else 0.0
    rel_path = path.relative_to(REPO_ROOT)
    print(f"{rel_path}")
    print(f"  frames: {num_frames}")
    print(f"  fps: {fps:g}")
    print(f"  time: 0.000s -> {duration_s:.3f}s")


def replay_motion(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    path: Path,
    frames: np.ndarray,
    fps: float,
    viewer: mujoco.viewer.Handle | None,
) -> None:
    dt = 1.0 / fps
    pelvis_id = model.body("robot/pelvis").id
    for frame_idx, frame in enumerate(frames):
        start = time.perf_counter()
        set_pose(model, data, frame)

        if viewer is not None:
            viewer.cam.lookat[:] = data.xpos[pelvis_id]
            viewer.sync()
            if not viewer.is_running():
                return

        elapsed = time.perf_counter() - start
        sleep_s = dt - elapsed
        if viewer is not None and sleep_s > 0.0:
            time.sleep(sleep_s)

    print(f"  played: {path.name}")


def main() -> None:
    args = parse_args()
    if args.fps <= 0.0:
        raise ValueError("--fps must be positive.")

    paths = dataset_paths(args.dataset, args.motion)
    model = build_plane_robot_model()
    data = mujoco.MjData(model)

    all_frames = [(path, load_motion(path)) for path in paths]
    total_frames = sum(frames.shape[0] for _, frames in all_frames)
    total_duration_s = sum(
        (frames.shape[0] - 1) / args.fps if frames.shape[0] > 1 else 0.0
        for _, frames in all_frames
    )

    print("Dataset replay")
    print("mode: kinematic qpos replay; no physics stepping")
    print(f"files: {len(all_frames)}")
    print(f"total frames: {total_frames}")
    print(f"total time @ {args.fps:g} fps: {total_duration_s:.3f}s")
    for path, frames in all_frames:
        print_motion_stats(path, frames, args.fps)

    viewer = None
    if not args.no_viewer:
        viewer = mujoco.viewer.launch_passive(
            model,
            data,
            show_left_ui=False,
            show_right_ui=False,
        )
        viewer.cam.distance = 4.0
        viewer.cam.azimuth = 210.0
        viewer.cam.elevation = -10.0

    try:
        for path, frames in all_frames:
            if viewer is not None and not viewer.is_running():
                break
            replay_motion(model, data, path, frames, args.fps, viewer)
    finally:
        if viewer is not None:
            viewer.close()


if __name__ == "__main__":
    main()
