from __future__ import annotations

import time

import mujoco
import mujoco.viewer

from mjlab_g1.assets.g1_constants import (
  G1_INIT_KEYFRAME,
  get_g1_spec,
)
from mjlab_g1.assets.toaster_constants import (
  TOASTER_INIT_KEYFRAME,
  get_toaster_spec,
)


def _build_model() -> mujoco.MjModel:
    spec = mujoco.MjSpec()
    spec.worldbody.add_geom(
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        name="floor",
        size=[10.0, 10.0, 0.1],
        pos=[0.0, 0.0, 0.0],
        rgba=[0.3, 0.35, 0.4, 1.0],
    )

    robot_frame = spec.worldbody.add_frame(name="robot_frame")
    toaster_frame = spec.worldbody.add_frame(name="toaster_frame")
    spec.attach(get_g1_spec(), prefix="robot/", frame=robot_frame)
    spec.attach(get_toaster_spec(), prefix="toaster/", frame=toaster_frame)
    return spec.compile()


def _set_freejoint_pose(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    joint_name: str,
    pos: tuple[float, float, float],
    quat: tuple[float, float, float, float],
) -> None:
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if joint_id < 0:
        raise ValueError(f"Joint not found: {joint_name}")
    qpos_adr = model.jnt_qposadr[joint_id]
    data.qpos[qpos_adr : qpos_adr + 3] = pos
    data.qpos[qpos_adr + 3 : qpos_adr + 7] = quat


def main() -> None:
    model = _build_model()
    data = mujoco.MjData(model)

    _set_freejoint_pose(
        model,
        data,
        "robot/floating_base_joint",
        G1_INIT_KEYFRAME.pos,
        (1.0, 0.0, 0.0, 0.0),
    )
    _set_freejoint_pose(
        model,
        data,
        "toaster/object_freejoint",
        TOASTER_INIT_KEYFRAME.pos,
        TOASTER_INIT_KEYFRAME.rot,
    )

    mujoco.mj_forward(model, data)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.lookat[:] = [0.15, 0.0, 0.55]
        viewer.cam.distance = 2.2
        viewer.cam.azimuth = 145.0
        viewer.cam.elevation = -18.0

        while viewer.is_running():
            viewer.sync()
            time.sleep(0.01)


if __name__ == "__main__":
    main()
