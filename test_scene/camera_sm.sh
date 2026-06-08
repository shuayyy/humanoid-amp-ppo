SCRIPT_DIR=$(dirname $(realpath $0))

if [ -z "$1" ]; then
  echo "Usage: $0 path_to_policy.onnx"
  exit 1
fi

ckpt_path=$1

export MUJOCO_GL=glfw

uv run python test_scene/sim.py \
    --xml ${SCRIPT_DIR}/mjlab_scene.xml \
    --policy ${ckpt_path} \
    --device cuda \
    --policy_frequency 50 \
    --camera_name robot/realsense_d435_depth \
    --camera_width 256 \
    --camera_height 192 \
    --camera_fps 10 \
    --camera_depth
