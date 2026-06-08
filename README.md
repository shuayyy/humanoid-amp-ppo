## Overview

[![method](media/method.jpg "method")]()


This repository currently contains the MJLab-based whole-body grasp workflow for Unitree G1. The active task in this repo is WB grasp with a toaster object, along with the RL training stack, motion utilities, and MuJoCo viewers used to debug and evaluate it.

This repository contains:
- The mjlab training framework ([`src/mjlab_husky`](src/mjlab_husky))
- Customized RL implementations ([`rsl_rl/`](rsl_rl/))
- Motion data for AMP and trajectory planning ([`dataset/`](dataset/))
- Lightweight MuJoCo evaluation scripts ([`test_scene/`](test_scene/))
- Tested checkpoints ([`ckpts/`](ckpts/))

## Install
This code has been tested on Ubuntu 22.04 with CUDA 13.0.
To install this repository, please follow these steps:

1. **Install the [`uv`](https://docs.astral.sh/uv/getting-started/installation/#installation-methods) package manager**  (if you don't have it yet):

   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. **Clone the repository**:

   ```bash
   git clone https://github.com/TeleHuman/humanoid_skateboarding.git
   cd humanoid_skateboarding
   ```

3. **Sync dependencies**:

   ```bash
   uv sync
   uv pip install -e .
   ```

## WB Grasp Workflow

### Visualize markers

```bash
PYTHONPATH=src uv run python scripts/vis_markers.py
```

### View 29-DoF qpos motions

```bash
MUJOCO_GL=glfw PYTHONPATH=src uv run python src/mjlab_husky/test/view_qpos_29dof.py \
dataset/human_push_1.npy \
dataset/dataset_g1_qpos.npy \
--fps 50 \
--loop
```

### Compile checks

```bash
python -m py_compile \
src/mjlab_husky/tasks/wb_grasp/mdp/rewards.py \
src/mjlab_husky/tasks/wb_grasp/wb_grasp_env_cfg.py \
src/mjlab_husky/envs/g1_wb_grasp_rl_env.py \
src/mjlab_husky/tasks/wb_grasp/config/g1/env_cfgs.py \
src/mjlab_husky/asset_zoo/robots/skateboard/g1_constants.py
```

### Quick env reset / step test

```bash
PYTHONPATH=src uv run python - <<'EOF'
import torch
from mjlab_husky.tasks.wb_grasp.config.g1.env_cfgs import unitree_g1_wb_grasp_env_cfg
from mjlab_husky.envs.g1_wb_grasp_rl_env import G1GraspManagerBasedRlEnv

cfg = unitree_g1_wb_grasp_env_cfg()
env = G1GraspManagerBasedRlEnv(cfg=cfg, device="cuda:0")

obs, extras = env.reset()
action = torch.zeros((env.num_envs, env.action_manager.total_action_dim), device=env.device)
out = env.step(action)

print("env step passed")
print("reward:", out[1])
env.close()
EOF
```

### Train

```bash
wandb login
PYTHONPATH=src uv run python src/mjlab_husky/scripts/train.py Mjlab-WB-Grasp-Unitree-G1
```

### Play a checkpoint

```bash
MUJOCO_GL=glfw PYTHONPATH=src uv run python src/mjlab_husky/scripts/play.py \
Mjlab-WB-Grasp-Unitree-G1 \
--checkpoint-file /path/to/model.pt \
--viewer native
```

### Play a checkpoint and save video

```bash
MUJOCO_GL=glfw PYTHONPATH=src uv run python src/mjlab_husky/scripts/play.py \
Mjlab-WB-Grasp-Unitree-G1 \
--checkpoint-file /path/to/model.pt \
--viewer native \
--video True
```

## Citation

If you are using the original HUSKY skateboarding work, cite:

```bibtex
@article{han2026husky,
    title={HUSKY: Humanoid Skateboarding System via Physics-Aware Whole-Body Control},
    author={Jinrui Han and Dewei Wang and Chenyun Zhang and Xinzhe Liu and Ping Luo and Chenjia Bai and Xuelong Li},
    journal={arXiv preprint arXiv:2602.03205},
    year={2026}
  }
```

## License

This codebase is under [CC BY-NC 4.0 license](https://creativecommons.org/licenses/by-nc/4.0/deed.en). You may not use the material for commercial purposes, e.g., to make demos to advertise your commercial products.

## Acknowledgements
- [mjlab](https://github.com/mujocolab/mjlab): Our training framework is based on `mjlab` by MuJoCo Lab.
- [rsl_rl](https://github.com/leggedrobotics/rsl_rl): The reinforcement learning algorithm is built upon the `rsl_rl` library.
- [mujoco_warp](https://github.com/google-deepmind/mujoco_warp.git): GPU-accelerated interface for rendering and physics simulation.
- [mujoco](https://github.com/google-deepmind/mujoco.git): High-fidelity rigid-body physics engine.
- [AMP](https://github.com/xbpeng/MimicKit): Used for motion-prior related components in this codebase.
