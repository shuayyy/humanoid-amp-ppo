# AMP-PPO: Human-Prior RL Training

AMP-based humanoid RL training for the following tasks:

- `Mjlab-G1-Locomotion`
- `Mjlab-G1-DualArm`

This repo includes a discriminator trained on human motion data retargeted to the Unitree G1 humanoid, and uses it to train RL policies with fewer task-specific rewards while encouraging more human-like behavior.

## Codebase Structure

- `src/mjlab_g1/`
  - task configs, envs, wrappers, viewers
- `rsl_rl/`
  - PPO and AMP-PPO code
- `dataset/`
  - AMP motion datasets

## Install

This repo is developed with `uv`.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
git clone https://github.com/TeleHuman/humanoid_skateboarding.git
cd humanoid_skateboarding
uv sync
uv pip install -e .
```

Installed package:

- `mjlab-g1`

## Task IDs

Registered tasks:

- `Mjlab-G1-Locomotion`
- `Mjlab-G1-DualArm`

## Training

Locomotion:

```bash
PYTHONPATH=src uv run python src/mjlab_g1/scripts/train.py Mjlab-G1-Locomotion
```

Dual arm:

```bash
PYTHONPATH=src uv run python src/mjlab_g1/scripts/train.py Mjlab-G1-DualArm
```

## Play

Locomotion:

```bash
PYTHONPATH=src uv run python src/mjlab_g1/scripts/play.py \
Mjlab-G1-Locomotion \
--checkpoint-file /path/to/model.pt
```

Dual arm:

```bash
PYTHONPATH=src uv run python src/mjlab_g1/scripts/play.py \
Mjlab-G1-DualArm \
--checkpoint-file /path/to/model.pt
```

## AMP datasets

Current task-specific AMP dataset paths:

- locomotion: `dataset/locomotion`
- dualarm: `dataset/dualarm`

## Utilities

View the robot and toaster at their init poses:

```bash
PYTHONPATH=src uv run python src/mjlab_g1/test/view_g1_toaster_init.py
```

View qpos motion clips:

```bash
PYTHONPATH=src uv run python src/mjlab_g1/test/view_qpos_29dof.py dataset/some_clip.npy
```

## Notes

- The toaster init pose is defined in:
  - `src/mjlab_g1/assets/toaster_constants.py`
- Dual-arm uses a `place_pos` command target and AMP.
- Locomotion and dual-arm are intentionally separate task trees.
