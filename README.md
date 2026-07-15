# humanoid-amp-ppo

AMP-based humanoid RL training for tasks:

- `Mjlab-G1-Locomotion`
- `Mjlab-G1-DualArm`

An AMP discriminator is trained on human motion data to help the humanoid RL policy learn with fewer task-specific rewards while encouraging more human-like behavior.

## Locomotion policy:

![Locomotion policy](assets/locomotion.gif)

[MP4 version](assets/locomotion.mp4)

## Dual-arm policy:

![Dual-arm policy](assets/dual_arm.gif)

[MP4 version](assets/dual_arm.mp4)

### Dual-arm training (ResMimic-style residual learning)

The dual-arm task trains a residual policy on top of a **frozen** locomotion
actor (following [ResMimic](https://arxiv.org/abs/2510.05070)): the base
policy supplies balance and whole-body coordination, and the RL policy learns
task-specific corrections (`a = a_base + 0.3 * a_residual`). The base
checkpoint (`models/locomotion.pt`) is part of the task's default env config,
so training and play compose actions identically.

Supporting mechanisms:

- **Success-adaptive curricula** — the virtual object controller's assistance
  and the object's reset-height bootstrap decay only while the lift-success
  EMA stays above threshold, so difficulty self-regulates instead of following
  a blind step schedule.
- **Assistance-force penalty** — the policy is increasingly penalized for the
  virtual-PD force actually used, paying it to take over the lift itself.
- **AMP as a style regularizer** — the AMP reward coefficient is capped near
  the task reward's standing baseline so imitating the mocap never outpays
  completing the task.

Train:

```bash
MUJOCO_GL=egl PYTHONPATH=src uv run python train_dualarm_from_locomotion.py
```

## Codebase Structure

- `src/mjlab_g1/`
  - task configs, envs, wrappers, viewers
- `rsl_rl/`
  - RSL-RL 5.2 with migrated AMP-PPO code
- `dataset/`
  - AMP motion datasets

## Install

This repo is developed with `uv`.

Install `uv`, clone the repository together with the DeFM submodule, and install the project:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh

git clone --recurse-submodules https://github.com/shuayyy/humanoid-amp-ppo.git
cd humanoid-amp-ppo

uv sync
```

Installed packages:

* `mjlab-g1`
* `rsl-rl-lib` from `rsl_rl`
* `defm` from `third_party/defm`

`uv sync` installs DeFM and its runtime dependencies. Do not install DeFM with
`--no-deps`; dual-arm needs those dependencies for the depth encoder.

## Task IDs

Registered tasks:

- `Mjlab-G1-Locomotion`
- `Mjlab-G1-DualArm`

## Training

Locomotion:

```bash
MUJOCO_GL=egl PYTHONPATH=src uv run python src/mjlab_g1/scripts/train.py \
Mjlab-G1-Locomotion \
--video False
```

Dual arm:

```bash
MUJOCO_GL=egl PYTHONPATH=src uv run python src/mjlab_g1/scripts/train.py \
Mjlab-G1-DualArm \
--video False
```

Dual-arm uses a depth camera and frozen DeFM features. On smaller GPUs, reduce
the environment count, for example:

```bash
MUJOCO_GL=egl PYTHONPATH=src uv run python src/mjlab_g1/scripts/train.py \
Mjlab-G1-DualArm \
--env.scene.num_envs 64 \
--video False
```

## Play

Trained Policies:

- `models/locomotion.pt`
- `models/dual_arm.pt`

Locomotion:

```bash
PYTHONPATH=src uv run python src/mjlab_g1/scripts/play.py \
Mjlab-G1-Locomotion \
--checkpoint-file models/locomotion.pt
```

Dual arm:

```bash
PYTHONPATH=src uv run python src/mjlab_g1/scripts/play.py \
Mjlab-G1-DualArm \
--checkpoint-file models/dual_arm.pt
```

## AMP datasets

Current task-specific AMP dataset paths:

- locomotion: `dataset/locomotion`
- dualarm: `dataset/dualarm`
