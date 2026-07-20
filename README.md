# humanoid-amp-ppo

AMP-based reinforcement learning for the Unitree G1 humanoid, with two tasks:

- `Mjlab-G1-Locomotion` — velocity-tracking locomotion
- `Mjlab-G1-DualArm` — bimanual lift-in-place of a box

An AMP discriminator trained on human motion capture supplies the style signal,
so the policy needs fewer hand-written task rewards and moves more like the
reference motion.

## Locomotion policy

![Locomotion policy](assets/locomotion.gif)

[MP4 version](assets/locomotion.mp4)

## Dual-arm policy

![Dual-arm policy](assets/dual_arm.gif)

[MP4 version](assets/dual_arm.mp4)

## Install

This repo is developed with `uv`.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh

git clone --recurse-submodules https://github.com/shuayyy/humanoid-amp-ppo.git
cd humanoid-amp-ppo

uv sync
```

That installs `mjlab-g1`, `rsl-rl-lib` from `rsl_rl/`, and `defm` from
`third_party/defm`. Do not install DeFM with `--no-deps`; the depth encoder
needs its runtime dependencies even though depth is off by default.

## Repository layout

| Path | Contents |
| --- | --- |
| `src/mjlab_g1/envs/` | Task environments, including the dual-arm RL env |
| `src/mjlab_g1/tasks/` | Task configs, MDP terms (rewards, observations), registry |
| `src/mjlab_g1/assets/` | G1 and object models |
| `rsl_rl/` | RSL-RL 5.2 with AMP-PPO migrated in |
| `dataset/` | AMP motion clips (`locomotion/`, `dualarm/`) |
| `scripts/` | Cluster job scripts and evaluation tooling |
| `models/` | Released policies and the frozen locomotion base |

## Training

The dual-arm entry point composes the frozen base policy and the residual, so
training and play build actions identically:

```bash
PYTHONPATH=src uv run python train_dualarm_from_locomotion.py
```

Either task can also be trained through the generic script:

```bash
PYTHONPATH=src uv run python src/mjlab_g1/scripts/train.py \
  Mjlab-G1-Locomotion --video False
```

Reduce `--env.scene.num_envs` on smaller GPUs.

### On a Slurm cluster

`scripts/train_dualarm.sbatch` handles module loading, `uv sync`, and
auto-resume from the newest checkpoint of `EXP_NAME`:

```bash
NUM_ENVS=4096 MAX_ITERS=20000 EXP_NAME=g1_dualarm_v17 RUN_NAME=scratch_v17 \
  sbatch scripts/train_dualarm.sbatch
```

Set `FRESH=1` to force a clean run instead of resuming. Note `MAX_ITERS` is
*additional* iterations when resuming, not an absolute ceiling.

To start a run from a different experiment's checkpoint — the usual way to
begin a fine-tune, including behind a still-running predecessor with
`--dependency=afterany` — use the seeded launcher, which resolves the source
checkpoint at run time:

```bash
EXP_NAME=g1_dualarm_v16 SEED_FROM=logs/rsl_rl/g1_dualarm_v15b \
  NUM_ENVS=4096 MAX_ITERS=8000 RUN_NAME=squat_v16 \
  sbatch scripts/train_dualarm_seeded.sbatch
```

## Play and evaluate

```bash
PYTHONPATH=src uv run python src/mjlab_g1/scripts/play.py \
  Mjlab-G1-DualArm --checkpoint-file models/dual_arm.pt
```

Rendering never uses OpenGL: it goes through mjlab's `mujoco_warp` ray-traced
camera with `MUJOCO_GL=disable`, because compute nodes commonly ship no
usable libEGL or libOSMesa.

```bash
CHECKPOINT=logs/rsl_rl/<exp>/<run>/model_<n>.pt OUT_DIR=models/videos/<name> \
  sbatch scripts/render_video.sbatch

sbatch scripts/render_motion_prior.sbatch   # kinematic playback of the AMP clips
```

Two evaluation tools are worth running before drawing conclusions from a
training curve:

- `scripts/eval_posture_stats.py --checkpoint-file <ckpt>` rolls a checkpoint
  out headlessly and prints posture statistics split by phase (reach, grab,
  hold). Reward curves aggregate over whole episodes and hide the moments that
  matter; these distributions are also what gate thresholds should be set from.
- `scripts/smoke_dualarm_env.py` builds the env, resolves every reward term and
  steps it, so sign and magnitude errors surface in minutes rather than after a
  long job. Run it after any reward change.

## Dual-arm task design

The dual-arm policy is a residual on a **frozen** locomotion actor, following
[ResMimic](https://arxiv.org/abs/2510.05070). The base supplies balance and
whole-body coordination; the RL policy learns task corrections:

```
joint_target = default_pos + action_scale * (base_action + residual_scale * policy_action)
```

`residual_scale` is per joint group (`residual_scale_legs`,
`residual_scale_waist`, `residual_scale_arms`). **This determines which
postures the policy can command at all.** With `action_scale = 0.5`, default
joint positions of zero and a large policy output of 3.0, the largest knee
angle reachable is about 0.4 rad at a leg scale of 0.1, but 2.5 rad at 1.5.
Since the reference squat bends the knees 1.9–2.2 rad, a low leg scale puts the
target motion outside the action space entirely, and no reward can then produce
it. Compute this bound before adding reward terms for a posture.

Changing `residual_scale` also changes what a stored action means, so a
checkpoint cannot simply be resumed across the change: the actor's rows for
those joints and their exploration std need rescaling by the inverse ratio, and
even then stale optimizer moments tend to destabilize training. Prefer a fresh
run.

Supporting mechanisms:

- **Contact-triggered lift** — the object's reference trajectory rises only
  after both grasp-marker contacts hold for a settle window, so an episode with
  a late grasp still gets a full attempt.
- **Success-adaptive curricula** — virtual-PD assistance and the object's
  reset-height bootstrap decay only while the lift-success EMA holds up, and
  recover if it collapses. Curriculum state lives in the env, not the
  checkpoint, so a resumed run re-weans assistance from scratch.
- **Assistance-force penalty** — the policy pays for the virtual-PD force
  actually used, which pays it to take over the lift itself.
- **Reference-state initialization** — `rsi_fraction` of resets start inside a
  motion-capture squat frame with the box at the hands, so postures that
  exploration will not discover on its own still enter the state distribution.
- **Phase-scheduled AMP** — pre-lift environments use
  `amp_prelift_reward_coef` while post-lift ones follow the annealed schedule.
  Style matters most during the reach, where task shaping is sparsest.

### A note on reward terms

Episode reward logs are normalized: `Episode_Reward/*` is the episode sum
divided by `episode_length_s`, so a term's per-episode cost is the logged value
times that duration.

Terms that charge once per episode (the `_episode_peak` helper in
`tasks/dualarm/mdp/rewards.py`) exist because per-timestep penalties are
speed-discounted — a policy can cross a penalized posture quickly and pay
almost nothing, which teaches it to hurry rather than to avoid the posture.
When reading a one-shot charge's logged mean, check it against the success rate
first: the charge only fires on episodes that reach the lift, so a drop in the
mean can mean fewer successful episodes rather than better posture.

Several reach-phase posture penalties remain defined in `rewards.py` but are
not in the active reward set. They were built to price a descent the policy had
no choice but to make while the squat was unreachable. Re-add them individually
if a specific failure reappears, rather than as a block.

## AMP datasets

- locomotion: `dataset/locomotion`
- dualarm: `dataset/dualarm`

Clips are `[frames, 36]` arrays of MuJoCo `qpos`: root position and quaternion
followed by the 29 joints in model order. `test/play_dataset.py` replays them.
