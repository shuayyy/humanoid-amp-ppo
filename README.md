<h1 align="center">humanoid-amp-ppo</h1>

<p align="center">
  <a href="#installation">Installation</a> &nbsp;|&nbsp;
  <a href="#quickstart">Quickstart</a> &nbsp;|&nbsp;
  <a href="#tasks">Tasks</a> &nbsp;|&nbsp;
  <a href="#dual-arm-task-design">Design Notes</a> &nbsp;|&nbsp;
  <a href="#todo">TODO</a> &nbsp;|&nbsp;
  <a href="#troubleshooting">Troubleshooting</a> &nbsp;|&nbsp;
  <a href="#citation">Citation</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.12%20%7C%203.13-blue.svg" alt="Python 3.12 | 3.13">
  <img src="https://img.shields.io/badge/license-Apache--2.0-green.svg" alt="Apache 2.0">
  <img src="https://img.shields.io/badge/simulator-MuJoCo%20%2F%20mjlab-orange.svg" alt="MuJoCo / mjlab">
</p>

<p align="center">
  <img width="45%" src="assets/locomotion.gif" alt="Locomotion policy">
  <img width="45%" src="assets/dual_arm_v5.gif" alt="Dual-arm policy">
</p>
<p align="center">
  <sub>
    Locomotion (<a href="assets/locomotion.mp4">mp4</a>) &nbsp;·&nbsp;
    Dual-arm lift (<a href="assets/dual_arm_v5.mp4">mp4</a>)
  </sub>
</p>

-------

**humanoid-amp-ppo** is a reinforcement learning framework for the Unitree G1
humanoid, built on [mjlab](https://github.com/mujocolab/mjlab) and MuJoCo. It
pairs PPO with an [Adversarial Motion Prior](https://arxiv.org/abs/2104.02180)
discriminator trained on human motion capture, so policies need fewer
hand-written task rewards and move more like the reference motion.

The repository covers two tasks: velocity-tracking **locomotion**, and a
bimanual **dual-arm** lift, where a residual policy learns manipulation on top
of a frozen locomotion base following
[ResMimic](https://arxiv.org/abs/2510.05070).

Features:

- **AMP-PPO** — RSL-RL 5.2 with an AMP discriminator, annealed style rewards, and per-phase reward coefficients.
- **Residual learning** — a frozen locomotion actor supplies balance while the task policy learns corrections, with per-joint-group authority.
- **Success-adaptive curricula** — assistance and task difficulty decay only while the policy demonstrably succeeds, and recover when it does not.
- **Reference-state initialization** — motion-capture frames seed a fraction of episodes so postures that exploration will not find still enter the state distribution.
- **Headless evaluation** — ray-traced rendering and phase-split posture statistics that run as cluster jobs, with no OpenGL dependency.

-------

## Installation

This repo is developed with [uv](https://docs.astral.sh/uv/).

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh

git clone --recurse-submodules https://github.com/shuayyy/humanoid-amp-ppo.git
cd humanoid-amp-ppo

uv sync
```

This installs `mjlab-g1`, `rsl-rl-lib` from `rsl_rl/`, and `defm` from
`third_party/defm`. Do not install DeFM with `--no-deps`; the depth encoder
needs its runtime dependencies even though depth is disabled by default.

Requires Python 3.12 or 3.13 and a CUDA-capable GPU.

-------

## Quickstart

Train the dual-arm task. This entry point composes the frozen base policy and
the residual, so training and play build actions identically:

```bash
PYTHONPATH=src uv run python train_dualarm_from_locomotion.py
```

Play a released policy:

```bash
PYTHONPATH=src uv run python src/mjlab_g1/scripts/play.py \
  Mjlab-G1-DualArm --checkpoint-file models/dual_arm.pt
```

Either task can also go through the generic training script. Reduce
`--env.scene.num_envs` on smaller GPUs:

```bash
PYTHONPATH=src uv run python src/mjlab_g1/scripts/train.py \
  Mjlab-G1-Locomotion --video False
```

-------

## Tasks

| Task ID | Description | Policy | AMP clips |
| --- | --- | --- | --- |
| `Mjlab-G1-Locomotion` | Velocity-tracking locomotion | `models/locomotion.pt` | `dataset/locomotion` |
| `Mjlab-G1-DualArm` | Bimanual lift-in-place of a box | `models/dual_arm.pt` | `dataset/dualarm` |

Motion clips are `[frames, 36]` arrays of MuJoCo `qpos`: root position and
quaternion followed by the 29 joints in model order. Replay them with
`test/play_dataset.py`.

-------

## Repository layout

| Path | Contents |
| --- | --- |
| `src/mjlab_g1/envs/` | Task environments, including the dual-arm RL env |
| `src/mjlab_g1/tasks/` | Task configs, MDP terms (rewards, observations), registry |
| `src/mjlab_g1/assets/` | G1 and object models |
| `rsl_rl/` | RSL-RL 5.2 with AMP-PPO migrated in |
| `dataset/` | AMP motion clips |
| `scripts/` | Cluster job scripts and evaluation tooling |
| `models/` | Released policies and the frozen locomotion base |

-------

## Training on a Slurm cluster

`scripts/train_dualarm.sbatch` handles module loading, `uv sync`, and
auto-resume from the newest checkpoint of `EXP_NAME`:

```bash
NUM_ENVS=4096 MAX_ITERS=20000 EXP_NAME=g1_dualarm_v17 RUN_NAME=scratch_v17 \
  sbatch scripts/train_dualarm.sbatch
```

Set `FRESH=1` to force a clean run instead of resuming. Note that `MAX_ITERS`
is *additional* iterations when resuming, not an absolute ceiling.

To seed a run from another experiment's checkpoint — the usual way to start a
fine-tune, including behind a still-running predecessor with
`--dependency=afterany` — use the seeded launcher, which resolves the source
checkpoint at run time rather than at submit time:

```bash
EXP_NAME=g1_dualarm_v16 SEED_FROM=logs/rsl_rl/g1_dualarm_v15b \
  NUM_ENVS=4096 MAX_ITERS=8000 RUN_NAME=squat_v16 \
  sbatch scripts/train_dualarm_seeded.sbatch
```

-------

## Evaluation

Rendering never uses OpenGL. It goes through mjlab's `mujoco_warp` ray-traced
camera with `MUJOCO_GL=disable`, because compute nodes commonly ship no usable
libEGL or libOSMesa:

```bash
CHECKPOINT=logs/rsl_rl/<exp>/<run>/model_<n>.pt OUT_DIR=models/videos/<name> \
  sbatch scripts/render_video.sbatch

sbatch scripts/render_motion_prior.sbatch   # kinematic playback of the AMP clips
```

Two tools are worth running before drawing conclusions from a training curve:

- **`scripts/eval_posture_stats.py --checkpoint-file <ckpt>`** rolls a
  checkpoint out headlessly and prints posture statistics split by phase
  (reach, grab, hold). Reward curves aggregate over whole episodes and hide the
  moments that decide behavior; these distributions are also what gate
  thresholds should be set from, rather than visual estimates.
- **`scripts/smoke_dualarm_env.py`** builds the env, resolves every reward term
  and steps it, so sign and magnitude errors surface in minutes instead of
  after a long job. Run it after any reward change.

Comparing a policy against the motion prior it was trained on is the most
direct read on style. `assets/prior_vs_policy_v5.png` is an example of that
comparison for the dual-arm carry.

-------

## Dual-arm task design

The dual-arm policy is a residual on a **frozen** locomotion actor. The base
supplies balance and whole-body coordination; the RL policy learns task
corrections.

### Reachability comes before reward

```
joint_target = default_pos + action_scale * (base_action + residual_scale * policy_action)
```

`residual_scale` is set per joint group (`residual_scale_legs`,
`residual_scale_waist`, `residual_scale_arms`), and it determines **which
postures the policy can command at all**. With `action_scale = 0.5`, default
joint positions of zero and a large policy output of 3.0, the largest knee
angle reachable is about 0.4 rad at a leg scale of 0.1, but 2.5 rad at 1.5.
Since the reference squat bends the knees 1.9–2.2 rad, a low leg scale places
the target motion outside the action space entirely, and no reward can then
produce it — the policy is not declining to squat, it is unable to.

Compute this bound before adding reward terms for a posture.

Changing `residual_scale` also changes what a stored action means, so a
checkpoint cannot simply be resumed across the change. The actor's rows for
those joints and their exploration std need rescaling by the inverse ratio,
and even then stale optimizer moments tend to destabilize training. Prefer a
fresh run.

### Supporting mechanisms

- **Contact-triggered lift** — the object's reference trajectory rises only after both grasp-marker contacts hold for a settle window, so an episode with a late grasp still gets a full attempt.
- **Success-adaptive curricula** — virtual-PD assistance and the object's reset-height bootstrap decay only while the lift-success EMA holds up, and recover if it collapses. Curriculum state lives in the env, not the checkpoint, so a resumed run re-weans assistance from scratch.
- **Assistance-force penalty** — the policy pays for the virtual-PD force actually used, which pays it to take over the lift itself.
- **Reference-state initialization** — `rsi_fraction` of resets start inside a motion-capture squat frame with the box at the hands.
- **Phase-scheduled AMP** — pre-lift environments use `amp_prelift_reward_coef` while post-lift ones follow the annealed schedule.

### Reading reward terms

`Episode_Reward/*` is the episode sum divided by `episode_length_s`, so a
term's per-episode cost is the logged value times that duration.

Terms that charge once per episode (the `_episode_peak` helper in
`tasks/dualarm/mdp/rewards.py`) exist because per-timestep penalties are
speed-discounted: a policy can cross a penalized posture quickly and pay almost
nothing, which teaches it to hurry rather than to avoid the posture. When
reading a one-shot charge's logged mean, check it against the success rate
first — the charge only fires on episodes that reach the lift, so a drop in the
mean can reflect fewer successful episodes rather than better posture.

Several reach-phase posture penalties remain defined in `rewards.py` but are
not in the active reward set. They were built to price a descent the policy had
no choice but to make while the squat was unreachable. Re-add them individually
if a specific failure reappears, rather than as a block.

-------

## TODO

Task and training:

- [ ] **Squat descent.** The dual-arm policy reaches the box by folding at the waist with straight legs instead of squatting like the motion prior. Leg authority was the blocker and is fixed; whether the prior alone now shapes the descent is being tested by a from-scratch run.
- [ ] Recover lift success after the leg-authority change — earlier fine-tunes traded `at_goal` down from 1.0 while adapting.
- [ ] Decide the fate of the inactive reach-phase penalties in `rewards.py`: delete them if the motion prior proves sufficient, or re-add the ones that earn their place.
- [ ] Revisit the locomotion task, which has not been touched since the dual-arm work began.
- [ ] Exercise the depth pathway. DeFM features are wired up but `use_depth` is off by default, so that code path is untested at scale.
- [ ] Carry-to-goal is unimplemented; the task is lift-in-place only.

Infrastructure:

- [ ] Add a `LICENSE` file. `pyproject.toml` declares Apache-2.0 but the license text is absent.
- [ ] Add a regression test for the reward terms. `smoke_dualarm_env.py` covers construction and gating, but nothing pins numerical behavior.
- [ ] Untrack `models/snapshots/` if the historical checkpoints are no longer needed for comparison, and consider rewriting history to drop the artifact blobs.
- [ ] Consolidate the sbatch scripts, which share a large preamble.

-------

## Troubleshooting

**Rendering produces black frames.** The chase camera must be world-mounted
with a target body, never attached to a moving link — a body-mounted camera
inherits that body's rotation and can dip below the floor plane, which
backface-culls the scene.

**`MUJOCO_GL=egl` fails or segfaults.** Expected on machines without libEGL or
libOSMesa. Every render path here uses `MUJOCO_GL=disable` with the
`mujoco_warp` ray tracer instead.

**A resumed run's reward drops sharply at the start.** Curriculum state lives
in the env rather than the checkpoint, so assistance re-weans from full on
every resume. This recovers over a few hundred iterations and is not a
regression.

**A posture reward has no effect.** Check the reachability bound above before
adjusting weights.

-------

## Citation

If you use this code, please cite the methods it builds on:

```bibtex
@article{peng2021amp,
  title   = {AMP: Adversarial Motion Priors for Stylized Physics-Based Character Control},
  author  = {Peng, Xue Bin and Ma, Ze and Abbeel, Pieter and Levine, Sergey and Kanazawa, Angjoo},
  journal = {ACM Transactions on Graphics},
  volume  = {40},
  number  = {4},
  year    = {2021}
}

@misc{resmimic,
  title        = {ResMimic: Residual Learning for Humanoid Whole-Body Loco-Manipulation},
  howpublished = {arXiv:2510.05070},
  url          = {https://arxiv.org/abs/2510.05070}
}
```

## Acknowledgments

Built on [mjlab](https://github.com/mujocolab/mjlab),
[RSL-RL](https://github.com/leggedrobotics/rsl_rl), and
[DeFM](https://github.com/leggedrobotics/defm). Licensed under Apache-2.0.
