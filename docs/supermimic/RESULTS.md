# SuperMimic vs PPO-Sim: walk, squat and lift

This page records the 2026-10-22 runs. They compare SuperMimic (model-based RL, arXiv 2610.28480; no public code, reimplemented here) with the paper's PPO-Sim baseline on one shared mjlab tracking env. Every number is copied from the running log of those runs, which this file replaces, except two table cells noted below. Design and audit: `IMPLEMENTATION_BRIEF.md`, `design_*.md`, `paper_fidelity.md`.

## Setup

| Item | Value |
|---|---|
| Tasks | `G1-SMTrack-Walk`: `dataset/tracking/locomotion1.npz` (199 frames at 50 Hz). `G1-SMTrack-Squat`: the lift motion with no box, which separates "can track the motion" from "can carry the box". `G1-SMTrack-Lift`: `dataset/tracking/dual_arm1_box.npz` (520 frames; the dual-arm mocap is treated as a 20 fps capture). |
| Box | 0.30 × 0.32 × 0.36 m, 1.5 kg. After the grasp frame its reference follows the mocap palm midpoint, offset 5 cm forward (thigh/shin clearance 6.6–8.3 cm). It rises from 0.18 m to about 0.97 m at ≤ 0.6 m/s. |
| Env | mjlab tracking task on mjlab's G1, with the paper's changes: joint-limit weight −2, no self-collision term, no pushes, per-episode DR (foot friction, encoder bias, torso CoM; the lift adds box mass and friction × U(0.8, 1.2)). |
| Start sampling | Both learners use mjlab's failure-adaptive sampler, a documented deviation from the paper's return-based sampler. The lift has two variants: `b2` uses mjlab's sampler as is; `u50` is a true 50/50 uniform/adaptive mixture (paper App. D.1). Walk and squat use mjlab's sampler. |
| SuperMimic | Paper Table A.4/A.5 values, 4096 envs, 2.5 env steps per env per iteration (about 10.2 k env steps per iteration), torch.compile on. The analytic reward equals the env reward on GPU to ≤ 3.9e-7 over 131 k transitions, with 0 termination/truncation mismatches. |
| PPO-Sim | rsl_rl PPO, 8192 envs × 24 steps (about 197 k env steps per iteration), BeyondMimic hyperparameters, and SuperMimic's encoder/decoder actor (`rl/supermimic/ppo_models.py`). The clip end is terminal (no bootstrap), the same as SuperMimic's `clip_end="stop"`. |
| Seeds | 2 per learner and task (`s0`, `s1`). |

Why `u50`: running one env from each start frame (`phase_map_track.py`) showed that an early SuperMimic lift checkpoint (s0, iteration 1000) already did the full pick-up from 18 of 21 start frames. It failed from the first second of standing (start frames 0 and 25) and from start frame 100, which slipped late, at 9 s. mjlab's sampler starves that second: its "10 % uniform" share is about 0 % at 4096 envs, and 25–27 % of starts land in the final hold second.

## Evaluation protocol

`eval_track.py --randomize`: training DR, observation noise and reset noise; every episode starts at frame 0; deterministic (mean) action; 64 episodes with fixed DR draws (seed 0).

- **Success:** no termination through the clip end.
- **Lift success:** success, and the box within 10 cm of its reference height at the clip end.
- **Norm return:** undiscounted reward sum divided by the maximum possible.
- **Anchor-rel err:** mean anchor-relative position error of the 14 tracked bodies.
- **Resolution:** one episode is 1/64 ≈ 0.016 of a rate.

**Observation-noise caveat (found 2026-10-23).** Under `--randomize`, PPO-Sim read the env's noisy `actor_hist`, but SuperMimic's inference policy rebuilt its observation frames with `obs_noise=False`. SuperMimic trains with the same noise (`features.OBS_NOISE`). So every SuperMimic number and figure below was measured on clean observations, while PPO-Sim's were noisy.

`eval_track.py` now turns SuperMimic's noise on under `--randomize` and records `obs_noise` in its output JSON. `--sm-clean-obs` restores the old behaviour. These walk, squat and lift checkpoints have not been re-evaluated with matched noise; do that before quoting them as final.

## Walk and squat

| Task | Learner | Ckpt | Env steps | Success | Norm return | Anchor-rel err | Peak q̇ (rad/s) |
|---|---|---|---|---|---|---|---|
| Walk | SuperMimic s0 / s1 | 1000 | 10 M | 0.98 / 1.00 | 0.832 / 0.858 | 2.9 / 2.8 cm | 11.3 / 10.5 |
| Walk | **SuperMimic s0 / s1** | 10000 / 6000 | 102 / 61 M | **1.00 / 1.00** | **0.954 / 0.934** | **1.4 / 1.5 cm** | 9.5 / 9.3 |
| Walk | PPO-Sim s0 / s1 | 500 | 98 M | 0.19 / 0.25 | 0.317 / 0.363 | 5.6 / 5.3 cm | 13.4 |
| Walk | PPO-Sim s0 / s1 | 2500 | 491 M | 0.95 / 0.97 | 0.851 / 0.848 | 2.6 / 2.7 cm | 10.7 / 10.2 |
| Squat | SuperMimic s0 / s1 | 1000 | 10 M | 1.00 / 1.00 | 0.917 / 0.907 | 2.2 / 2.6 cm | 13.4 / 14.9 |
| Squat | **SuperMimic s0 / s1** | 5000 / 7000 | 51 / 72 M | **1.00 / 1.00** | **0.978 / 0.980** | **1.2 / 1.2 cm** | 4.6 / 4.2 |
| Squat | SuperMimic s0 | 11000 | 113 M | 1.00 | 0.946 | 2.4 cm | 4.4 |
| Squat | PPO-Sim s0 / s1 | 2500 | 491 M | 0.98 / 1.00 | 0.927 / 0.947 | 2.7 / 2.5 cm | 6.9 / 6.5 |

- PPO-Sim walk and squat were stopped at iteration 2500 once converged.
- SuperMimic is at ≥ 98 % success from its first checkpoint (10 M env steps) on both tasks. On walk, that first checkpoint is already level with PPO-Sim's last (491 M): success 0.98–1.00 vs 0.95–0.97, norm return 0.83–0.86 vs 0.85.
- At its best checkpoints SuperMimic has about half of PPO-Sim's body error.

## Lift

Latest checkpoints, as of 15:14 UTC 2026-10-22:

| Run | Ckpt | Env steps | Lift success | Box tilt (median) | Norm return | Anchor-rel err |
|---|---|---|---|---|---|---|
| SuperMimic s0 (u50) | 15000 | 154 M | **1.00** | 1.2° | **0.985** | 1.6 cm |
| SuperMimic s1 (u50) | 12000 | 123 M | **1.00** | 1.6° | 0.983 | 1.8 cm |
| SuperMimic s0 (b2) | 17000 | 174 M | **1.00** | 1.3° | 0.981 | 1.8 cm |
| SuperMimic s1 (b2) | 18000 | 184 M | **1.00** | 1.2° | 0.982 | 1.6 cm |
| PPO-Sim s0 (u50) | 8000 | 1.57 B | 0.97 | 3.5° | 0.896 | 2.9 cm |
| PPO-Sim s1 (u50) | 8000 | 1.57 B | 0.98 | 5.1° | 0.899 | 2.5 cm |
| PPO-Sim s0 (b2) | 6000 | 1.18 B | 0.61 | 2.5° | 0.574 | 6.5 cm |
| PPO-Sim s1 (b2) | 6000 | 1.18 B | 0.78 | 5.9° | 0.685 | 8.2 cm |

- **SuperMimic:** all four runs reach 100 %, norm return 0.98 and box error 0.6–0.7 cm.
- **PPO-Sim `u50`:** plateaus at 97–98 % success, norm return about 0.90 and box error about 3.5 cm after 1.6 B env steps.
- **PPO-Sim `b2`:** oscillates between 61 % and 86 % across its late checkpoints.

Lift success over env steps, `u50`:

| SuperMimic | 10 M | 20 M | 31 M | 41 M | 51 M | 61 M |
|---|---|---|---|---|---|---|
| s0 | 0.47 | 0.98 | 1.00 | 1.00 | 0.89 | 1.00 |
| s1 | 0.55 | 0.97 | 1.00 | 1.00 | 1.00 | 1.00 |

The running log's 12:14 table had no row for s0 at 41 M or s1 at 61 M. Those two cells come from the eval JSONs beside the checkpoints (`eval_rand_model_4000.json`: 1.00, norm return 0.965; `eval_rand_model_6000.json`: 1.00, norm return 0.976).

| PPO-Sim | 197 M | 295 M | 393 M | 491 M | 590 M | 688 M | 786 M |
|---|---|---|---|---|---|---|---|
| s0 | 0.89 | 0.88 | 0.94 | 0.97 | 0.97 | 0.97 | 0.97 |
| s1 | 0.81 | 0.80 | 0.84 | 0.94 | 0.91 | 0.95 | 0.97 |

PPO-Sim `u50` at 98 M env steps (iteration 500): 23–38 %.

- **Sample efficiency.** SuperMimic reaches 97–98 % on all four runs: both `u50` seeds at 20 M env steps and both `b2` seeds at 51 M. It reaches 100 % at 31 M. PPO-Sim `u50` reaches 94–97 % at 491 M.
- **Tracking quality at those points:** norm return 0.96 vs 0.83–0.85, and box error 1.3–1.4 cm vs 4.5–6.0 cm.
- **Start sampling matters far more for PPO-Sim.** With mjlab's sampler (`b2`), PPO-Sim is at 42–62 % between 393 M and 491 M and ends at 0.61 / 0.78 at 1.18 B. SuperMimic's `b2` runs reach 97–100 %. Early points: PPO-Sim `b2` was at 0 % at iteration 500 and 6–14 % at iteration 1000; SuperMimic `b2` was at 0 % at iteration 1000 (10 M).

## Checkpoint variance

- **Late-checkpoint dips:**
  - SuperMimic squat s0 at iteration 11000: norm return 0.978 → 0.946.
  - Lift s0 `u50` at 5000: 100 % → 89 %, norm return 0.965 → 0.861; back to 100 % / 0.971 at 6000.
  - Lift s1 `b2` at 8000: 100 % → 86 %; back to 100 % at 9000.
- **Oscillation, not decline.** The eval DR draws are fixed, so the episodes that fail at 8000 (squat steps 120–175, lift steps 200–221) succeeded at 7000. The deterministic policy oscillates between checkpoints in the two hard phases. Through the lift s0 `u50` dip the training metrics stay smooth: sim norm return 0.877 → 0.881, the WM loss keeps falling, and there is no termination spike.
  - Report the mean over the last few checkpoints.
  - Polyak/EMA actor weights for evaluation are a candidate fix.
- **Second DR draw.** SuperMimic s0 iteration 4000, with 128 episodes and DR seed 1, gave 88 %:
  - 13 falls in the squat (frames 90–161, `anchor_pos`);
  - 2 box drops at the lift (frames 202–214).

  No DR parameter separates the failures strongly. The largest effect is the torso CoM x shifted back, at 0.8 std. Box mass and friction terciles do not matter (9–14 % failure each).

## Figures

- Per task: `walk_rand_success.png`, `walk_rand_norm_return.png`, `squat_rand_success.png`, `squat_rand_norm_return.png`, `lift_rand_success.png` and `lift_rand_norm_return.png`. Each shows randomized success or normalized return against env steps, one line per run and one panel per start-sampling variant.
- `all_rand_norm_return.png`: the three normalized-return figures stacked.
- The figures were made on 2026-10-22 at 11:42 UTC. They stop at the checkpoints evaluated by then, and the observation-noise caveat applies to them.

Regenerate one figure per task and metric:

```
PYTHONPATH=src .venv/bin/python src/mjlab_g1/scripts/plot_rand_success.py \
  --task G1-SMTrack-Lift --output docs/supermimic/lift_rand_success.png
PYTHONPATH=src .venv/bin/python src/mjlab_g1/scripts/plot_rand_success.py \
  --task G1-SMTrack-Lift --metric norm_return --output docs/supermimic/lift_rand_norm_return.png
```

For the other figures, use `G1-SMTrack-Walk` or `G1-SMTrack-Squat` with the matching output name.

## Hardware and wall-clock

PPO-Sim ran on RTX PRO 6000 (Blackwell, 96 GB) and SuperMimic on L40S (48 GB), so this page does not compare wall-clock time. Success and env-step curves do not depend on the GPU.

Compiling SuperMimic's per-step imagination and WM loss cut an iteration from 3.5 s to about 1.0 s on an H100:
- WM step: 2580 → 470 ms; actor step: 680 → 251 ms.
- One-time compile cost: about 10 min.
- Compiled matches eager to 1e-4 on returns and to about 0.1 % on gradients.

## Where the runs are

- **Runs:** `logs/smtrack/G1-SMTrack-{Walk,Squat,Lift}/{ppo,supermimic}/<date>_<run>/`, which are not in git. Run names are `ppo_s{0,1}` and `sm_s{0,1}`, with a `_b2` or `_u50` suffix on the lift.
- **Evaluations:** randomized evals sit beside the checkpoints as `eval_rand_model_<it>.json`. `progress_track.py` prints a progress table.
- **Videos:** the side-by-side videos from `render_track.py` (SuperMimic left, PPO-Sim right; nominal, frame 0) are `models/videos/compare/{walk,squat,lift}_supermimic_vs_ppo.mp4` and `lift_supermimic12000_vs_ppo6000.mp4`. They are local only, because `models/videos/` is gitignored.
