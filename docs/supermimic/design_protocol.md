# SuperMimic vs PPO: comparison protocol and plan (protocol part)

Conventions: `(§x)`, `(Eq. n)`, `(Table A.n)`, `(App. X)` cite the paper. `[IR §x]` cites the integration report and `[SPEC §x]` the paper spec. **(ours)** marks a number or choice the paper does not state. I made no new tool calls; everything below comes from the verified reports.

---

## 0. Decisions pinned here because the input reports disagree

| # | Conflict | Protocol rule (ours) |
|---|---|---|
| D1 | Env steps per SuperMimic iteration: 2.5 per env [SPEC §5 reading (a)] vs 1 [IR §8] | The protocol does not depend on it. Every checkpoint's `infos` must record three counters, and all curves are plotted against them, never against iterations. See the counter table below. |
| D2 | Which world-model heads are analytic: reward + obs analytic, termination learned [SPEC §9.5] vs reward + termination analytic, obs learned [IR §8] | State-prediction error is always reported. Head errors are reported only for heads that are learned (§1.6). |
| D3 | Root body for the reward (pelvis vs torso_link); PPO network size (Table A.4 vs mjlab (512,256,128)) | Not decided here. Both arms must use one env config: assert that the serialized `params/env.yaml` is identical apart from `scene.num_envs`. PPO network size goes through the tuning rule (§2.3), and the better PPO becomes the bar. |
| D4 | What the clip end means | Clip end is a **time-out**: d = 0, bootstrapped, in **both** arms. Required check: the PPO wrapper only exposes `time_outs` for non-finite-horizon tasks [IR §3, `vecenv_wrapper.py:91-92`]. If this is set wrong, PPO treats the clip end as terminal and is handicapped. M0 asserts it. |
| D5 | The 320k-iteration cosine schedule (Table A.5) vs a 20 h wall-clock cap | Measure seconds per iteration in M0. Set `max_iterations = floor(20 h / s_per_iter)` and the cosine horizon to `0.78 × max_iterations` (the paper's 250k/320k ratio). This guarantees the LR actually decays. |

The three counters required by D1:

| Counter | Definition |
|---|---|
| `env_steps` | Vectorized steps × num_envs, collection only |
| `wall_clock_s` | Training process only; evaluation excluded, as in the paper (§4.1) |
| `imagined_steps` | Imagined rollouts × H, per iteration, summed |

---

## 1. Metrics: exact definitions

All metrics are computed by **one learner-agnostic `scripts/eval_tracking.py`** from simulator rollouts. Never use training-time `extras["log"]`: it is never cleared [IR §3], and `Episode_Reward` there is sum divided by episode seconds (memory note).

Common notation:
- T = number of 50 Hz control steps in the clip (walk ≈ 199/207; lift ≈ 347).
- 𝒯 = the 14 tracked bodies.
- Step t is valid if the episode has not terminated by step t.

### 1.1 Return and failure

| Metric | Definition |
|---|---|
| **Normalized return** R_norm (paper name, §4) | R_norm = (Σ_{t=1}^{T} r_t) / (T · r_max). r_t is the total Δt-scaled reward, penalties included. Steps after termination contribute 0. r_max = 0.1 for the Table A.2 tracking terms (5.0 × 0.02). Penalties can make R_norm negative; it is **not clipped**. |
| **Tracking-only R_norm** | The same formula using only the 6 Table A.2 tracking kernels. For lift this excludes box terms, so walk and lift numbers are comparable. |
| **Failed motion** (paper name, App. F.1) | The training termination fires before step T: \|e_{p,z}\| > 0.25 m **or** e_tilt > 0.8, on the pelvis (App. A). |
| **Fall rate** | Failed episodes / evaluated episodes, per condition. |

### 1.2 Tracking errors

These are our additions; the paper reports no MPJPE (§6.4 of the spec). They are averaged over valid steps only and always reported alongside the fall rate.

| Metric | Definition | Unit |
|---|---|---|
| **E_g-mpbpe** (global body position) | mean_{t,b∈𝒯} ‖p_b − p_b^ref‖₂ in the world frame (includes drift) | mm |
| **E_l-mpbpe** (root-relative) | mean_{t,b} ‖R_pel^T (p_b − p_pel) − R_pel,ref^T (p_b^ref − p_pel^ref)‖₂ (same as mjlab's root-relative MPKPE, `MJ/tasks/tracking/mdp/metrics.py:82-105`) | mm |
| **E_ori** | mean_{t,b} ‖log(R_b^ref R_b^T)‖ | deg |
| **E_joint** (MPJAE) | mean_{t,j} \|q_j − q_j^ref\| over 29 joints | rad |
| **E_jvel** | mean_{t,j} \|q̇_j − q̇_j^ref\| | rad/s |
| **E_root-xy / E_root-z / E_root-ori** | pelvis xy drift and z error; pelvis geodesic orientation error | m / m / deg |
| **Joint-velocity ratio** | p95 \|q̇\| / p95 \|q̇_ref\|; flags the v18-style 9× over-speed | – |

### 1.3 Lift success

The box spec belongs to the envs engineer. These definitions take its reference trajectory as given: the box follows the palm midpoint from the grasp to the clip end.

| Metric | Definition (all thresholds ours) |
|---|---|
| **Lift-strict** (primary) | All of the following hold: (i) no termination; (ii) for every step in the **last 1.0 s of the clip** (50 steps, the "hold"): \|z_box − z_box^ref\| ≤ 0.10 m, ‖xy_box − xy_box^ref‖ ≤ 0.15 m, box tilt ≤ 30°; (iii) no box–floor contact after t_grasp^ref + 0.5 s. |
| **Lift-lenient** | No termination, and at the clip end the box centre is ≥ 0.30 m above its spawn height and the box is not touching the floor. |
| **Box tracking error** | mean ‖p_box − p_box^ref‖ over steps from t_grasp^ref to T |
| **Grasp-time error** | t_grasp − t_grasp^ref. t_grasp is the first step with both palms in contact with the box; t_grasp^ref = 2.17 s (clip 1) / 2.23 s (clip 2). |
| **Max drop** | max over post-grasp steps of (z_box^ref − z_box) |

A hold extended past the clip end (for example +1 s of the last frame) must be decided **before M4**. If adopted, it applies identically to training and evaluation in both arms.

### 1.4 Efficiency

| Metric | Definition |
|---|---|
| **X (threshold)** | X_task = 0.9 × the median over seeds of PPO's nominal R_norm at the budget's final checkpoint (ours). PPO reaches X by construction. A second threshold X_fail0 = 0 failed motions. |
| **Steps-to-X** | The first checkpoint `env_steps` at which nominal R_norm ≥ X **and stays there for the next 2 checkpoints**; linear interpolation in log(env_steps). Reported as "> budget" if never reached. |
| **Time-to-X / GPU-h-to-X** | The same, using `wall_clock_s` and `wall_clock_s × n_gpus`. |
| **GPU-hours** | Training wall-clock × GPUs. Evaluation is excluded (as in the paper, §4.1). Tuning GPU-h are reported separately. |
| **Imagined steps** | SuperMimic only, reported as a separate column. Imagined steps are never counted as env steps. |
| **Throughput** | Simulator steps/s vs world-model imagined steps/s on the same GPU (Fig. 5a analogue). |

### 1.5 World-model error (SuperMimic only)

- **Validation stream.** 64 envs (ours) whose transitions never enter the replay buffer.
- **Normalizer σ_d.** Per-dimension std of s over the first 10⁶ collected transitions, then **frozen** for the whole run. The paper's "predefined nominal error" (Fig. 3c) is UNSPECIFIED; this choice is ours.

| Metric | Definition |
|---|---|
| **e_k** (k-step prediction error; paper name "8-step prediction error", Fig. 3c) | e_k = mean_{windows} (1/D) Σ_d ((ŝ_{k,d} − s_{k,d}) / σ_d)². ŝ is rolled out autoregressively from s_0 on the logged actions, with the logged ξ. Reported for k = 1, 8, 32. |
| **Physical-unit errors at k = 8 and k = 32** | Independent of the normalizer: pelvis z (m), pelvis tilt (rad), mean tracked-body position in the heading frame (m), joint position (rad), and box position (m) for lift. |
| **Head errors (only if learned; D2)** | Reward: MSE / Var(r). Termination: Brier score and AUC. Observation: per-term RMSE in physical units. |
| **Imagination gap Δ_J** | Take 1024 eval-env snapshots (s, ξ, obs history) at fixed clip times. Imagine 32 steps with the deterministic policy and the same η realization, then continue the simulator 32 steps from the same snapshots. Δ_J = (Ĵ_32 − J_32^sim) / max(\|J_32^sim\|, 1e-3). No state write-back is needed. |
| **Action magnitude** | p99 \|a\| in imagination vs in the simulator. Divergence signals model exploitation. |

---

## 2. Protocol

### 2.1 Arms, seeds, budgets

| Item | Rule |
|---|---|
| Arms per task | **SuperMimic** (4096 envs, Table A.5) and **PPO-Sim** (8192 envs, BeyondMimic hyperparameters, §4.1; network size from §2.3). Same env config (D3), same Table A.1 observations + K = 10 goal, same critic inputs (s, g, ξ) (ours: keeps PPO's privileged critic symmetric with V_ψ). |
| Reference rows | The existing AMP `Mjlab-G1-Locomotion` / `Mjlab-G1-DualArm` (toaster): evaluated once and labelled "different objective" [IR §4.5]. |
| Seeds | Final seeds {0, 1, 2} for both arms (≥ 3 as required). Tuning seeds {100, 101} are disjoint. Optional: seeds {3, 4} on walk if GPU-h allow. |
| Fresh runs only | No warm start from AMP checkpoints. A preempted or resumed SuperMimic run is excluded, because the replay buffer is not checkpointed [IR §2]. Resubmit it instead. |
| **Equal wall-clock** | Each run has a 20 h training cap (ours), on the **same GPU type for every run of a task** (recorded from `nvidia-smi` in the log; a mismatched run is excluded). Same `--cpus-per-task` and memory. |
| **Equal env-step comparison** | E\* = SuperMimic's median `env_steps` at its 20 h cap. PPO is evaluated at the checkpoint whose `env_steps` is closest to E\* (PPO collects far more steps per hour, so this checkpoint exists). |
| Checkpoint grid | Save at each log-spaced env-step milestone {1e6, 3e6, 1e7, 3e7, 1e8, 3e8, 1e9, 3e9, 1e10} **and** every 30 min of wall-clock, so the E\* and steps-to-X matches exist. Checkpoint size: SuperMimic ≈ 0.2–0.3 GB with three Adam states (ours); about 100 checkpoints per run, so plan for ~25 GB per run on `/scratch`. |
| Primary endpoint | The **final checkpoint at the cap**. Best-of-run is secondary only. |

### 2.2 Evaluation conditions

| Condition | Settings | Episodes |
|---|---|---|
| **Nominal** (App. F.1, exact) | Each clip played from frame 0. Friction 1.0, ξ nominal (CoM offset 0, encoder bias 0), no observation noise, no pushes, no reset noise, deterministic action (PPO uses the Gaussian mean). Lift: nominal box mass/friction/spawn. | 1 per clip. Run 16 identical copies to detect simulator nondeterminism and report the spread, which should be about 0. |
| **In-distribution** | Training DR (ξ resampled per episode) + Table A.1 observation noise, starting from frame 0, no pushes. | 512 per clip |
| **Robustness sweep** (final + E\* checkpoints only) | One factor at a time; every other factor nominal (table below). | 256 per cell |

Robustness sweep factors:

| Factor | Values |
|---|---|
| Foot friction μ | {0.2, 0.3, 0.6, 1.0, 1.2, 1.5}; 0.2 and 1.5 are out of distribution |
| Torso CoM offset | 8 corners of the Table A.3 box at 1× and 2× |
| Encoder bias | {±0.01, ±0.02} rad, all joints |
| Pushes (pelvis Δv) | Δv ∈ {0.25, 0.5, 1.0, 1.5} m/s, 8 directions. Walk: at t = 1.0 s and 2.5 s. Lift: at t = 1.0 s (pre-grasp) and t = 4.5 s (mid-lift). |
| Box mass (lift) | {0.5, 1, 1.5, 2, 3} × nominal |
| Box friction (lift) | {0.3, 0.5, 0.8, 1.0} |

- Pushes were removed from training in **both** arms (App. A), so the push results are out of distribution for both.
- Reported per cell: fall rate, Lift-strict, E_g-mpbpe. Summary: mean success over all cells ("robustness area").

### 2.3 Equal tuning budget

- **Frozen.** Values the paper states (Tables A.2–A.5) are fixed for SuperMimic. Only UNSPECIFIED / (ours) knobs are tunable.
- **PPO tunable set.** {network (512,256,128) vs Table A.4; LR 1e-3 adaptive vs 5e-4; entropy {0.005, 0.01}; num_envs {4096, 8192}}.
- **Budget per arm per task.** ≤ 6 configurations × 1 seed (seed 100) × 4 h, ranked by nominal R_norm at 4 h. Lift gets ≤ 4 configurations per arm.
- **What is reported.** Every tried configuration, including failures.
- **Shared-env rule.** Any env, reward or termination change made while debugging either arm invalidates **both** arms' runs on that task, and both are rerun.

### 2.4 Pre-registered verdict (per task)

| Endpoint | Pass condition for SuperMimic (ours) |
|---|---|
| **P1 non-inferiority at equal wall-clock** | Median nominal R_norm ≥ PPO median − 0.02, **and** in-distribution fall rate ≤ PPO's + 2 pp. Lift: Lift-strict ≥ PPO − 10 pp. |
| **P2 time-to-X** | Median time-to-X ≤ 0.5 × PPO's (≥ 2× faster) |
| **P3 sample efficiency** | Median steps-to-X ≤ 0.1 × PPO's (≥ 10× fewer env steps) |
| **P4 robustness** | Robustness area ≥ PPO's − 0.05 |

Overall verdict:

| Verdict | Condition |
|---|---|
| **"Performs well"** | P1 **and** (P2 or P3) on **both** tasks, P4 not failed |
| **"Mixed"** | All of the above on walk only |
| **"Does not"** | P1 fails on walk |

- **Superiority claims** ("better than PPO") need **all 3 SuperMimic seeds** to beat the PPO median, with a non-overlapping 95% bootstrap CI (resampling seeds × episodes). Otherwise the result is "inconclusive".
- **Expectation, stated before running.** On only about 8 s (walk) / 14 s (lift) of mocap, PPO should also saturate R_norm. The paper never shows an advantage on single clips (§4.1–4.2). The informative endpoints are therefore P2, P3 and P4, not P1.

---

## 3. Milestones

Durations are our estimates: developer days plus compute.

| M | Content | Go criteria (all ours) | No-go action | Duration |
|---|---|---|---|---|
| **M0** Env + data + smoke | npz conversion with the user's G1 FK [IR §4.3]; tracking env per paper MDP; `factory.py`; eval script; train.py type changes [IR §1]; one-time `uv sync` (needs user approval) | **(a)** npz body positions vs env FK < 1 mm. **(b)** Oracle replay (state set to the reference each step) gives R_norm ≥ 0.99, 0 fails, E_g < 5 mm; this validates reward, metrics and pairing. **(c)** Static setpoint test: every reference pose is reachable with `clip_actions=None` (\|a\| max per IR §4.4). **(d)** ξ readback varies across envs (histogram of `geom_friction`; catches the dead `foot_friction` event). **(e)** D4 time-out bootstrapping is active. **(f)** Env-config hash is identical for both arms. **(g)** 200-iteration PPO and SuperMimic smoke runs with no NaN. **(h)** Measured steps/s and s/iter on the chosen GPU, used to set D5. | Fix before any training | 4–5 dev days; ~5 GPU-h |
| **M1** PPO walk | Tuning (≤ 6 × 4 h) then seeds 0–2 × 20 h | ≥ 2 of 3 seeds reach nominal R_norm ≥ 0.70 with 0 failed motions. BeyondMimic-style PPO tracks 4 s walks routinely, so failure means an env bug. | Debug env (reward, termination, action scale); repeat M0 (b), (c) | 1 dev day; ~1.5 days wall; ~85 GPU-h |
| **M2** World model offline on walk | Fit the world model (Eq. 1, k = 8) **offline** on rollouts from M1 PPO checkpoints at 3 quality levels plus 0.1 action noise. Evaluate on held-out validation-seed rollouts. This separates world-model bugs from learner-loop bugs. | e_8 ≤ 0.02 (ours; the paper reports 0.007–0.011 with an unknown normalizer); e_32 ≤ 0.10. Physical units at k = 8: pelvis z ≤ 2 cm, tilt ≤ 0.05 rad, body position ≤ 3 cm, joints ≤ 0.05 rad. Δ_J (M1 PPO policy, 32 steps) ≤ 10%. Learned termination head (if any): AUC ≥ 0.95. | Check state representation and normalization; try LayerNorm + SiLU [SPEC §9.4]; more data | 3 dev days; ~10 GPU-h |
| **M3** SuperMimic walk | Full loop (Alg. A.1). Tuning (≤ 6 × 4 h) then seeds 0–2 × 20 h | No NaN; online e_8 stays within the M2 gate; \|Δ_J\| ≤ 20% at the final checkpoint; nominal R_norm ≥ 0.8 × PPO median → proceed to M5 | ≤ 2 debug rounds (§5 R2), then write up the walk result as negative and run M5 as PPO-only | 5–7 dev days; ~1.5 days wall; ~85 GPU-h |
| **M4** Lift env + PPO lift | Mocap-matched box per the envs engineer; box reference = palm midpoint from grasp to clip end; RSI with the box in hand for post-grasp starts | **Gate 0 (blocking):** box width vs hand-reference consistency. Hand separation is 0.43–0.47 m at grasp but about 0.36 m at clip end, so a rigid 0.40 m box conflicts with the hand reference during the lift. The envs engineer must state how this is handled (width, compliant squeeze, or a reward choice) **before** PPO training. **Gate 1:** ≥ 2 of 3 PPO seeds reach nominal Lift-lenient = 1 and Lift-strict ≥ 50% in-distribution | Iterate the box design (width, friction, spawn). Reruns use PPO only. No SuperMimic lift run until PPO lifts. | 4–5 dev days; ~1.5 days wall; ~80 GPU-h |
| **M5** SuperMimic lift | s += box pose/velocity (13); ξ += box mass/friction; **M2-style offline gate first** on M4 PPO-lift rollouts | Offline: box position error at k = 8 ≤ 2 cm, reported separately for the pre-grasp, grasp ±0.3 s and carry phases; \|Δ_J\| ≤ 20%. Online: as in M3 | §5 R1 fallbacks, then report as an out-of-paper-scope negative result (§6 Limitations) | 3–5 dev days; ~1.5 days wall; ~80 GPU-h |
| **M6** Report | Evaluate every checkpoint (nominal + in-distribution); robustness sweep on final and E\* checkpoints; verdict per §2.4; curves vs env_steps and wall_clock | – | – | 2 dev days; ~15 GPU-h |

Calendar estimate: about 4 weeks, with M4 env work able to overlap M3.

---

## 4. Slurm plan

### 4.1 Placement

| Item | Choice |
|---|---|
| Account / partition | `--account=mmaghelih`, `--partition=short` (24 h, 12 GPUs per user) [IR §5]. Never `cchamzas` and never `quick` (1 GPU per user). |
| GPU | `rtx_pro_6000_b` for **all** comparison runs. It is plentiful (40 GPUs), and 2 tasks × 2 arms × 3 seeds = 12 fits the 12-GPU limit concurrently. Optional: 1 seed per arm on H100 for paper-comparable timing (`--gres=gpu:H100:1`). Never mix GPU types within a comparison. |
| Allocation | `--time=23:30:00`, internal training cap 20 h, `--cpus-per-task=8`, `--mem=64G` (ours) |
| Wall-clock cap | Wrap training in `timeout --signal=INT 20h`, with 30-min checkpoints (§2.1), plus the D5 iteration count. Whichever checkpoint was last saved before 20 h is "final". |

### 4.2 Per-run wall-clock estimates

These are assumptions only; M0 measures the real values and the plan is updated from them.
- **PPO.** 8192 × 24 = 196,608 steps per iteration. Assume about 2e5 sim steps/s (order of Fig. 5a at 4–8k envs) plus about 0.3 s of update, so about 1.3 s per iteration. 30k iterations would take about 11 h on H100 and roughly 11–16 h on rtx_pro_6000_b. This fits the 20 h cap.
- **SuperMimic.** The paper's single-H100 curve runs to about 9.4 h (Fig. A.1). That covers 320k iterations only if one iteration takes about 0.1 s, which is unconfirmed. On rtx_pro_6000_b with TF32 MLPs, assume 0.1–0.25 s per iteration, so 320k iterations would take 9–22 h. D5 truncates the run to whatever fits 20 h.

### 4.3 Scripts to add (new files; the existing AMP launcher is untouched)

```
scripts/cmp/train_cmp.sbatch    # TASK, ARM, SEED=$SLURM_ARRAY_TASK_ID, NUM_ENVS, MAX_ITERS (absolute, not relative)
scripts/cmp/eval_cmp.sbatch     # array over run dirs; evaluates all checkpoints -> eval/<run>/*.json
scripts/cmp/submit_all.sh       # 4 training arrays (--array=0-2) + eval arrays with --dependency=afterany:<jobid>
```

Core of `train_cmp.sbatch`:

```bash
#!/bin/bash
#SBATCH --account=mmaghelih
#SBATCH --partition=short
#SBATCH --gres=gpu:rtx_pro_6000_b:1
#SBATCH --time=23:30:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --array=0-2
#SBATCH --output=logs/slurm/%x_%A_%a.out
set -euo pipefail
export GIT_PYTHON_REFRESH=quiet MUJOCO_GL=disable WANDB_MODE=offline PYTHONPATH=src
SEED=${SLURM_ARRAY_TASK_ID}
cd <repo>
nvidia-smi --query-gpu=name --format=csv,noheader   # recorded for the GPU-type rule
timeout --signal=INT 20h uv run --no-sync python src/mjlab_g1/scripts/train.py "${TASK}" \
  --env.scene.num_envs "${NUM_ENVS}" --agent.seed "${SEED}" --agent.max_iterations "${MAX_ITERS}" \
  --agent.run_name "${ARM}_s${SEED}" --video False --init_at_random_ep_len "${INIT_RANDOM:-True}"
```

- No `uv sync` inside jobs. Run it once by hand, after user approval.
- Submit with `/cm/shared/apps/slurm/current/bin/sbatch --export=ALL,TASK=...,ARM=...` if `sbatch` is not on `PATH`.

### 4.4 GPU-hours

| Item | GPU-h |
|---|---|
| M0 smoke + throughput | ~5 |
| Tuning: walk 2 arms × 6 × 4 h; lift 2 arms × 4 × 4 h | 48 + 32 = 80 |
| Final runs: 12 × ~24 h allocation | ~290 |
| M2 + M5 offline world model | ~20 |
| Evaluation (all checkpoints + sweep) | ~15 |
| **Core total** | **~410** |
| Optional: H100 1-seed replicate (4 × 20 h) | +80 |
| Optional: walk seeds 3–4 (4 × 24 h) | +96 |
| Optional: PPO-4096 ablation (6 × 20 h) | +120 |
| Optional: robot-only lift tracking control, PPO (3 × 20 h) | +60 |

---

## 5. Risks and mitigations

| # | Risk | Mitigation |
|---|---|---|
| R1 | **World model on contact-rich box dynamics.** Friction-only grasp stick/slip is discontinuous, and a deterministic MLP smooths it. The paper excludes objects (§6; App. E) and removed pushes because the model cannot capture discontinuities (App. A). | Box state in the palm-midpoint/heading frame; phase-split error gates (M5); return-based reset weighting to put data around the grasp; box width/friction chosen to keep contacts in the stick regime (M4 Gate 0). If it still fails: (a) predict contact flags as an auxiliary output (ours); (b) an "assisted grasp" variant (a weld when both palms are in contact) applied to **both** arms and labelled as such; (c) report as a negative result. |
| R2 | **Exploding or vanishing BPTT gradients** | Paper clipping: global norm 1.0 then element-wise 0.1 (Table A.5). Log pre-clip norm and fraction clipped; ELU, with LayerNorm + SiLU as fallback; spectral-norm probe of ∂f/∂s (power iteration; alarm if > 1.05). If unstable, H = 16 with the critic (the paper reports it matches critic-free H = 32, §4.3); never H = 8 (fails in the paper). Alarm when \|Δ_J\| > 30% or when imagined p99 \|a\| exceeds the simulator's. |
| R3 | **Timing-locked tracking vs episode length.** The reference advances on the clock, so a robot that lags is penalized without terminating. Mid-clip starts give short episodes. The clip end can fall inside H. (Same class as the AMP 4.60 s vs 4.62 s timing root cause in memory.) | D4 time-out bootstrapping in both arms (asserted in M0 (e)); clamp the reference index and mask rewards after the clip end inside H [SPEC §9.14]; G_max per bin uses the remaining steps; grasp-time error reported (§1.3); the hold-extension decision is frozen before M4. |
| R4 | **Paper results do not transfer to about 8 s of mocap** vs 264 h | No reproduction claim. The verdict focuses on P2–P4 (§2.4). Table A.4 sizes are kept (shrinking adds a confound [IR §8]). Log e_8 on robustness-sweep states to measure world-model out-of-distribution error. Note that the 1 s bins give only 4–7 bins per clip, so the return-based sampler is nearly uniform. |
| R5 | **AMP-era pitfalls leaking into the baseline** | M0 assertions forbid each item in the list below. |
| R6 | **Fairness threats** | See the list below. |

AMP-era pitfalls that M0 must forbid (R5):
- `clip_actions=1.5` (`tasks/dualarm/config/g1/rl_cfg.py:48`); use `None` [IR §4.4].
- The residual knee formulation (the v15b reachability root cause); use plain q_sp = α·a + q0.
- Auto-reset discarding terminal states (use `auto_reset=False`).
- Startup-mode DR and the dead `foot_friction` event (CLAUDE.md pitfall 4).
- `push_robot` left on.
- npz files under `dataset/{locomotion,dualarm}` (the AMP loader calls `listdir` there).
- `uv run train` console-script collision.
- Action rescaling on resume (the v16 collapse).
- Relative `MAX_ITERS` on resume.
- Account `cchamzas`.
- The AMP `train.py` assert.

Fairness threats (R6):
- **8192 vs 4096 envs.** This is the paper's design; it can go into PPO tuning, with an optional PPO-4096 ablation.
- **PPO network size.** Decided by the tuning rule, and the better PPO is the bar.
- **Critic inputs.** Identical (s, g, ξ).
- **Unequal implementation maturity.** The new SuperMimic code is the more likely to contain bugs. Disclose this and show M2/M3 diagnostics.
- **Evaluation.** Simulator only, never in the world model. One evaluation script; deterministic actions for both arms; final checkpoint, not best-of-run.
- **Hardware.** Same GPU type and CPU allocation; evaluation time excluded from wall-clock for both.
- **Shared env.** Any env change reruns both arms (§2.3).

---

**Files:** none written, and no tool calls were made.