Re your question about why the design agent is slow: I can't see the timing of the other agents, so I can't say why it is slow. This review step made no tool calls. The protocol being reviewed is large: it plans 20 h runs for every final seed, a 6-factor robustness sweep and several optional runs. Issue M3 below lists cuts that shorten both the design and the training calendar.

# Protocol review: issues ranked by severity

## Blockers

**B1. The env-step efficiency test (P3) and the equal-env-step checkpoint (E\*) depend on an unpinned choice, not on the method.**
- **Why it matters.** D1 leaves env steps per iteration open (2.5, 1 or 12.5 per env). Under reading (a), SuperMimic collects 10,240 transitions per iteration.
  - At the protocol's own 0.1–0.25 s per iteration, that is about 41k–102k sim steps/s. The assumed PPO rate is about 151k/s (196,608 per 1.3 s). So SuperMimic uses env steps at roughly 0.3–0.7× PPO's rate.
  - Steps-to-X ratio = rate ratio × time ratio. P3 (≤ 0.1×) is therefore just P2 with a stricter bar, and "P2 or P3" collapses into one test.
  - Under reading (b) (51,200 per iteration), SuperMimic collects more than PPO. E\* then exceeds PPO's 20 h total of about 1.1e10 steps, and the claim in §2.1 that the matching PPO checkpoint exists is false.
- **Paper fidelity.** The paper never claims env-step efficiency against PPO-Sim. §4.1 and Fig. 4a claim wall-clock and GPU-hours; Fig. 5b compares imagined steps against PPO-WM.
- **Fix.**
  - Pin reading (a) before M0: 2.5 steps per env per iteration via a fractional accumulator. Remove the update-to-data ratio from every tunable set.
  - Make P3 descriptive only. The verdict becomes P1 and P2.
  - Compute E\* only if it is ≤ PPO's final `env_steps`. Otherwise report "PPO budget exhausted at E_PPO".

**B2. D2 (which heads are analytic) is left open, but it decides whether the actor gets any gradient through termination.**
- **Why it matters.**
  - An analytic termination, d = 1[|e_z| > 0.25 ∨ e_tilt > 0.8], has zero gradient almost everywhere. The survival weight ĉ becomes piecewise constant, which removes the soft-survival mechanism of §3.1 and Eq. 2. §3.1 says the learned model "predicts a termination probability d̂ ∈ [0,1]".
  - The same problem applies to lift. Any box-drop termination, contact indicator or contact-force reward gives zero gradient through backpropagation-through-time (BPTT), yet PPO can use it fully. That is a structural handicap the protocol would score as a difference between the methods.
- **Fix.**
  - Pre-register the heads: reward analytic, observation analytic, termination learned with BCE (SPEC §9.5).
  - Require every lift reward and termination term to be a smooth function of (ŝ, a, g, ξ), or else to be predicted by a learned head.
  - Write this restriction into the shared env spec so that PPO trains on the same reward.

**B3. Parity between the env and the learner is untested, and the oracle test cannot catch the most likely bug.**
- **The oracle test is too weak.** The kernels are wide (0.3 m, 0.4 rad, 1.0 m/s, 3.14 rad/s). A one-frame reference misalignment at 1 m/s is about 2 cm, which gives exp(−0.02²/0.3²) ≈ 0.996. M0(b)'s R_norm ≥ 0.99 therefore passes with an off-by-one error. The pairing matters: mjlab computes r_{t+1} on the post-step state against the pre-advance frame (`manager_based_rl_env.py:436-456`), and the learner's analytic reward on ŝ_{t+1} must use exactly the same index.
- **The contract is incomplete.** It exposes the next K reference frames for live envs only. Imagination needs reference frames at arbitrary (motion_id, frame) taken from the buffer, up to t+H+K = +42 frames ahead. It also needs the reference re-anchored to the robot's xy and yaw, recomputed from ŝ at every step.
- **Fix.**
  - Add test M0(i). On at least 10k logged sim transitions, recompute from the raw tensors: the per-term reward r(s_{t+1}, a_t, a_{t−1}, ref_t), the noise-free observation, g_t and the termination rule. Require each to match the env within 1e-5 (fp32).
  - Add a ±1-frame shift as a negative control that must fail the parity test.
  - Replace M0(b) with kinematic replay (write qpos/qvel, run mj_forward, no physics step). Require raw E_g < 1 mm and every kernel > 0.9999.
  - Extend the contract so the env exposes the whole motion library, or a `ref_at(motion_id, frame)` function.

## Major

**M1. The statistics cannot support the stated claims.**
- Nominal evaluation is 1 deterministic episode per clip on 2 clips, so each checkpoint yields 2 numbers. X, time-to-X, P1 and the tuning ranking all rest on them, and a single fall changes the result.
- A 95% bootstrap over 3 seeds has only 10 distinct seed resamples, so the interval is meaningless.
- **Fix.**
  - Use in-distribution R_norm as the primary metric for X, P1 and tuning: 512 episodes per clip with common random numbers, meaning the same ξ draws and the same η streams for both arms.
  - Keep nominal R_norm as the paper-faithful secondary metric.
  - For superiority, use an exact permutation test: all 3 SuperMimic seeds above all 3 PPO seeds gives p = 1/C(6,3) = 0.05.

**M2. Time-to-X is measured too coarsely for the regime where the answer will fall.**
- Both arms will probably cross X within 1–3 h. The paper reports SuperMimic at 0.82 on 2.45 h of LAFAN1 within 1.2 h (Fig. 4a).
- 30-minute checkpoints resolve P2's 2× threshold with only 2–6 points.
- **Fix.** Run a light in-process evaluation every 5 minutes on a separate eval env:
  - normalizers frozen;
  - the timer paused, so the time is excluded from `wall_clock_s` in both arms;
  - 30-minute checkpoints kept for the full evaluations.

**M3. The 20 h cap is about 3–5× larger than needed and puts the milestones in series.**
- The final runs take about 290 of the 410 GPU-h. The protocol's own expectation in §2.4 is that both arms saturate on 8 s or 14 s of mocap.
- **Fix.**
  - Set cap = max(6 h, 3 × PPO's median time-to-plateau in the tuning runs), the same for both arms, and recompute D5 from it.
  - Decouple M2 from M1. Collect world-model training data from M0 with a reference-feedforward policy, a = (q_ref − q0)/α + N(0, 0.1²). Keep rollouts from PPO checkpoints as an extra held-out set.
  - SuperMimic development (M2–M3) then overlaps M1, which saves about 1–2 calendar days.

**M4. M0(c) tests the wrong feasibility property; for lift this blocks progress.**
- **Why it matters.**
  - With clip_actions=None, any q_sp is reachable as a setpoint, so the test as written is trivially satisfied.
  - The real constraint is quasi-static: the setpoint offset is about τ_grav/K_p, and BeyondMimic's gains are low (about 10 Hz with armature inertia).
  - In the deep squat (pelvis 0.39–0.46 m, knees 1.9–2.2 rad), the knee setpoint needed may exceed the joint range or ctrlrange, or τ may exceed the effort limit. This is the same class of failure as the v15b reachability problem.
- **Fix.** For every reference frame:
  - compute τ_ff from mj_inverse;
  - run a privileged feedforward-PD replay with q_sp = q_ref + τ_ff/K_p;
  - report per-joint max |τ|/τ_max and the setpoint against the ctrlrange;
  - require both to be < 0.9;
  - repeat the check with the nominal box mass for lift.

**M5. Gradients through the analytic heads may be NaN.**
- **Why it matters.**
  - Orientation errors computed as quaternion → axis-angle usually use atan2 or acos plus a `torch.where` branch for small angles. Inf or NaN in the unselected branch still propagates, because 0·Inf = NaN.
  - The same risk applies to sqrt(Σx²) at 0, normalizing the 6D rotation at zero norm, and atan2 when removing yaw.
  - Clipping by global norm does not remove NaN, and one NaN corrupts the whole 32-step BPTT update.
- **Fix.**
  - Add a gradient-check suite to M0: `torch.autograd.gradcheck` in float64 on every analytic reward, observation, goal and termination function, at zero error, a π rotation, zero velocity, and upright and inverted poses.
  - During training, skip and count any update with a non-finite gradient. Raise an alarm above 0.1% of updates.

**M6. Buffer, ξ and reset bookkeeping are untested, and auto_reset=False changes the shared env.**
- **Why it matters.**
  - Risks: windows that cross an episode boundary or the FIFO seam; time-out steps followed by post-reset states; ξ from the next episode written into the terminal transition when a reset-mode DR event fires before logging; obs history or a_{t−1} at episode start not matching the env's reset values.
  - rsl_rl PPO relies on the env's auto-reset. If the two arms reset through different paths, the DR events, reset noise and return-based sampler can fire differently, which breaks the "identical env" guarantee.
- **Fix.**
  - Use one env-side `reset_idx` path for both drivers, called after logging.
  - Add these M0 tests:
    - over 1e6 sampled windows, none crosses an episode-id change or the ring seam;
    - ξ is constant within every logged episode;
    - terminal steps have d=1, time-out steps have d=0 and never appear inside a window;
    - a KS test shows the ξ and start-bin distributions are identical under the PPO and SuperMimic drivers with the same seed;
    - the history and a_{t−1} at the first step equal the env's reset values.

**M7. Clip-end bootstrapping differs between the arms, and with episodes ending at the clip end it describes a continuation that never happens.**
- **Why it matters.**
  - rsl_rl PPO bootstraps a time-out as r += γ·V(o_t), the value of the pre-step observation. SuperMimic (SPEC §9.14) bootstraps with V(ŝ_end).
  - With episodes ending at the clip end, both arms value "holding the clamped last frame", which evaluation never scores.
  - In addition, `--init_at_random_ep_len True` in the sbatch adds early time-outs in the first episode for whichever arm honours the flag.
- **Fix.**
  - Preferred option: treat the clip end as the true end of a finite-horizon task in both arms. Use discount 0, keep the BCE label d=0, and add normalized time-to-clip-end to the critic input in both arms.
  - Alternative: keep bootstrapping, but adopt the hold extension so the continuation actually happens.
  - Test either choice on a 3-step toy in both arms. Set `init_at_random_ep_len=False` for both.

**M8. SuperMimic's tuning set is unbounded, and the 4 h ranking uses a different LR schedule from the finals.**
- **Why it matters.** "Any UNSPECIFIED knob" covers about 30 items, compared with PPO's 4. A 4 h run truncated on a 20 h cosine schedule ranks configurations at a learning rate the final runs never use at that point.
- **Fix.**
  - Enumerate SuperMimic's set: {ELU vs LayerNorm + SiLU, H = 32 vs 16, critic on (fixed)}, at most 6 configurations, with everything else frozen at the SPEC §9 defaults.
  - Tuning runs use the final runs' D5 schedule, truncated.
  - Rank by the area under the in-distribution R_norm curve, not by the value at 4 h.
  - Use the same TF32 and compile/CUDA-graph policy in both arms. Add a hash of the runtime config next to the env-config hash.

**M9. Lift success and R_norm definitions conflict with the reference.**
- **Why it matters.**
  - "No box–floor contact after t_grasp^ref + 0.5 s" fails whenever the reference box is still on the floor 0.5 s after the grasp.
  - With box terms included, R_norm against r_max = 0.1 can exceed 1.
- **Fix.**
  - Define t_liftoff^ref as the first frame where the bottom of the reference box is more than 2 cm above the floor. Forbid floor contact from t_liftoff^ref + 0.3 s.
  - Set r_max_task = Σ(positive weights) × Δt for each task, and keep tracking-only R_norm as the number compared across tasks.

## Minor

- **m1.** Resumed runs are excluded, so intermediate checkpoints need no Adam state. Store weights and normalizers only, and keep full state for the last checkpoint. This makes them about 3× smaller.
- **m2.** `timeout --signal=INT 20h uv run …` signals uv, not python, and rsl_rl does not save on SIGINT. Use an internal monotonic-clock deadline that saves and exits, with `timeout` as a backstop at cap + 30 min. Call `.venv/bin/python` directly.
- **m3.** Start `wall_clock_s` at the first env.step after env construction and Warp kernel compilation, in both arms.
- **m4.** Measure D5's seconds per iteration over at least 2000 iterations after the buffer is full, not in a 200-iteration smoke run.
- **m5.** σ_d for e_k currently comes from an untrained, seed-dependent policy. Compute it once from kinematic replay of the reference clips, with a floor of 1e-3, and share it across runs. Alternatively, compute e_k in the eval job from the checkpointed world model on fresh sim rollouts, so the 64-env validation stream costs no training time, enters no buffer, normalizer or sampler, and needs no `env_steps` accounting.
- **m6.** For the imagination gap Δ_J:
  - report mean |Δ_J| as well as the signed mean, because positive and negative errors can cancel;
  - weight the sim side by the hard c and the imagination side by ĉ;
  - add a "predicted vs realized improvement" check: ΔĴ against ΔJ_sim across actor updates on a fixed start set. A negative correlation is an alarm that the policy is exploiting the world model.
- **m7.** Add a stop-gradient test. After the actor's backward pass, the world-model and critic parameters must have no gradient. Perturbing the termination head's bias must change the actor's gradient, which shows that ĉ carries signal.
- **m8.** Add a BPTT gradient check on a toy linear system with a known analytic gradient.
- **m9.** The verdict table is not exhaustive: P1 can pass on walk while P2 fails. Add a "Comparable, not faster" verdict.
- **m10.** The return-based sampler: state that G_b comes from sim rollouts in both arms. With 4–7 bins per clip it is close to uniform; record that in R4.
- **m11.** The preemption-exclusion rule must apply to PPO too, or the results carry survivorship bias.
- **m12.** The 12-GPU per-user limit includes your running jobs (for example v19, job 2108914). Schedule the 12 finals after tuning, and record the hostname for each run.
- **m13.** The robustness sweep can drop the 2× CoM corners (16 cells) or use 128 episodes per cell without losing anything that matters.