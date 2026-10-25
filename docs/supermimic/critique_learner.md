# Review of the SuperMimic learner design: ranked issues and fixes

I made two tool calls to check code, both read-only. The citations from them are:
- `MJ/envs/manager_based_rl_env.py:386-401` and `:432-454`, which show the order in which mjlab's `step()` runs.
- `rsl_rl/rsl_rl/algorithms/ppo.py:151-155`, which shows how the vendored PPO handles time-outs.

Other `MJ/...` citations come from the spec.

## BLOCKERS

**B1. The reward and termination parity check (§4.2) will abort at startup.**
- **Why:** mjlab computes terminations and rewards on body kinematics that are one physics substep (5 ms) out of date.
  - The order is the decimation loop, then `termination_manager.compute()` and `reward_manager.compute()` (`manager_based_rl_env.py:432-441`), then the only `sim.forward()` (`:454`).
  - The docstring says so directly: "termination and reward managers see derived quantities that are stale by one physics substep" (`:396-398`).
- **Effect:**
  - `TorchFK(post-step qpos, qvel)` gives up-to-date kinematics. Foot and hand positions will differ by a few mm to about 1.5 cm while walking, and velocities come from a different qvel.
  - The difference per term is about 1e-3 to 1e-2, far above the 1e-4 threshold.
  - Reproducing the stale values exactly from the raw tensors is impossible, because they depend on the qvel before the last substep, which is not exposed.
- **Fix (env side, shared by both arms, so it stays fair):**
  - In the SuperMimic/PPO env subclass, override `step()` to call `self.sim.forward()` once more before `termination_manager.compute()`. This costs one extra forward per control step.
  - Then compare the analytic and env rewards per term, on unscaled term values, with a relative tolerance of 1e-4.
  - Keep a regression test (m9 below).

**B2. The PPO arm has no observation, goal or critic pipeline, so the "identical env" premise does not hold.**
- **Why:**
  - Under the contract the learner builds every feature, including the noise η and the 10-frame history.
  - The file plan has no PPO adapter. Without one, PPO falls back to mjlab's tracking ObservationManager, which feeds the actor the anchor position, base linear velocity and current command, and adds its own noise.
  - The paper's PPO-Sim "shares … observations, reward, termination, domain randomization, actor and critic network sizes, and evaluation protocol" (§4.1).
- **Fix:**
  - Add `ppo_adapter.py`, a VecEnv wrapper that calls the same `features.py` (`PolicyObsHistory`, `goal`, `sensor_noise`, `encode_xi`). It emits:
    - `{"policy": [hist 930/1020, goal 640/730]}`
    - `{"critic": [φ, g, enc ξ, time feature, privileged block from M1]}`
  - Add an rsl_rl actor module with the same encoder and decoder: (2048, 1024, 512, 256), z = 32 with tanh, plus a Gaussian std. The critic is (256, 128).
  - Use 8192 envs, as PPO-Sim did (§4.1).
  - Test: for the same raw step and noise seed, the PPO and SuperMimic observations must be bitwise identical.

**B3. The two arms optimize different objectives at the clip end.**
- **Why:**
  - The vendored PPO adds `gamma * V` on every `extras["time_outs"]` (`ppo.py:151-155`).
  - The env reports the clip end as a time-out (spec #16), while design D4 sets the continuation to 0 there.
  - So PPO bootstraps a future that does not exist, while SuperMimic optimizes the finite episode. The paper's normalized return is also finite-episode (§4).
- **Fix:**
  - The env emits a separate `clip_end` flag.
  - The PPO adapter removes `clip_end` from `time_outs`, so only a true wall-clock time-out bootstraps, if one exists.
  - Both critics get the same time-remaining feature (see M7).
  - Test: on a synthetic episode that ends at `clip_last`, the PPO return target must equal the SuperMimic λ-return target.

## MAJOR

**M1. The critic, and to a lesser extent the actor's goal, cannot see what the reward depends on.**
- **Why:**
  - φ ignores world xy and yaw by construction.
  - g contains q_ref, q̇_ref and a relative orientation only.
  - The reward needs more than that:
    - the global anchor position (weight 0.5);
    - the reference anchor z, which is used in re-anchoring body positions (`commands.py:396-405`);
    - the reference root linear and angular velocity, which the global body-velocity terms need.
  - V cannot represent its own reward, so the bootstrap carries a systematic bias, and that bootstrap is weighted γ^32 ≈ 0.72.
  - For walking, the actor never sees the reference's forward speed. It has no base linear velocity and no reference root velocity.
- **Fix:**
  - Add a privileged critic block computed from ŝ and the motion library, given to both arms:
    - R_Hᵀ(p_ref,anchor − p_anchor) (3);
    - yaw error as sin/cos (2);
    - ref anchor z (1);
    - R_Hᵀ(v_ref − v) (3);
    - ω error (3).
  - Optionally add to the goal (labelled ours, applied to both arms) the reference root linear and angular velocity in the reference's own heading frame plus the reference root height: +7 per frame, giving 71 or 80 per frame. These are independent of the robot state, so no state estimator is needed.

**M2. The normalization and scale statistics drift; for the lift this risks NaNs.**
- **Why:**
  - σ_out, σ_nom and the φ normalizer are cumulative and seeded by a 32-step prefill with near-zero actions, so their first values come from a static or falling robot.
  - **(i)** σ_out de-normalizes the output of the trained layer. Every change to σ_out silently changes what f_ω's weights mean. This is the same pathology the user found in v16, where rescaling a resumed policy collapsed training.
  - **(ii)** Before grasp, the box's one-step change is about 1e-6, so σ_nom puts loss weights of about 1e10 on the box dimensions.
  - **(iii)** φ's statistics shift by about 10× once the robot starts tracking.
- **Fix (ours):**
  - Compute σ_a, σ_out, σ_nom and φ's mean and std once, from the motion library's 50 Hz finite differences.
  - Apply physical floors: 1 mm, 5 mrad, 0.02 m/s and 0.05 rad/s for the deltas. Floor the box dimensions at the robot root's values.
  - Freeze these statistics for the whole run and checkpoint them.
  - Report the Fig. 3c error normalized by this frozen σ_nom.

**M3. The actor's input normalization is unspecified (`norm(Hst)`, `norm(g)`), which allows leakage.**
- **Fix:**
  - Actor and critic normalizers are buffers updated only in `collect_step`, from sim frames that include η.
  - They are never updated in `actor_update` or `critic_fit`, never on imagined data, and gradients never flow through them.
  - The same buffers go into the exported policy. PPO follows the same rule.
  - Test: `actor_update`, `critic_update` and `wm_update` leave every normalizer buffer byte-identical.

**M4. It is ambiguous whether a_{t−1} is the clean policy output or the executed action.**
- **Why:**
  - In the sim the executed action is a + ε (σ = 0.1). If the observation uses mjlab's `last_action`, the actor sees a noisy a_{t−1} in the sim but a clean one in imagination.
  - The action-rate term is also biased in the sim only: −0.1·(2·29·0.01)·0.02 ≈ −0.00116 per step, about 1.2% of the maximum reward of 0.1.
- **Fix:**
  - Store both `a_pol` and `a_exec` in replay.
  - The WM consumes `a_exec`. The observation history and imagination use `a_pol`.
  - The env's action-rate term uses `a_exec`, and the parity check feeds `a_exec`.
  - Document this as a deviation.

**M5. The learned termination head will be sharp and badly calibrated.**
- **Why:**
  - Its inputs include the analytic e_pz and e_tilt, so the best head is almost a step function at 0.25 and 0.8. That gives near-zero gradient inside the safe region and a spike at the boundary, which the 0.1 element-wise clip then hides.
  - Terminations are under 1% of steps, so out-of-distribution states drift to d̂ ≈ 0, which the actor can exploit through ĉ.
- **Fix (stays learned, as in §3.1):**
  - Initialize the output bias to logit(empirical termination rate).
  - Log AUROC and Brier score on fresh windows, and compare the imagined and sim termination rates.
  - Pre-register an ablation with an analytic soft termination, σ(max(|e_pz| − 0.25, e_tilt − 0.8)/0.02). This is ours.
  - Decide the anchor body once (pelvis or torso_link) and read it from the env, so e_pz matches `terminated`.

**M6. Replay windows can silently splice unrelated data.**
- **Why:** A window that spans the ring's write pointer joins the newest and oldest data with no term/trunc flag. Mixing targets from `s[t0+j]` and `s_next` also invites reset-splice bugs.
- **Fix:**
  - Exclude t0 values whose window would cross the write pointer, i.e. t0 within k of it.
  - Take every target from `s_next[t0+j−1]`.
  - Carry `ref_idx` per step, which the d̂ inputs need, and use ξ = ξ[t0] for the whole window.
  - Add a test for the ring wrap.

**M7. Clamping `t_left` at 64 makes very different values look identical.**
- **Why:** V_max = 0.1/(1−γ) = 10. With 64 steps left, the remaining discount mass 1−γ^64 is 0.47. With 200 steps left (a 4 s clip) it is 0.87. Value targets differing by about 4 therefore get the same input.
- **Fix:** Feed 1−γ^{t_left}, which is exact and bounded, to both arms' critics.

**M8. Global TF32 corrupts the FK, quaternion, re-anchoring and reward math.**
- **Why:** At a 10-bit mantissa, errors of about 1e-3 build up along a 7-link chain. That fails the FK test's 1e-5 tolerance on GPU and biases goals and Jacobians.
- **Fix:**
  - Enable TF32 only inside the MLP forward passes.
  - Write FK and quaternion operations as explicit element-wise products in fp32.
  - Run `test_fk_matches_mujoco` on GPU as well.

**M9. The lift task is outside the paper's scope and mixes two separate causes of failure.**
- **Why:**
  - The paper excludes object interaction (§6, limitation 3).
  - It also removed pushes because a deterministic world model cannot capture discontinuities (App. A). Grasp onset and slip are discontinuities of the same kind.
  - Before contact, ∂box/∂a = 0, so box rewards send no gradient until the hands touch.
- **Fix:**
  - Run L0 (lift clip without the box, within the paper's scope) and L1 (with the box) for both learners.
  - Log the gap between imagined and sim box reward, to catch a WM that "lifts" the box without a grip.
  - **Flag for the env team:**
    - A 0.40 m box leaves a gap of 1.3 cm per side for clip 1 and 3.6 cm for clip 2 at grasp, since the palms are 0.426 / 0.472 m apart. It is squeezed only near the clip end, where the palms are 0.36 m apart.
    - So grip force would arise only from tracking error. The width or the reference must be fixed per clip.

**M10. ξ completeness is never checked.**
- **Why:** Any randomized parameter missing from ξ makes f_ω non-Markov (§3.2: "the simulator's randomness has known sources").
- **Fix:** Add a startup assertion that:
  - maps every `EventTermCfg` that edits model fields (mode startup or reset) to a slice of ξ, and refuses to run otherwise;
  - requires that no `interval` or `step` events exist, i.e. no pushes;
  - exempts reset-time state noise.

**M11. The compute and schedule protocol is not fair as written.**
- **Why:**
  - 320k iterations at 0.2–0.3 s each take 18–27 h, more than the 1-day `short` partition.
  - The cosine schedule is tied to 250k iterations, while PPO-Sim had an 18 h budget.
  - On about 8 s (walking) and 14 s (lift) of reference, both arms will probably saturate early.
  - The GPU pool mixes H100, H200, A100 and RTX PRO 6000.
- **Fix:**
  - Give both arms the same wall-clock budget on the same pinned GPU model.
  - Rescale SuperMimic's cosine schedule to the measured iterations in that budget, labelled as a deviation. Keep BeyondMimic's adaptive-KL schedule for PPO.
  - Run at least 3 seeds per arm.
  - Plot return against wall-clock and against sim transitions: SuperMimic collects about 10k transitions per iteration, PPO 8192×24 ≈ 197k per iteration.
  - Also run PPO with BeyondMimic's native (512, 256, 128) networks as a strong reference, so the comparison does not rest on a weak PPO configuration. The Table A.4 network sizes with lr 1e-3 may be unstable for PPO.
  - Compute normalized return directly in the evaluation. Do not derive it from `Episode_Reward` logs (the user's notes say these are sum/episode-seconds).
  - With only 2 clips, "failed motions" is binary. Add many start frames, a robustness evaluation with domain randomization on, MPKPE, and a pre-registered box success rate.

**M12. The reset sampler is left undecided ("protocol engineer").**
- **Fix:**
  - Primary: both arms use `ReturnBasedResetSampler`. Secondary: PPO with BeyondMimic's failure-based sampler.
  - Define the G_b update as a per-iteration batch mean per bin. A per-episode EMA of 0.01 has a time constant of about one iteration at around 100 episodes per iteration.

## MINOR

- **m1. Off-by-one at the clip end.** In the §4.1 pseudocode, `m_32` is the stale loop variable `alive(k0+31)`. Recompute `alive(k)` after the last increment, and add the case `clip_last = k0+31` to `test_gae_targets`.
- **m2. Keep the J and c indexing.** `J += γ^t c m r` followed by the update of c matches Eq. 2: r_{t+1} is weighted by c_t, so the reward at the terminal step is paid, as in mjlab. Add an explicit test for this.
- **m3. `requires_grad_(False)` toggling.** Under `torch.compile`, toggling it every iteration triggers recompiles. Use `torch.func.functional_call` with detached parameter dicts instead, which also skips the weight-gradient matmuls (about 1/3 of backward FLOPs).
- **m4. Round-trip `write_state`.** The qvel[3:6] frame conversion must also invert correctly in `write_state`. Test that `write_state(read_state())` leaves the sim unchanged.
- **m5. Pre-register the fallbacks.** "WM-only restart" and "H ramp" change the method in the middle of an experiment. Make them separate pre-registered arms, not silent fallbacks.
- **m6. Weak action excitation on some joints.** On low-α joints (wrist), ε = 0.1 in action units is about 7 mrad, so the WM sees little excitation. This is faithful to Table A.5; log the variance of the executed action per joint.
- **m7. Goal frames include the current one.** The goal frames k+off+{0..9} include the current reward frame, while the paper says "future" frames. This is fine, but label it as ours.
- **m8. Checkpoint cost.** The checkpoint with replay is 4.5 GB. Save the replay only at job end. On resume, treat MAX_ITERS as relative, as the user's notes recommend.
- **m9. Missing tests.** Add:
  - a closed-loop gap test between imagination and sim: write 256 buffer starts into held-out eval envs, roll the current actor for 32 steps in both, and compare J, c_H and per-term rewards. This is the main detector of world-model exploitation;
  - float64 `gradcheck` for TorchFK, the integrator (including the exp map near 0) and the reward;
  - normalizer freeze (M3);
  - ring wrap (M6);
  - ξ coverage (M10);
  - PPO/SuperMimic observation parity (B2);
  - clip-end bootstrap parity (B3);
  - stale-kinematics regression after the B1 fix;
  - resume determinism (iteration N+1 after save and load ≈ uninterrupted);
  - the evaluation protocol: friction 1.0, no DR, no noise, play-once. This differs from `MJ/tasks/tracking/scripts/evaluate.py:73`.

## Why the design agent is slow (your question)
From this step I can only see my own input, not the design agent's run. This review used 2 tool calls. Its input was the full paper spec (about 9k words) plus the learner design (about 6k words), all inline. If every design and review agent gets the same full context and writes a long document, those calls run back to back, which most likely explains the wall-clock time. That is an inference, not something I measured. Two ways to speed it up:
- Pass each agent only the spec sections it needs.
- Run the reviews in parallel.