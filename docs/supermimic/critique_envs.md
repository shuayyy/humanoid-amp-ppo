# Review of the "envs" design: SuperMimic vs PPO side by side

**Short answer to the user's question about time:** this step cannot see how long the design agent took. What it can see is the size of the job. Each review step receives the whole paper spec plus the whole design (about 15k words) and must back every claim with a file:line citation. This review was capped at 3 tool calls. I spent all 3 checking mjlab v1.4.0 directly, and two of those checks found the blockers B1 and M1.

Paths below are relative to `MJ = <mjlab v1.4.0>/src/mjlab`.

## Blockers

**B1. Resetting finished envs by hand makes every other env skip a reference frame and record a duplicate history frame.**
- **Cause:**
  - `ContractVecEnv` calls `env.reset(env_ids=done_ids)` after every `env.step`.
  - `reset()` runs `command_manager.compute(dt=0.0)` (`MJ/envs/manager_based_rl_env.py:372`).
  - `CommandTerm.compute` always calls `_update_command()` (`MJ/managers/command_manager.py:109-115`). That function runs `self.time_steps += 1` for **all** envs (`MJ/tasks/tracking/mdp/commands.py:407-408`), on top of the normal advance in `step()` at `:456`.
  - `reset()` then calls `observation_manager.compute(update_history=True)` (`:374`), which adds a second history frame to every env (`MJ/managers/observation_manager.py:352-355`).
- **Effect:**
  - With 4k–8k envs, some env terminates on almost every step. The reference therefore plays up to about 2× too fast, and the 10-frame histories hold repeated frames.
  - Both learners are hit identically, so nothing looks unfair. But the replay's frame indices, goal windows and observation histories are all corrupted.
  - The M0 check "frame increments by 1" only catches this if other envs reset during the same step.
- **Fix:**
  - Keep `auto_reset=True`. That uses mjlab's own order: reset (`:443-448`), then forward (`:454`), then advance (`:456`), then one history push (`:464`). History rows of reset envs are backfilled on their next push (`MJ/utils/buffers/circular_buffer.py`).
  - Capture the terminal transition with a recorder term in `record_pre_reset` (`:446`). It should record robot qpos/qvel, box qpos/qvel, ξ, reference index, per-term flags, last executed action and an episode uid, and put them in `extras`.
  - Delete the manual-reset path. mjlab also raises an error if an env that is waiting for a reset gets stepped (`:411-416`, `:469`).
  - Also add a guard so that `MultiClipMotionCommand._update_command` does not advance when `dt == 0`.
- **Test:** force-terminate a random 10% of envs on every step. For all the other envs, assert that the frame advances by exactly 1 and that the history equals the last 10 proprio frames.

**B2. `features.py` cannot be computed from the contract's raw tensors inside imagination, so the analytic reward head and the goal cannot be implemented.**
- The goal needs the robot torso pose, which comes from the pelvis plus the 3 waist joints. The body reward terms need positions, orientations and velocities of 14 bodies. `hands_on_box` needs the palm sites. The critic features need body poses in the torso frame. All of these come from forward kinematics (FK).
- The contract only carries qpos/qvel, and it labels `aux` as "not part of s". So the claim that "SuperMimic's reward head can compute it analytically" has nothing to rest on.
- **Fix:** ship `features.fk(qpos, qvel)`, a batched, differentiable torch FK for G1.
  - Build it from MjModel's static arrays: `body_parentid`, `body_pos`, `body_quat`, joint axes and positions, and the palm `site_pos`.
  - Propagate spatial velocity down the chain so the output follows the same conventions as `robot.data.body_link_{pos,quat,lin_vel,ang_vel}_w`.
  - Validate it on GPU against those fields and `site_xpos`, to < 1e-5.
- If the team does not want FK, the alternative is to commit now to a learned reward head and put body states into s. Either way, the waist chain is needed for the goal.

## Major

**M1. Reward and termination use body kinematics that are one physics substep (5 ms) old, and body targets are re-anchored from the previous state s_t.**
- **Cause:**
  - Terminations and rewards run before `sim.forward()` (`:434-440` vs `:454`). The code comment itself says derived quantities are "stale by one physics substep".
  - `update_relative_body_poses()` runs during the previous step's command update (`commands.py:413` via `:456`). So the reference is re-anchored to the robot anchor of s_t.
- **Effect:** r_{t+1} and d_{t+1} depend on a substep state that is never logged. The analytic terms are then off by about velocity × 5 ms in position and acceleration × 5 ms in velocity, which is large during contacts and grasps.
- **Fix:** override `step` to call `sim.forward()` and then `update_relative_body_poses()` before termination and reward, for both learners. This costs one extra forward per step and makes r = R(s_{t+1}, ref) and d = D(s_{t+1}, ref) exact.

**M2. The PPO baseline does not match the paper's controlled PPO-Sim baseline.**
- The paper's PPO-Sim uses the same actor and critic sizes as SuperMimic (§4.1). The design's primary PPO uses (512, 256, 128) for both, and the size-matched variant is only optional and is a flat stand-in for the encoder/decoder.
- **Fix:** make the primary PPO use the exact Table A.4 encoder/decoder actor as one shared torch module, wrapped as an rsl_rl actor with a learned std, and a (256, 128) critic. Keep mjlab's stock network as a secondary row.

**M3. What happens at the end of a clip is defined three different ways.**
- `motion_end` is a time-out, so PPO bootstraps from the value of the last state. With r about 0.1 per step, that value converges toward r/(1−γ) = 10.
- The per-bin G_max in the reset sampler assumes nothing comes after the clip end.
- The normalized-return metric treats the clip end as the end of the episode.
- **Fix:** define it once.
  - Make `motion_end` "terminal but not a failure". PPO gets done=1 and time_out=0, so no bootstrap.
  - SuperMimic keeps the failure label d at 0 for the termination classifier, but applies an exact mask from the reference index that sets reward and value to 0 after the clip end, both in imagination and in the λ-return targets.
  - Give the critic the normalized number of steps left in the clip, for both learners.

**M4. Domain randomization is drawn once at startup; the paper says per episode (App. A, Table A.3).**
- Besides the fidelity gap, PPO with 8192 envs gets twice as many ξ draws as SuperMimic with 4096, so it is also unequal between the learners.
- **Fix:**
  - Use `mode="reset"` for the friction, CoM, encoder-offset and box randomizations.
  - Snapshot ξ before the reset happens (with the B1 recorder).
  - Check that reset-mode writes to `geom_friction` and `body_ipos` reach the per-world model arrays, without having to recapture the CUDA graph.

**M5. The evaluation protocol is not the paper's App. F.1 protocol.**
- The design evaluates with observation noise and randomization on, over 1024 envs. App. F.1 uses friction 1.0, no randomization, no noise, start from frame 0, deterministic physics, play each clip once.
- **Fix:** report two things. The primary is the F.1 protocol, one env per clip. The secondary is a robustness run (randomization plus noise, 1024 envs, mean and confidence interval).

**M6. Starting episodes mid-lift (after the grasp) does not account for joint reset noise.**
- The box is placed with T_pert ∘ reference, which only moves it with the root, while the arm joints are perturbed by ±0.1 rad. That moves the palms by several cm, far more than the box-width margin of ≤ 1 cm. The result is interpenetration or the box being ejected.
- **Fix:**
  - For starts at f ≥ t_g − 0.5 s, cut arm joint noise to ±0.02 rad.
  - Place the box from the actual palm sites after the noise is applied.
  - Lengthen the settle test from 10 steps to 50 or more.
  - If the fallback (no starts after the grasp) is used, mask the bins that can no longer be sampled in the reset sampler. With 1 s bins and t_g about 2.2 s, that is 5 of each clip's 7 bins. Learning the hold phase then has to start from the squat, which is the exploration trap seen in the user's earlier v14 run.

**M7. Grasp feasibility under the low-gain PD controller is assumed, not measured.**
- The claim that the 0.4–8.7 cm reference squeeze produces 4.9–10.2 N of grip force has not been tested.
- **Fix:** add a pinned-root hold test. Drive the root kinematically along the reference, run reference-PD on the arms from t_g to the clip end, and use the worst case (1.25 kg, μ 0.6). Measure box slip and drop. This separates "can the arms hold it" from "can the robot balance".

**M8. The comparison is not statistically sound yet.**
- The design gives no seed count, and these are single-clip tasks with high variance.
- Training-episode returns cannot be compared between the learners. PPO's std 1.0 exploration alone costs about 2·29·1·0.1·0.02 = 0.116 per step in action-rate penalty, which is more than the maximum per-step reward of 0.1.
- **Fix:**
  - Run at least 3 seeds per learner per task.
  - Save checkpoints at fixed wall-clock intervals for both learners.
  - Evaluate them offline with one deterministic script, and plot against both wall-clock and env steps.

**M9. The robot-only Squat task should be required, not optional.**
- It is within the paper's scope. Contact with an object is outside it: the paper names the lack of terrain and object interaction as its third limitation (§6). Without the Squat run, a SuperMimic failure on the lift cannot be blamed on the box rather than on the motion.

## Minor

1. **Done flags:** if a failure termination and `motion_end` fire on the same step, the failure should win. Set `truncated &= ~terminated`. Otherwise the "terminated ∧ truncated = ∅" test fails.
2. **`write_state` frame convention:** a native reset leaves the command one frame ahead (f+1) once `compute` has run. `write_state(frames=f)` does not advance, so the next reward is scored against f instead of f+1. Make `write_state` mirror `_resample_command` followed by the advance.
3. **Episode uid:** add an explicit `episode_uid [N]`. A reset can draw the same clip at frame+1, so frame jumps alone cannot reliably detect that a window crosses a reset.
4. **Reset sampler details:**
   - Normalize each episode's return by G_max from its actual start frame, not from the bin start.
   - The 0.01 EMA rate used for "past softmax weight" is not slow. Use 0.001, as in spec item 18.
5. **Box mass randomization:** also scale `body_inertia` by the same factor as the mass.
6. **Box yaw:** after t_g, let the box's reference yaw follow the hand axis (up to 7° of drift) instead of keeping it fixed.
7. **`hands_on_box` gate:** the gate opens at t_g − 0.25 s, but the reference window only exposes `post_grasp`. Add a `hands_gate [N,K]` field.
8. **Which body is "root":** termination uses the pelvis, but the root reward and the re-anchoring use torso_link. Use one, or document both.
9. **Reachability:** the |a| figure of 3.9–4.5 for the knees is purely kinematic. Under 10 Hz PD gains the knee sags under gravity, adding about τ_gravity/K_p of offset. Check it with a quasi-static `mj_inverse` at the squat bottom, and confirm the torque stays under the effort limit. Missing reachability was the root cause in the user's v15b run.
10. **`init_at_random_ep_len=True`:** it does nothing useful once episodes end at the clip end. Set it to False.
11. **Exported metadata:** export the ξ ranges and the observation-noise ranges in `extras["meta"]`, so that SuperMimic re-adds exactly the env's noise in imagination. Keep `features.py` unnormalized; normalizers belong to each learner.
12. **Round-trip test tolerance:** `write_state` does not restore MuJoCo Warp's `qacc_warmstart`. Zero it in both runs of the round-trip test, and check run-to-run determinism before relying on "deterministic" evaluation.
13. **Lift return:** the `box_pos` term pays out for free before the grasp. Report returns split before and after t_g.
14. **Checked and fine:**
    - The foot geoms have priority 1 and friction 0.6 (`MJ/asset_zoo/robots/unitree_g1/g1_constants.py:223-224`), so foot-friction randomization does take effect.
    - The box's priority 1 ties with the feet, which is irrelevant.
    - Keep test M4(c) for the hand-box contact.

## Missing tests

| # | Test | Pass |
|---|---|---|
| T1 | Reward and termination parity after the M1 fix: on 10k logged transitions, `features.reward` and `features.term` against the env's per-term values | < 1e-5 |
| T2 | Gradient check and finite differences for every reward term and goal feature. Cover zero error, small error, near 180°, and the quaternion sign flip q vs −q. The orientation terms use `norm(quat_box_minus)` (`MJ/utils/lab_api/math.py:688-699`). | no NaN |
| T3 | Partial-reset invariance (B1) | exact |
| T4 | ξ round-trip: values read back equal the model fields, and the terminal transition carries the ξ from before the reset | exact |
| T5 | FK parity (B2): bodies and palm sites against mjlab data | < 1e-5 |
| T6 | Env-config identity: the PPO and SuperMimic task YAMLs hash the same, apart from `num_envs` and the runner | exact |
| T7 | Reset sampler: with a fixed G_b, the empirical bin frequencies match the mixture | within 2σ |
| T8 | Pinned-root grasp hold (M7) | box error < 5 cm |
| T9 | With noise off, features computed from raw tensors equal the env observations | exact |
| T10 | The executed noisy action is what gets logged, and `last_action` equals it | exact |