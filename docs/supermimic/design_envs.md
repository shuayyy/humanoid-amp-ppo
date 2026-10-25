*Process note: I made 3 tool calls. One failed because the pinned mujoco version wasn't available offline. They were used only to settle the box size, where the two reports disagree (integrator ≈0.40 m, tracking report 0.29–0.32 m), and to read the paper's App. A tables. The FK and box-fit script is `<scratch>/tools/boxfit.py`.*

# Envs, data and PPO baseline design

## 0. Decisions at a glance

| Item | Decision | Basis |
|---|---|---|
| Robot asset (all 4 runs) | **mjlab G1** (`get_g1_robot_cfg()`): KNEES_BENT q0, BeyondMimic gains and α | Paper App. A "Action" uses low BeyondMimic gains. Lift reachability: |a| about 3.4 on wrist roll vs 20 on the user's model (25 Nm, α 0.439 vs 5 Nm, α 0.075), and knees 3.9–4.5 vs 5.8–6.4. **My FK: palm and knee positions are identical between mjlab `g1.xml` and `g1_realsense.xml` on every 5th lift frame (max diff 0.0), so every box number below transfers exactly.** The hand, wrist and elbow collision capsules are identical too. |
| Episodes | Run from the RSI frame to the clip end. A `motion_end` **time-out** ends the episode. There is no wrap or teleport. | Clip wrap teleports (`commands.py:407-411`) are discontinuities. The paper drops pushes for exactly that reason (App. A, "Sim context"). |
| Pushes | Removed | App. A, Table A.3 caption |
| Reward | Table A.2: BeyondMimic's 6 tracking terms plus action-rate −0.1 and joint-limit **−2**; no self-collision term | App. A, Table A.2 |
| Terminations | Pelvis \|e_z\| > 0.25 m and tilt > 0.8 (pelvis frames). No end-effector term. | App. A, "Termination" |
| DR | Table A.3: foot friction U[0.3, 1.2], encoder offset U[±0.01], torso CoM (±0.025, ±0.05, ±0.05). Startup mode. | App. A, Table A.3. The paper says "per-episode"; startup mode is our choice, equal for both learners. |
| Reset sampling | App. D.1 return-based mixture, inside the env, shared by both learners | App. D.1 |
| Lift box | **0.20 (fwd) × 0.36 (hand axis) × 0.30 (h) m, 1.0 kg, μ 0.8 with geom priority 1** | §3, sized by FK |
| Env/learner boundary | `TrackingContractEnv` (auto_reset=False) plus one `ContractVecEnv` wrapper used by **both** learners | §4 |

---

## 1. Data conversion

**Script:** `scripts/npy_to_motion_npz.py`. It uses plain CPU MuJoCo, no W&B and no `/tmp`. It mirrors `MJ/scripts/csv_to_npz.py` but reads the user's wxyz `.npy` and runs FK on the **training** robot.

```python
CANON_JOINTS = [...]   # 29 names, order of MJ/scripts/csv_to_npz.py:394-424 (verified identical in both XMLs)
def load_model(robot: Literal["mjlab_g1","g1_realsense"]) -> mujoco.MjModel   # mjlab: asset_zoo/robots/unitree_g1/xmls/g1.xml
def load_npy(path) -> np.ndarray            # [T,36] wxyz; flip q -> -q where dot(q_t, q_{t-1}) < 0
def resample(q30, fin=30, fout=50):         # == csv_to_npz.py:69-121
    t = np.arange(0, (len(q30)-1)/fin, 1/fout); i = floor(t*fin); a = t*fin - i
    pos, joints = lerp(q30[i], q30[i+1], a);  quat = slerp(q30[i,3:7], q30[i+1,3:7], a)
def velocities(q50, dt=0.02):               # == csv_to_npz.py:123-151
    v_root = np.gradient(pos, dt); dq = np.gradient(joints, dt)
    w_root = so3_log(q[t+1] ⊗ q[t-1]^-1) / (2 dt)          # world frame, one-sided at the ends
def fk(m, q50, v, w, dq) -> dict:           # per frame:
    d.qpos = q50[f]; d.qvel[:3] = v; d.qvel[3:6] = R(q)^T w   # MuJoCo free-joint angular velocity is body frame
    mj_forward; body_pos=d.xpos[1:], body_quat=d.xquat[1:]
    mj_objectVelocity(m,d,mjOBJ_BODY,b,res,0) -> ang=res[:3], lin=res[3:]   # world-oriented, at the body origin
    palm_pos = d.site_xpos[left_palm, right_palm]  # both XMLs have these sites; site = wrist_yaw_link + R·(0.08,0,0), verified err 0.0
def synth_box(palm_L, palm_R, size, dt) -> dict   # §3.3
def main(): assert joint names == CANON_JOINTS; np.savez(out, ...)
```

**npz keys:**
- **Read by `MotionCommand`:** `joint_pos [T,29]`, `joint_vel`, `body_pos_w [T,B,3]`, `body_quat_w`, `body_lin_vel_w`, `body_ang_vel_w`, with B = 30 for mjlab G1, in model body order.
- **Ours:** `fps=[50]`, `body_names`, `joint_names`, `robot="mjlab_g1"`, `palm_pos_w [T,2,3]`.
- **Lift only:** `object_pos_w [T,3]`, `object_quat_w [T,4]`, `object_lin_vel_w`, `object_ang_vel_w`, `object_size [3]`, `t_grasp`, `post_grasp [T]`.

The command subclass asserts at load that `robot` and `body_names` equal `robot.body_names`. This guards against the 30-vs-31 body off-by-one.

**Outputs:**
- `dataset/tracking/walk/<clip>.npz`: 199 and 207 frames.
- `dataset/tracking/lift/<clip>.npz`: 347 frames each.
- Never write into `dataset/{locomotion,dualarm}`, because the AMP loader lists those directories (`motion_loader_g1.py:129,158`).

**Validation:**

| # | Where | Check | Pass |
|---|---|---|---|
| V1 | CPU | Joint names equal CANON; \|quat\| = 1; no hemisphere flips | exact |
| V2 | CPU | Resampled values at the original timestamps k/30 equal the input | < 1e-6 |
| V3 | CPU | Finite difference of `body_pos_w` vs `body_lin_vel_w` | RMS < 0.05 m/s |
| V4 | CPU (lift) | Regression vs this design: t_g ≈ 2.17 / 2.23 s (≈ frame 108 / 112 at 50 Hz), palm separation at t_g 0.426 / 0.472 m, box clearance at frame 0 ≥ 0.19 m | ±0.01 |
| V5 | GPU | Replay: for each f, `write_state` from the npz, then `sim.forward()`, then compare `robot.data.body_link_{pos,quat,lin_vel}_w` with the npz (+ env origin). This also checks that `mj_objectVelocity` matches mjlab's cvel path (`MJ/entity/data.py:20-31`). | pos < 1e-4 m, vel < 1e-3 |

Expected and harmless: the reference foot capsules sit 6–8 mm into the floor.

---

## 2. Walk-tracking task

- **IDs:**
  - `Mjlab-G1-Track-Walk-PPO` (`MjlabOnPolicyRunner`)
  - `Mjlab-G1-Track-Walk-SuperMimic` (`SuperMimicRunner`)
  - Optional sanity check: `Mjlab-G1-Track-Walk-PPO-BM` with mjlab's stock observations.
  - All of them use one env cfg function.
- **Derivation:** `make_g1_track_env_cfg(task="walk", motion_files=glob("dataset/tracking/walk/*.npz"))`. It starts from `unitree_g1_flat_tracking_env_cfg()` (`MJ/tasks/tracking/config/g1/env_cfgs.py`), which is built on `make_tracking_env_cfg` (`MJ/tasks/tracking/tracking_env_cfg.py:40-313`), and applies these changes:

| Field | Value |
|---|---|
| class / cfg | `TrackingContractEnv` / `TrackingContractEnvCfg(ManagerBasedRlEnvCfg)`, `auto_reset=False`, `ref_window_k=10` |
| scene | mjlab G1, plane, `num_envs` 4096 (SuperMimic) or 8192 (PPO). `self_collision` sensor removed. Sim stays at dt 0.005 × 4, nconmax 35, njmax 250. |
| command | `MultiClipMotionCommandCfg(motion_files, sampling_mode="return", end_mode="truncate", anchor="torso_link", 14 tracked bodies of env_cfgs.py:41-57)`. Reset noise unchanged: pose xy ±0.05, z ±0.01, rp ±0.1, yaw ±0.2; velocity ranges as mjlab; joints ±0.1, clipped to soft limits. |
| events | keep `base_com`, `encoder_bias`, `foot_friction` (startup); **pop `push_robot`** |
| rewards | `motion_global_root_pos` 0.5/σ 0.3, `…_root_ori` 0.5/0.4, `motion_body_pos` 1.0/0.3, `…_ori` 1.0/0.4, `…_lin_vel` 1.0/1.0, `…_ang_vel` 1.0/3.14, `action_rate_l2` −0.1, `joint_limit` **−2**; `self_collisions` deleted. All × dt; r_max = 0.1 per step. |
| terminations | `pelvis_height`: \|z_pelvis,ref − z_pelvis\| > 0.25. `pelvis_tilt`: \|g_z,ref − g_z\| > 0.8 in pelvis frames. `motion_end` (`time_out=True`): `time_steps ≥ T_c − 1`. `time_out` 10 s as a safety net. **Removed:** `anchor_pos`, `anchor_ori`, `ee_body_pos`. |
| episode | Ends at clip end, so at most 3.96 / 4.12 s from frame 0. No loop: `_update_command` clamps instead of resampling. |
| actor obs (`obs_set="supermimic"`, also used by PPO) | Proprio per Table A.1 with `history_length=10` in mjlab per-term flattened layout: `joint_pos_rel` (biased) U±0.01, `joint_vel` U±0.5, `base_ang_vel` (imu) U±0.2, `projected_gravity` U±0.05, `last_action` (no noise) = 93 × 10. Plus `goal` (no noise, no history). |
| goal (contents **UNSPECIFIED** in App. A beyond "K=10 future reference sequence"; our default) | For k = 0..9: ref `joint_pos − q0` (29), ref `joint_vel` (29), ref torso position in the robot torso frame (3), relative orientation as 2 rotation-matrix columns (6) = 67 × 10 = 670. Implemented once in `mdp/features.py` (pure torch) and imported by the SuperMimic learner, so both arms see identical features. |
| critic obs | Noise-free proprio (current frame, 93), goal (670), ξ (33), tracked-body pos (42) and ori (84) in the torso frame (BeyondMimic critic terms, `observations.py:44-69`) |

Return-based sampling (App. D.1), implemented in the command:
- The mixture is 50 % uniform over (clip, 1 s bin), 40 % softmax(−25 Ĝ_b), 10 % ∝ EMA of past softmax weights. The frame is uniform within the bin.
- Walk has 4 + 5 = 9 bins; lift has 7 + 7 = 14.
- Ours: `Ĝ_b = G_b / G_max,b` with a **per-bin** `G_max,b = r_max·(1−γ^{H_b})/(1−γ)`, where H_b is the number of steps from the bin start to the clip end and γ = 0.99. The paper's scalar G_max would make end-of-clip bins (short horizon) look bad regardless of policy quality.
- G_b is an EMA (α = 0.01, ours) of the realized discounted return of episodes that started in b. The past-weight EMA also uses α = 0.01 (ours).
- `motion_end` is a time-out, so it does not count as a failure.

---

## 3. Lift task with a box sized to the mocap

### 3.1 Why 0.36 m and not 0.40 m
The integrator's 0.43 / 0.47 m are **palm-site** separations. The hand capsule radius is 0.035 m (`left_hand_collision`, size [0.035, 0.041]), so the surface-to-surface separation is 0.356 / 0.402 m at grasp and 0.27–0.36 m while holding.

FK sweep, using `mj_geomDistance` against a mocap box that follows §3.3. Values are signed distances in m; negative means the reference penetrates the box.

| W (D 0.25, H 0.30) | clip1 at t_g L/R | clip2 at t_g L/R | clip1 descent min (R hand) | clip2 hold max (gap) |
|---|---|---|---|---|
| 0.30 | +0.020 / +0.025 | +0.038 / +0.039 | −0.006 | +0.043 |
| 0.34 | +0.000 / +0.005 | +0.018 / +0.019 | −0.026 | +0.023 |
| **0.36** | **−0.010 / −0.005** | **+0.008 / +0.009** | −0.036 | +0.013 |
| 0.38 | −0.020 / −0.015 | −0.002 / −0.001 | −0.046 | +0.003 |
| 0.40 | −0.030 / −0.025 | −0.012 / −0.011 | −0.056 | −0.007 |

W = 0.36 is the minimax of |gap| at t_g across both clips (≤ 1.0 cm).

Fixed depth and height (W 0.36), with clearances to non-arm bodies:
- **D = 0.20:** torso clearance while holding +0.075 (clip1) / +0.098 (clip2). At D 0.25 it is +0.050 / +0.073; at D 0.30 it is +0.025. Shins clear by ≥ 0.108 during the squat.
- **H = 0.30:** the clip1 palm is 7.7 cm below the top at grasp. With H 0.25 it would be 2.7 cm below.

### 3.2 Box asset
`src/mjlab_g1/assets/box_constants.py::get_box_cfg(size=(0.20,0.36,0.30), mass=1.0, mu=0.8) -> EntityCfg`. It follows the same pattern as `get_toaster_cfg`.

| Property | Value | Why (ours) |
|---|---|---|
| Body | free joint (qpos 7, qvel 6), one box geom `box_geom`, uniform inertia from mass | The world-model state becomes 43 / 41 |
| Mass | 1.0 kg; DR U[0.75, 1.25] | Normal force needed per hand = m·g/(2μ): 4.9 N at μ 1.0 and 10.2 N in the worst case (1.25 kg, μ 0.6). The 0.4–8.7 cm reference squeeze below supplies this through the low-gain PD. |
| Friction | (0.8, 0.005, 0.0001), condim 3, **`priority=1`**; DR μ U[0.6, 1.0] | Hand geoms have μ 1.0, and MuJoCo max-combines at equal priority, so without priority any μ_box < 1 would do nothing. condim 4 is the fallback if the box pitches in M4. |
| Solver (lift scene) | nconmax 256, njmax 1024, ccd_iterations 50, contact_sensor_maxmatch 64 | Copied from the dual-arm task (`dualarm/config/g1/env_cfgs.py:43-46`), which is verified working in this repo |

### 3.3 Reference synthesis (`synth_box`, at 50 Hz, per clip)
1. m(t) = (palm_L + palm_R)/2 and t_g = argmin m_z. That gives clip1 2.17 s and clip2 2.23 s, i.e. about frame 108 / 112.
2. Yaw ψ = yaw(palm_L − palm_R at t_g) − 90°: 5.7° and 6.5°. This stays **fixed**. After grasp the hand-axis yaw drifts only −6.8..+2.0° (clip1) and −5.9..+4.0° (clip2), even though the pelvis yaw drifts −10° / −13°.
3. p(t) = (m_x(t_g), m_y(t_g), H/2) for t < t_g, else p(t_g) + m(t) − m(t_g). This is continuous, and the box stays centred between the palms after grasp.
4. Linear velocity = `np.gradient` (zero before t_g); angular velocity = 0.

Resulting values (mocap world frame, env origin added at runtime):

| | spawn centre | ahead of pelvis at frame 0 / at t_g | end centre z | lift |
|---|---|---|---|---|
| clip1 | (0.464, 0.017, 0.150) | 0.409 / 0.512 m | 0.951 m | 0.80 m |
| clip2 | (0.484, 0.039, 0.150) | 0.427 / 0.522 m | 0.939 m | 0.79 m |

The spawn is collision-free at frame 0: hands +0.29 / +0.30 m, feet ≥ +0.195 m.

**Known reference-vs-box mismatches.** These are features of the data, not bugs, and the policy absorbs them:
- Clip1: the right hand passes 3.6 cm into the box volume during the descent (t ≈ 1.5 s; palm z 0.24 is below the box top).
- Clip2: the reference hands stay 0.8–1.3 cm outside the box for about 0.5 s after t_g while the box reference rises about 10 cm.
- Holding: the reference penetrates 0.4–8.7 cm per side. That penetration is the squeeze.
- Following the box instead of the reference means a 1–4 cm deviation on 2 of the 14 tracked bodies. That costs below 0.5 % of `motion_body_pos` (σ 0.3).

### 3.4 Reset: the box is written from `MultiClipMotionCommand._resample_command`
Reset events run before the command reset and do not know the sampled frame (`manager_based_rl_env.py:558-562` vs `:581`). So after `super()`:
- Compute T_pert = (robot root pose actually written) ∘ (reference root at f)⁻¹.
- **f < t_g − 0.5 s:** box at the world reference, at rest.
- **t_g − 0.5 s ≤ f < t_g:** apply the planar part of T_pert (xy, yaw) to the box, so it stays on the floor and in the same place relative to the robot.
- **f ≥ t_g:** box pose = T_pert ∘ reference; box velocity = reference velocity + Δv_root + Δω_root × r.

Whether post-grasp RSI is allowed depends on the M4 settle test (§6). The fallback is to sample only f ≤ t_g − 0.2 s.

### 3.5 Rewards (added on top of §2; weights and σ are ours, since the paper has no object terms)
None of these uses a contact sensor. Each is an analytic function of (robot FK, box pose, reference), so SuperMimic's reward head can compute it analytically or learn it from the same targets.

| Term | Formula | w | σ |
|---|---|---|---|
| `box_pos` | exp(−‖p_b − p̂_b‖²/σ²) | 1.0 | 0.10 m |
| `box_ori` | exp(−‖log(R̂_b R_bᵀ)‖²/σ²) | 0.5 | 0.4 rad |
| `hands_on_box` | exp(−(d_L² + d_R²)/σ²), gated on frame ≥ t_g − 0.25 s | 0.5 | 0.05 m |

Here d_side is the distance in the box frame from the palm site to the face patch {y = ±(W/2 + 0.035), |x| ≤ D/2, |z| ≤ H/2}. It is one-sided in y: going deeper than the face counts as 0.

With these, r_max = 0.14 per step. Also report the tracking-only normalized return (/0.1) so it is comparable with walk.

### 3.6 Terminations, sensors, success, episode length
- **Terminations:** those of §2 plus `box_pos_error > 0.25 m` (ours, mirroring the 0.25 m pelvis threshold; it catches both drops and kicks).
- **Episode length:** RSI frame to clip end, at most 6.92 s (347 frames). The `motion_end` time-out bootstraps; the clip ends in a static hold.
- **Contact sensors** (mjlab `ContactSensorCfg`, pattern of `dualarm/config/g1/env_cfgs.py:121-191`, fields `found` and `force`). These are used for metrics and success only:

| Sensor | Primary | Secondary |
|---|---|---|
| `left_hand_box`, `right_hand_box` | `{side}_hand_collision`, `{side}_wrist_collision` | `box_geom` |
| `box_floor` | `box_geom` | terrain |
| `body_box` | robot geoms except hand and wrist | `box_geom` |

- **Success (ours).** Evaluated from frame 0. All four must hold:
  1. no termination before the clip end;
  2. ‖p_b − p̂_b‖ ≤ 0.10 m on every step of the final 1.0 s;
  3. box z ≥ 0.80 m at the last frame;
  4. both hand sensors `found` on ≥ 90 % of steps in [t_g + 0.5 s, end].

  Also report the maximum lift height, mean box error and the `body_box` contact fraction.
- **Observations:**
  - Actor: add `box_pose_b` (position in the pelvis frame 3 + 6D rotation 6, noise U±0.02 m / ±0.05, ours) to the 10-frame history. Add the reference box pose relative to the robot torso (9) to each goal frame.
  - Critic: also gets box linear and angular velocity in the pelvis frame (6), box mass and μ (2), and d_L, d_R.
  - The world model gets box state from the contract: `box_qpos [N,7]` and `box_qvel [N,6]` go into s, `box_mass` and `box_mu` go into ξ, and the box reference goes into the reference window.
- **DR:** Table A.3 plus box mass and μ (startup). ξ dimension: D_lift = 35.
- **Optional free ablation:** `Mjlab-G1-Track-Squat-*` is the same cfg without the box (robot-only, within the paper's scope).

---

## 4. Contract implementation

`src/mjlab_g1/envs/tracking_contract_env.py`:

```python
@dataclass
class TrackingContractEnvCfg(ManagerBasedRlEnvCfg):
    auto_reset: bool = False; ref_window_k: int = 10; box_entity: str | None = None

class TrackingContractEnv(ManagerBasedRlEnv):
    xi_names: list[str]            # ["foot_mu","com_dx","com_dy","com_dz", enc_bias x29, (+"box_mass","box_mu")]
    reward_term_names: list[str]; termination_term_names: list[str]
    def read_state(self) -> dict[str, Tensor]:
        # robot_qpos [N,36]: root xyz - env_origins (env-local), quat wxyz, 29 joints
        # robot_qvel [N,35]: root lin vel WORLD, root ang vel WORLD (converted; MuJoCo-native is body frame), 29 joint vel
        # box_qpos [N,7], box_qvel [N,6] (same conventions), last_action [N,29]
        # aux (convenience, not part of s): body_{pos,quat,lin_vel,ang_vel}_w of the 14 tracked bodies, palm_pos_w [N,2,3]
    def read_xi(self) -> Tensor:   # [N,D]: geom_friction[:,foot_gids,0] (assert all equal per env, so 1 value),
                                   # body_ipos[:,torso]-default (sim.get_default_field), robot.data.encoder_bias,
                                   # body_mass[:,box_bid], geom_friction[:,box_gid,0]   -> D = 33 walk / 35 lift
    def read_reference(self, K: int | None = None) -> dict[str, Tensor]:
        # k = 0..K-1, index clamp(time_steps + k, T_c - 1); k = 0 IS THE FRAME THE NEXT REWARD SCORES s_{t+1} AGAINST
        # (mjlab advances the command after rewards: manager_based_rl_env.py:436-456; kept for parity, not "fixed")
        # joint_pos/joint_vel [N,K,29]; body_{pos,quat,lin_vel,ang_vel}_w [N,K,14,·] env-local;
        # box_{pos,quat,lin_vel,ang_vel} [N,K,·]; post_grasp [N,K]; valid [N,K]; clip_id [N]; frame [N]
    def read_step_reward(self) -> tuple[Tensor, Tensor]:   # total [N], terms [N,n] = reward_manager._step_reward * step_dt
    def read_done(self) -> tuple[Tensor, Tensor, Tensor]:  # terminated, truncated (time_out | motion_end), per-term flags [N,n]
    def write_state(self, env_ids, robot_qpos, robot_qvel, box_qpos=None, box_qvel=None,
                    clip_ids=None, frames=None, last_action=None) -> None:
        # robot.write_root_state_to_sim([pos+origin, quat, lin_w, ang_w]); write_joint_state_to_sim; box likewise
        # command.clip_ids/time_steps[ids] = ...; action_manager prev/current action[ids] = last_action (or 0)
        # observation_manager.reset(ids) (clears history); sim.forward(); command.update_relative_body_poses()
        # all writes IN PLACE (CUDA graphs keep pointers, MJ/sim/sim.py:161-180)
```

`src/mjlab_g1/rl/contract_vecenv.py::ContractVecEnv(mjlab RslRlVecEnvWrapper)`. **Both learners go through it**, so there is only one reset path:
- `step(a)`:
  1. `env.step(a)` with auto_reset off.
  2. Fill extras: `raw_next = read_state()` (the true terminal state for done envs), `reward_terms`, `terminated`, `time_outs`, `ref_index`.
  3. For `done_ids`: `env.reset(env_ids=done_ids)`, store `extras["raw_reset"]`, take the obs returned by reset, and merge the reset `extras["log"]` for the rsl_rl Logger.
- `extras["time_outs"]` is always set, so PPO bootstraps at `motion_end`.
- `unwrapped` returns `env.unwrapped`.
- Static `xi_names` and `reward_term_names` go in `extras["meta"]` on the first call.

Shared pure-torch features live in `src/mjlab_g1/tasks/tracking/mdp/features.py`:
- `proprio(raw, xi, last_action)`
- `goal(raw, ref)`
- `box_obs(raw)`
- `relative_body_poses(ref, robot_anchor)`, a torch copy of `commands.py:377-405` that the SuperMimic analytic reward head needs.

The PPO observation terms are thin wrappers around these.

---

## 5. PPO baseline

**Config.** Start from `MJ/tasks/tracking/config/g1/rl_cfg.py:10-46` with mjlab's `RslRlOnPolicyRunnerCfg`, which has `upload_model` and `obs_groups={"actor":("actor",),"critic":("critic",)}`.

| | Value |
|---|---|
| runner | `MjlabOnPolicyRunner` (strips the cnn/rnn keys, `MJ/rl/runner.py:23-31`); `class_name` "OnPolicyRunner"; algorithm stock vendored `PPO` |
| actor / critic | MLP (512, 256, 128), ELU, obs normalization on; Gaussian with scalar std, init 1.0 |
| PPO | clip 0.2, value coef 1.0, clipped value loss, entropy 0.005, 5 epochs × 4 minibatches, lr 1e-3 adaptive (desired_kl 0.01), γ 0.99, λ 0.95, max_grad_norm 1.0 |
| rollout | 8192 envs (paper §4.1 PPO-Sim) × 24 steps = 196,608 transitions per iteration; minibatch 49,152 |
| length | 30,000 iterations = 5.9e9 env steps; save_interval 500 |
| misc | `clip_actions=None` (required for reachability); `upload_model=False`; logger tensorboard; seed per run; `init_at_random_ep_len=True` |
| observations | §2 / §3 "supermimic" set, identical to SuperMimic's policy inputs |
| variants (optional) | `PPO-SM`: size-matched actor MLP (2048,1024,512,256), a flat stand-in for the Table A.4 encoder/decoder (ours). `PPO-BM`: mjlab stock 160-d observations, walk only, 1 seed, as an env sanity check. |

**Budget.**
- Wall-clock is an **estimate to be measured in M1**. At 1.3–2.6e5 steps/s (paper Fig. 5a gives 1.32e5 at 4096 envs), 30k iterations take about 6–12 h on one H100 or RTX PRO 6000. The lift env is slower (box plus 256/1024 contact buffers).
- Every iteration logs `env_steps`, `wall_clock_s` and `gpu_type`.
- Both learners are capped at the same wall-clock (≤ 23.5 h, `short`) on the same GPU type.

**Launch.**
- Use the user's `train.py` after the integration-report changes: base cfg types, env factory, assert relaxed to mjlab's `RslRlBaseRunnerCfg`, YAML dump before construction.
- mjlab's own `python -m mjlab.scripts.train` cannot see the `mjlab_g1` registry.

**Checkpoints and evaluation.**
- Checkpoints: `logs/rsl_rl/g1_track_{walk,lift}/<ts>_<run>/model_<it>.pt` plus `params/{env,agent}.yaml`.
- `scripts/eval_tracking.py` is learner-agnostic. It loads the task's `runner_cls`, then calls `load` and `get_inference_policy`.
- Eval settings: `sampling_mode="start"` per clip, observation noise on, DR from the training distribution, 1024 envs, one episode each.
- Metrics: normalized return Σr / (T_c·r_max), failure rate (terminated before clip end), MPJPE/MPKPE and joint-velocity RMS (`MJ/tasks/tracking/mdp/metrics.py:13-79`), plus the lift metrics of §3.6. Output is JSON next to the checkpoint.

---

## 6. File plan, CLI, smoke tests

**New files:**
```
scripts/npy_to_motion_npz.py                         # §1 (load_model, resample, velocities, fk, synth_box, main)
scripts/eval_tracking.py                             # eval(task_id, checkpoint, num_envs, out_json) -> dict
scripts/smoke_tracking_env.py                        # GPU smoke tests below
src/mjlab_g1/assets/box_constants.py                 # get_box_cfg(size, mass, mu) -> EntityCfg
src/mjlab_g1/envs/tracking_contract_env.py           # TrackingContractEnvCfg, TrackingContractEnv (§4)
src/mjlab_g1/rl/contract_vecenv.py                   # ContractVecEnv
src/mjlab_g1/tasks/tracking/__init__.py
src/mjlab_g1/tasks/tracking/tracking_env_cfg.py      # make_g1_track_env_cfg(task, motion_files, obs_set="supermimic", play=False)
src/mjlab_g1/tasks/tracking/mdp/{__init__,commands,observations,rewards,terminations,features}.py
    # commands: MultiClipMotionCommand(Cfg): multi-npz pad+lengths, "return" sampling, end_mode="truncate", box write (§3.4)
    # rewards: box_pos, box_ori, hands_on_box ; terminations: pelvis_height, pelvis_tilt, motion_end, box_pos_error
src/mjlab_g1/tasks/tracking/config/g1/{__init__,env_cfgs,rl_cfg}.py   # registers Walk/Lift x {PPO, SuperMimic} (+PPO-BM, Squat)
tests/tracking/test_convert.py, test_box_ref.py, test_features.py      # CPU
dataset/tracking/{walk,lift}/*.npz                   # generated
```

**CLI** (requires a one-time `uv sync`, which needs the user's approval first):
```bash
for f in dataset/locomotion/*.npy; do PYTHONPATH=src uv run --no-sync python scripts/npy_to_motion_npz.py \
  --input $f --output dataset/tracking/walk/$(basename ${f%.npy}).npz --robot mjlab_g1; done
for f in dataset/dualarm/*.npy; do PYTHONPATH=src uv run --no-sync python scripts/npy_to_motion_npz.py \
  --input $f --output dataset/tracking/lift/$(basename ${f%.npy}).npz --robot mjlab_g1 --box 0.20 0.36 0.30; done
PYTHONPATH=src uv run --no-sync python -m pytest tests/tracking -q                       # CPU
/cm/shared/apps/slurm/current/bin/sbatch --account=mmaghelih --partition=quick --gres=gpu:rtx_pro_6000_b:1 \
  --time=00:30:00 --wrap "cd $PWD && PYTHONPATH=src uv run --no-sync python scripts/smoke_tracking_env.py --task Mjlab-G1-Track-Lift-PPO --num-envs 64"
/cm/shared/apps/slurm/current/bin/sbatch --account=mmaghelih --partition=short --gres=gpu:H100:1 --time=23:30:00 \
  --export=ALL,TASK=Mjlab-G1-Track-Walk-PPO,NUM_ENVS=8192,MAX_ITERS=30000,EXP_NAME=g1_track_walk,RUN_NAME=ppo_s0,INIT_RANDOM=True,SKIP_SYNC=1 \
  scripts/train_slurm.sbatch
PYTHONPATH=src uv run --no-sync python scripts/eval_tracking.py Mjlab-G1-Track-Walk-PPO \
  --checkpoint-file logs/rsl_rl/g1_track_walk/<run>/model_30000.pt --num-envs 1024
```

**Smoke tests:**

| Milestone | Test | Pass |
|---|---|---|
| M0 CPU | V1–V4; `synth_box` continuous at t_g (\|Δp\| < 1e-6), end z 0.951 / 0.939; `features.py` shapes (walk actor 930 + 670) | exact or ±0.01 |
| M0 GPU, 64 envs, quick | V5 replay; ξ foot friction shared, so D = 33 / 35; `write_state(read_state())` then the same action gives the same next state | < 1e-3 |
| | Σ `reward_terms` = reward; terminated ∧ truncated = ∅; `motion_end` fires at T_c − 1; frame increments by exactly 1 within an episode (no teleport) | exact |
| | Open-loop reference-PD replay a = (q_ref − q0)/α from frame 0: record per-step reward/r_max and the step of first termination | logged baseline, no threshold |
| M1 PPO walk | 300 iterations at 4096 envs on quick: no NaN, episode length rising, iterations/s measured (sets the wall-clock budget); then the full run on short | Gate (ours): eval failure rate < 10 % on both walk clips at the end, before any SuperMimic comparison |
| M4 lift env | (a) Reset at frame 0: zero `body_box`/hand contacts; box static (< 1 cm) for 50 reference-PD steps | yes |
| | (b) **Settle test:** reset at every f ≥ t_g − 0.5 s, 10 reference-PD steps | p95 box error < 5 cm and box speed < 1 m/s, else restrict RSI to f ≤ t_g − 0.2 s |
| | (c) On a hand–box contact, `sim.data.contact` friction equals μ_box (mjwarp's handling of `geom_priority` is unverified) | equal, else set hand-geom μ equal to the sampled μ_box |
| | (d) At 8192 envs: max `nacon` < 256, no njmax overflow | yes |
| M4 PPO lift | 500 iterations: no NaN; `hands_on_box` and `box_pos` episode terms rising; then the full run | success metric (§3.6) logged per eval |

AMP `Mjlab-G1-DualArm` (toaster, user's robot) is reported only as a separate reference row. It optimizes a different objective on a different robot asset and a different object.