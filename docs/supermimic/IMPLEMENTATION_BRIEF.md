# SuperMimic side-by-side: implementation brief (2026-10-22)

Goal: implement SuperMimic (Liao et al., arXiv 2610.28480, CoRL 2026; no public code) and a
PPO baseline on ONE shared mjlab tracking env, for (1) walking and (2) box lifting, then compare.
Design documents this brief condenses (read them for detail; this brief wins on conflicts):
`docs/supermimic/design_learner.md`, `docs/supermimic/design_protocol.md`,
`docs/supermimic/paper_spec.md`, `docs/supermimic/mjlab_tracking.md`.

## 0. Global decisions (fixed)

| Topic | Decision |
|---|---|
| Robot | **mjlab's G1**: `mjlab.asset_zoo.robots.get_g1_robot_cfg()`, 30 bodies, default pose = its KNEES_BENT (`robot.data.default_joint_pos`), action scale `G1_ACTION_SCALE`, `use_default_offset=True`, **no action clip**. Not the repo's `g1_realsense.xml`. MJCF: `mjlab/asset_zoo/robots/unitree_g1/xmls/g1.xml`. |
| Motions | `dataset/tracking/locomotion1.npz` (walk, 199 frames @50 Hz) and `dataset/tracking/dual_arm1_box.npz` (lift, 520 frames; the dual-arm clips are treated as **20 fps** capture, user decision). One clip per task. Format = mjlab MotionCommand npz (+ extra keys for the box, see §3). |
| Control | dt 0.02 (timestep 0.005 × decimation 4), as mjlab tracking. |
| Episodes | Start frame from mjlab's **adaptive** sampler for BOTH learners (paper uses return-based sampling: documented deviation, keeps the comparison fair). Episode ends at clip end as a **truncation** (time_out=True), never by wrapping the clip. |
| Terminations | mjlab `anchor_pos` (z err > 0.25) + `anchor_ori` (> 0.8), anchor = `torso_link` (mjlab/BeyondMimic; paper text says pelvis: documented). Remove `ee_body_pos`. Lift adds `box_pos` (box pos err > 0.3 m). |
| Reward | mjlab tracking terms, weights 1/1/1/1 body pos/ori/linvel/angvel, 0.5/0.5 anchor pos/ori, action_rate −0.1, **joint_limit −2**, **no self_collisions** (paper Table A.2). dt-scaled (mjlab default). Lift adds `box_pos` exp(−‖e‖²/0.1²) w 1.0 and `box_ori` exp(−θ²/0.4²) w 0.5 (ours). No contact-force terms (must be analytic from state). |
| DR | `base_com`, `encoder_bias`, `foot_friction` exactly as mjlab ranges but **mode="reset"** (per episode, paper). **No push_robot.** Lift: box mass U(0.8,1.2)×nominal, box friction U(0.8,1.2)×nominal, also mode="reset". |
| Reset noise | keep mjlab pose_range / velocity_range / joint_position_range. |
| PPO baseline | vendored `rsl_rl` OnPolicyRunner + PPO (no AMP), BeyondMimic hparams (mjlab `tasks/tracking/config/g1/rl_cfg.py`), 8192 envs, actor MLP (2048,1024,512,256) ELU, critic MLP (256,128) ELU (paper: PPO-Sim shares network sizes), obs normalization on, init_std 1.0. Same env, same obs as SuperMimic's actor (history + goal). |
| SuperMimic | per `design_learner.md`, with the overrides in §4. |
| Code location | env: `src/mjlab_g1/tasks/smtrack/` + `src/mjlab_g1/envs/smtrack_env.py`; learner: `src/mjlab_g1/rl/supermimic/`; scripts: `src/mjlab_g1/scripts/{train_track.py, eval_track.py, make_box_reference.py}`; Slurm: `scripts/train_track.sbatch`; tests: `tests/supermimic/`. **Do not modify the AMP code** (`rsl_rl/rsl_rl/algorithms/amp_ppo.py`, dual-arm task, `train.py`). |
| Logging | rsl_rl Logger with `logger="tensorboard"`; SuperMimic also appends one JSON line per log interval to `<log_dir>/metrics.jsonl`. |

## 1. Shared state conventions (learner + env must agree exactly)

All tensors float32, batch-first, quaternions **wxyz**.

- `qpos [N,36]` = `[root_pos (env-local: world − env_origin) 3, root_quat_w 4, joint_pos 29]` — read
  from raw `sim.data.qpos` via `robot.indexing.free_joint_q_adr` / `joint_q_adr` (not derived
  quantities, which lag by one substep).
- `qvel [N,35]` = `[root_lin_vel_w 3, root_ang_vel_w 3, joint_vel 29]`. MuJoCo's free-joint
  `qvel[3:6]` is **body-local**; convert: `ang_vel_w = quat_apply(root_quat_w, qvel_local[3:6])`.
- Lift: `box_qpos [N,7]` (env-local pos + quat), `box_qvel [N,6]` (lin w, ang w; same conversion).
- World-model state `s` = `cat(qpos, qvel[, box_qpos, box_qvel])`: 71 (walk) / 84 (lift).
- `xi [N,D]` raw DR context: `[foot_friction (1), encoder_bias (29), torso_com_offset (3)]`
  (+ lift `[box_mass_scale (1), box_friction_scale (1)]`) = 33 / 35, with `xi_lo, xi_hi [D]` from
  the env's DR cfg. Encoding `2(ξ−lo)/(hi−lo)−1` (0 where hi==lo).
- `ref_index [N]` long = the motion frame the reward at the next step is computed against
  (mjlab pairing: reward/termination of r_{t+1} use the **pre-advance** frame; the command advances
  after reset). `reward_ref_offset = 0` under this convention; parity tests decide.
- Base angular velocity observation = gyro at site `imu_in_pelvis` = `R_pelvisᵀ ω_w` (site frame =
  pelvis frame; verify the site has no rotation in g1.xml).

## 2. Env contract — `SMTrackEnv` (subclass of mjlab `ManagerBasedRlEnv`)

```python
class SMTrackEnv(ManagerBasedRlEnv):
  # raw state after the last step, BEFORE auto-reset (terminal state for done envs)
  pre_reset_state: torch.Tensor        # [N,71|84], valid for all envs every step
  def get_state(self) -> torch.Tensor  # [N,71|84] current (post-reset) state, exact (raw qpos/qvel)
  def get_xi(self) -> torch.Tensor     # [N,33|35] raw DR context of the current episode
  xi_lo: torch.Tensor; xi_hi: torch.Tensor          # [D]
  def get_ref_index(self) -> torch.Tensor          # [N] long, current command time_step
  motion_lib: MotionLibrary            # shared tensors (see §3), same object the learner uses
  reward_term_names: list[str]; def get_reward_terms(self) -> torch.Tensor  # [N,T] last step, dt-scaled
  env_origins: torch.Tensor            # [N,3]
  has_box: bool
```
- `pre_reset_state` is captured inside an override of `_reset_idx` (before resetting) and copied
  for non-reset envs at the end of `step`, so it always equals s_{t+1} as produced by physics.
- `extras["time_outs"]` from the RSL-RL wrapper marks truncations (clip end / time limit).
- Observation groups: `actor_hist` (per-frame terms, `history_length=10`, flattened, noise on):
  `joint_pos_rel(biased)` 29 ±0.01, `joint_vel_rel` 29 ±0.5, `base_ang_vel` 3 ±0.2,
  `projected_gravity` 3 ±0.05, `last_action` 29 (lift +`box_pos_b` 3 ±0.01, `box_rot6d_b` 6 ±0.02)
  → 930 / 1020; `actor_goal` (no history, no noise): K=10 future frames
  `[q_ref 29, q̇_ref 29, rot6d(R_robot_anchorᵀ R_ref_anchor) 6]` (lift +`[box_ref_pos rel 3, rot6d 6]`)
  → 640 / 730; `critic` (no noise): mjlab critic terms + goal (+ box state). The goal and box obs
  functions call the SAME torch functions as the learner (`supermimic/features.py`) on env tensors.

## 3. Motion library + box reference

- `MotionLibrary(npz_path, device)`: `joint_pos/joint_vel [F,29]`, `body_pos_w/quat_w/lin_vel_w/
  ang_vel_w [F,30,·]` (all 30 bodies, mjlab order), optional `box_pos_w [F,3]`, `box_quat_w [F,4]`,
  `box_lin_vel_w`, `box_ang_vel_w`, `grasp_frame` (int), `num_frames F`; `frame(k)` gathers with
  clamp to `F−1`; `alive(k) = k <= F−1`.
- `make_box_reference.py` (CPU, mujoco FK on mjlab g1.xml): palm sites `left_palm/right_palm` from
  reference qpos; grasp frame g = first frame where palm-midpoint z is within 1 cm of its minimum
  (in the first half of the clip); box half-extents (0.15, 0.19, 0.125) m (width 0.38 m across the
  hands, matched to the 0.36–0.40 m palm separation while holding); box ref for k<g = spawn pose
  (xy = palm midpoint at g, z = 0.125, yaw = pelvis yaw at g); for k≥g the box keeps the grasp-time
  offset to the palm midpoint (position) and rotates with the pelvis yaw change since g;
  velocities by finite difference. Writes `dataset/tracking/dual_arm1_box.npz` (all original keys +
  box keys + `grasp_frame`). Print palm separation at g and while holding as a sanity check.
- Box entity: MJCF free body, one box geom (condim 3, friction 1.0, solref default), mass 1.5 kg.
  Resets put the box at `box_ref[k_start]` with zero velocity (in hand if k_start ≥ g).

## 4. SuperMimic overrides to `design_learner.md`

- TorchFK built from **mjlab's g1.xml** (30 bodies), with sites `left_palm`, `right_palm`,
  `imu_in_pelvis`. Tracked bodies = mjlab G1 `motion_cmd.body_names` (14), anchor `torso_link`.
- Reset sampling: the env's mjlab adaptive sampler (no `ReturnBasedResetSampler` in v1).
- `max_iterations` configurable (default 320_000); cosine LR schedule length = `0.78 × max_iterations`
  (paper ratio 250k/320k) so shorter runs keep the paper's schedule shape.
- `num_envs` 4096 for SuperMimic (paper), 8192 for PPO (paper's PPO-Sim).
- Mandatory startup parity checks (reward, termination, obs) on real transitions; abort with a
  clear message if `max|r_analytic − r_env| > 1e-3` (relaxed from 1e-4; float32 FK).
- Checkpoint every `save_interval` iterations; `metrics.jsonl` every `log_interval`.
- CPU-runnable smoke mode: `--smoke` sets N=8, hidden widths ÷16, H=4, k=2, batch 16, 3 iterations.

## 5. Scripts

- `train_track.py TASK --learner {ppo,supermimic} [--num-envs N] [--max-iterations M] [--seed S]
  [--run-name R] [--device cuda:0|cpu] [--smoke]` → `logs/smtrack/<task>/<learner>/<date>_<run>/`.
  Task ids: `G1-SMTrack-Walk`, `G1-SMTrack-Lift` (own small registry in `tasks/smtrack/__init__.py`).
- `eval_track.py --task T --learner L --checkpoint P [--episodes 64] [--device]` → JSON with:
  success rate (no termination through clip end, from frame 0, no DR, no obs noise, foot friction 1.0),
  normalized return (undiscounted reward sum / max possible), mean/max body position error (MPKPE,
  14 bodies, global and anchor-relative), joint pos error, peak joint speed, fall rate;
  lift: box success (box z ≥ box_ref z − 0.1 at clip end and no termination), box pos error,
  peak box vertical speed, box tilt at end, time from +5 cm to +55 cm box rise.
- `scripts/train_track.sbatch`: env vars TASK LEARNER SEED NUM_ENVS MAX_ITERS RUN_NAME; partition
  `short`, `--gres=gpu:1` (any of H100 / rtx_pro_6000_b / A100 / H200), 24 h, 16 CPUs, 128G;
  `GIT_PYTHON_REFRESH=quiet MUJOCO_GL=disable WANDB_MODE=disabled`; no `uv sync` (venv exists);
  `uv run --no-sync`.

## 6. Tests (`tests/supermimic/`, CPU, pytest)

Everything in `design_learner.md` §8 that applies, plus: env contract smoke (build SMTrackEnv walk
and lift on CPU with N=4, step 20 times, check shapes, `pre_reset_state` on a forced termination,
xi ranges), box reference sanity, and an end-to-end `train_track.py --smoke` for both learners on
both tasks.
