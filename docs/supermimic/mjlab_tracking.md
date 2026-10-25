# mjlab v1.4.0 motion-tracking task: a PPO tracking baseline and the env a SuperMimic world model would train against

**Path prefixes used below**
- `MJ/` = `<mjlab v1.4.0>/src/mjlab/`
- `MJROOT/` = the same checkout's root (`.../4360096/`)
- `REPO/` = `<repo>/`

**Sources for paper claims.** I had only the prior summary in `core_methods.json` (entry `supermimic`), not the PDF. Every SuperMimic claim below is attributed to that summary. I cite no section or equation numbers. Where the summary is silent I write UNSPECIFIED. The summary itself points to the paper's Appendix Tables A.2–A.5 for the full hyperparameters.

**How I checked.** Everything was read-only. I made no edits, ran no `uv sync`, no training and no job submissions. I computed the clip statistics and forward kinematics on the login node with throw-away environments: `uv run --offline --no-project --with numpy [--with mujoco==3.10.0] python -I <script>`. The scripts are in `<scratch>/tools/` (`npy_stats.py`, `limits.py`, `fk.py`, `grasp.py`, `xmltree.py`).

---

## 0. Headline findings

1. **The env is BeyondMimic-shaped, close to what SuperMimic uses.** Task IDs are `Mjlab-Tracking-Flat-Unitree-G1` and `...-No-State-Estimation` (`MJ/tasks/tracking/config/g1/__init__.py:7-21`). The reward, std values, the three DR terms and the 1 s sampling bins are all there. The summary says SuperMimic changes the following:
   - joint-limit weight −2 instead of −10
   - no self-collision penalty
   - no pushes
   - return-based bin sampling instead of failure-based
   - K=10 future reference frames in the observation, where mjlab gives the policy one frame
2. **Default `num_envs` is 1** (`MJ/tasks/tracking/tracking_env_cfg.py:286`, not overridden in `env_cfgs.py`). You must pass `--env.scene.num-envs 4096`, as mjlab's own docs do (`MJROOT/docs/source/training/motion_imitation.rst:55-57`).
3. **The user's `train.py` cannot launch it.** It uses its own registry, asserts an AMP config and only builds the two custom env classes. Launch with `python -m mjlab.scripts.train` instead. `uv run train` is ambiguous because both `mjlab` and `mjlab_g1` define a `train` console script.
4. **A motion `.npz` only works with the robot model it was generated from.** `csv_to_npz` hard-codes mjlab's 30-body G1. The user's `g1_realsense.xml` has 31 bodies (an extra `realsense_d435_link` under the torso), which shifts every arm body index by +1. I checked this with MuJoCo FK.
5. **The `fps` key in the npz is never read.** The command advances one frame per env step, so the npz must be at exactly 50 Hz (0.005 s × 4).
6. **When a clip ends mid-episode, the robot is teleported without any reset signal.** Episodes are 10 s, but the walking clips are about 4 s and the lift clips 6.93 s. A world-model replay buffer has to mask these transitions, or the episode length should be set to the clip length.
7. **The dual-arm clips go past the hard waist-pitch limit:** 0.526–0.528 rad against a hard limit of 0.52 and a soft limit of 0.468. That happens on 61–69 frames out of 209. Reset clips the pose to the soft limits, and the −10 joint-limit penalty fires during tracking.
8. **Tracking the clips' wrists exactly will not grasp the toaster.**
   - Mocap palm separation is 0.43 / 0.47 m at grasp and 0.355–0.395 m while holding.
   - Touching the toaster's grasp markers needs about 0.26 m.
   - So a lifting variant must edit or relax the arm reference, or use a wider box.
9. **World-model state dimensions:** G1 has qpos 36 / qvel 35. With the toaster it is 46 / 44: a free joint plus 3 cosmetic internal joints.

---

## 1. Task IDs, configs, PPO hyperparameters and launching

### 1.1 Registered tasks
These are registered into **mjlab's** registry (`MJ/tasks/registry.py:19,22-40`):

| Task ID | env cfg | Runner |
|---|---|---|
| `Mjlab-Tracking-Flat-Unitree-G1` | `unitree_g1_flat_tracking_env_cfg()`; play variant `(play=True)` | `MotionTrackingOnPolicyRunner` (`MJ/tasks/tracking/config/g1/__init__.py:7-13`) |
| `Mjlab-Tracking-Flat-Unitree-G1-No-State-Estimation` | `(has_state_estimation=False)`: drops `motion_anchor_pos_b` and `base_lin_vel` from the actor (`env_cfgs.py:74-84`) | same (`__init__.py:15-21`) |

### 1.2 Env cfg
The base is `make_tracking_env_cfg()` in `MJ/tasks/tracking/tracking_env_cfg.py`. The G1 overrides are in `MJ/tasks/tracking/config/g1/env_cfgs.py`. It is a re-implementation of BeyondMimic at commit `f8e20c8` (`tracking_env_cfg.py:6-9`).

| Item | Value | Cite |
|---|---|---|
| Physics dt / decimation / control rate | 0.005 s / 4 / 50 Hz | `tracking_env_cfg.py:306,311` |
| Solver | iterations 10, ls_iterations 20 | `:307-308` |
| nconmax / njmax | 35 / 250 (too small once a box is added, see §6) | `:303-304` |
| Episode length | 10 s (500 steps) | `:312` |
| Terrain / num_envs | plane / **1** | `:286` |
| Robot | mjlab asset G1 `get_g1_robot_cfg()` | `env_cfgs.py:22` |
| Sensors | `self_collision` contact sensor, pelvis subtree vs itself, history 4 | `env_cfgs.py:24-33` |
| Action | `JointPositionActionCfg`, all actuators, `use_default_offset=True`, scale `G1_ACTION_SCALE` | `tracking_env_cfg.py:126-133`, `env_cfgs.py:35-37` |
| Anchor body / tracked bodies | `torso_link` / 14 bodies: pelvis first, legs, torso, shoulders, elbows, wrist_yaw links | `env_cfgs.py:41-57` |
| Rewards scaled by dt | yes (`scale_rewards_by_dt=True`) | `MJ/envs/manager_based_rl_env.py:157`, `MJ/managers/reward_manager.py:116-133` |

### 1.3 PPO cfg (`MJ/tasks/tracking/config/g1/rl_cfg.py:10-46`, defaults from `MJ/rl/config.py`)

| Setting | Value |
|---|---|
| actor / critic | MLP (512, 256, 128), ELU, obs normalization on for both |
| action distribution | Gaussian, init_std 1.0, scalar std |
| PPO | clip 0.2, value_loss_coef 1.0, clipped value loss, entropy 0.005, 5 epochs × 4 minibatches, lr 1e-3 adaptive (desired_kl 0.01), γ 0.99, λ 0.95, max_grad_norm 1.0 |
| rollout | 24 steps/env, max_iterations 30 000, save every 500 iterations, experiment `g1_tracking` |
| base defaults | seed 42, `clip_actions=None` (no action clipping), logger `wandb`, `upload_model=True` (`MJ/rl/config.py:86-125`) |

### 1.4 Launching with mjlab's own scripts
- **Train** with `MJ/scripts/train.py`. For tracking tasks it requires an existing local `--env.commands.motion.motion-file` or a W&B `--registry-name` (`train.py:73-100`). It hard-codes `init_at_random_ep_len=True` (`:175-177`), and `video` defaults to False (`:30`), so no GL is needed.
- **Play** with `MJ/scripts/play.py`. It accepts a local `--motion-file` (`play.py:91-93`). Without a display it falls back to the web viewer `viser` (`play.py:283-293`).
- **Evaluate** with `MJ/tasks/tracking/scripts/evaluate.py`. It **requires W&B**: it pulls both the motion artifact and the checkpoint from a run (`evaluate.py:63-69,80-83`). Offline, you need a local copy of its loop (§3.9).

The command I suggest, from `REPO/` once a venv exists. The `uv` and the logger flags are ours.

```bash
WANDB_MODE=offline uv run python -m mjlab.scripts.train Mjlab-Tracking-Flat-Unitree-G1 \
  --env.commands.motion.motion-file dataset/tracking/<clip>.npz \
  --env.scene.num-envs 4096 --agent.max-iterations <N> \
  --agent.logger tensorboard --agent.upload-model False --agent.run-name <name>
```

Why `python -m`: `REPO/pyproject.toml:24-25` and `MJROOT/pyproject.toml:60-61` both define `train` and `play` console scripts, so the bare name depends on install order. Running from the user's venv works:
- mjlab is a git dependency.
- mjlab requires `rsl-rl-lib==5.2.0` (`MJROOT/pyproject.toml:49`), which the user's repo satisfies with the vendored editable 5.2.0 (`REPO/pyproject.toml:41`, `REPO/rsl_rl/pyproject.toml`).
- Its `ppo.py` and `on_policy_runner.py` contain no AMP references, so `MjlabOnPolicyRunner` builds on stock 5.2 PPO.
- `import mjlab` loads the `mjlab.tasks` entry point (`MJ/__init__.py:36-52`). That only imports `mjlab_g1.tasks`, which registers lazily into its own registry, so there is no clash.

### 1.5 Can the user's `train.py` launch it? No, not without edits.
- It reads `mjlab_g1.tasks.registry` (`REPO/src/mjlab_g1/scripts/train.py:22`). That is a separate `_REGISTRY` (`REPO/src/mjlab_g1/tasks/registry.py:30`), and the user's tasks register into it (`REPO/src/mjlab_g1/tasks/dualarm/config/g1/__init__.py:1`), so the two registries are disjoint.
- `TrainConfig.env` is typed as the union of the two custom cfgs (`:38`), and `from_task` asserts `RslRlAMPOnPolicyRunnerCfg` (`:61`).
- `run_train` builds only `G1LocomotionManagerBasedRlEnv` or `G1DualarmManagerBasedRlEnv` and raises `TypeError` for anything else (`:98-111`).
- There is no motion-file handling.

**Suggestion (ours):** use mjlab's trainer for the PPO baseline. Write the SuperMimic trainer as its own script that constructs `mjlab.envs.ManagerBasedRlEnv(cfg)` directly.

---

## 2. Motion file format and how to convert the user's clips

### 2.1 What `MotionCommand` loads (`MJ/tasks/tracking/mdp/commands.py:36-60`)
The file is loaded with `np.load`. Only these six keys are read. Extra keys are ignored, so object data can be added to the same file (§6).

| key | shape | content |
|---|---|---|
| `joint_pos` | [T, 29] | joint angles in the entity's joint order |
| `joint_vel` | [T, 29] | joint velocities |
| `body_pos_w` | [T, B, 3] | **all** robot bodies, world frame, without env origin (it is added at runtime, `:137-141,155-160`) |
| `body_quat_w` | [T, B, 4] | wxyz |
| `body_lin_vel_w` | [T, B, 3] | world frame |
| `body_ang_vel_w` | [T, B, 3] | world frame |

Here B is the number of bodies in the training robot entity, excluding the world body (`MJ/entity/entity.py:418-420`): 30 for mjlab's G1, 31 for the user's. The loader slices `[:, body_indexes]` (`commands.py:55-59`), using indexes from `robot.find_bodies(cfg.body_names, preserve_order=True)` on the **training** robot (`:75-79`). Two consequences:
- Body order must match the model used to generate the file. mjlab's docs warn about exactly this (`MJROOT/docs/source/training/motion_imitation.rst:40-48`).
- `cfg.body_names[0]` must be `pelvis`, because the reset uses `body_*_w[:, 0]` as the root (`commands.py:328-331`).

There is no fps field in use. The command advances one frame per env step (`commands.py:407-408`), and the bins use `1/env.step_dt` (`:93`). The csv converter writes an `fps` key (`MJ/scripts/csv_to_npz.py:207`) that nothing reads. **The npz must be at 50 Hz.**

### 2.2 The converter, `MJ/scripts/csv_to_npz.py`
- **Input CSV format.** Each row is `x, y, z, qx, qy, qz, qw, 29 joints`, comma-separated. The quaternion is reordered from xyzw to wxyz at `:59-64`. The docs give the same convention (`motion_imitation.rst:22-24`). The joint columns are matched by name to a fixed list in the standard Unitree order (`:394-424`, mapped through `robot.find_joints(..., preserve_order=True)` at `:204`).
- **Resampling** (`:69-121`). It builds `times = arange(0, (N-1)/in_fps, 1/out_fps)`, so the last input frame is dropped. Root position and joints are linearly interpolated, and the root quaternion is slerped. CLI defaults are input 30 fps, output 50 fps (`:339-347`).
- **Velocities** (`:123-151`).
  - Joint and root linear velocity come from `torch.gradient` at 50 Hz.
  - Root angular velocity is `axis_angle(q_{t+1} q_{t-1}^{-1}) / 2dt`, which is world frame.
- **Kinematic replay, no physics stepping** (`:247-277`).
  - Each frame it writes the root state (world-frame velocities) and the joint state, then calls `sim.forward()` and logs `robot.data.body_link_*`.
  - Body velocities therefore come from MuJoCo's `cvel` FK (`MJ/entity/data.py:20-31,291-298`) and are consistent with qvel.
  - It asserts that the pelvis velocity matches the written root velocity (`:279-284`).
- **Output and W&B.** It saves to a **hard-coded `/tmp/motion.npz`** (`:308-309`), then unconditionally calls `wandb.init`, `log_artifact` and `link_artifact` into the `motions` registry (`:311-325`). The npz is written before the W&B calls, so an offline failure still leaves the file. The `output_name` argument is the W&B collection name, not a path, whatever the docstring says.
- **Model.** It builds the scene from `unitree_g1_flat_tracking_env_cfg().scene` (`:366`), which is always mjlab's G1.
- **GPU and GL.** The default device is `cuda:0`, with a fallback to CPU when CUDA is missing (`:359-361`). Whether mujoco_warp runs the whole loop on CPU is unverified. GL is needed only for `--render True` (OffscreenRenderer at `:373-389`, then `mujoco.Renderer`). Importing the module needs no GL, so ignore the docs' `MUJOCO_GL=egl`. That setting is only for rendering, and rendering is impossible on this cluster anyway.

### 2.3 Does the user's data match?
- **Quaternion convention.** The user's npy files are wxyz. Frame 0 is about `[1.0, −0.002, −0.006, 0.002]`, mean |w| is 0.97–0.998, and there are no sign flips between consecutive frames.
- **Layout.** `[root pos 3, quat wxyz 4, 29 joints]`, as the AMP loader comment states (`REPO/rsl_rl/rsl_rl/utils/motion_loader_g1.py:131-145`).
- **Joint names and order.** They are identical across mjlab `g1.xml`, the user's `g1_realsense.xml` and csv_to_npz's list: 29 hinges in the same order with the same ranges (I parsed both XMLs).
- **Body sets differ.** The user's model adds `realsense_d435_link` (0.072 kg) under `torso_link` (`REPO/src/mjlab_g1/assets/g1/g1_realsense.xml`, the diff around line 205). FK confirms the tracked-body indices:
  - user: `[0,2,4,6,8,10,12,15,18,20,23,25,27,30]`
  - mjlab: `[0,2,4,6,8,10,12,15,17,19,22,24,26,29]`
  - An npz from mjlab's model loaded on the user's robot would silently point all six arm targets at the wrong bodies.
- **Totals.** Subtree mass is 33.413 kg (user) vs 33.341 kg (mjlab). Both have the `imu_ang_vel` and `imu_lin_vel` sensors that the observations use. The user's file lacks `imu_upvector`, which the tracking task does not use.

| Clip | Frames @30 fps | Duration | Frames @50 Hz | Adaptive bins | Notes |
|---|---|---|---|---|---|
| locomotion1 | 120 | 3.97 s | 199 | 4 | straight walk, +3.96 m in x, about 1.0 m/s |
| locomotion2 | 125 | 4.13 s | 207 | 5 | straight walk, +4.3 m, about 1.05 m/s |
| dual_arm1 | 209 | 6.93 s | 347 | 7 | pelvis minimum 0.455 m; waist_pitch above hard limit on 69 frames |
| dual_arm2 | 209 | 6.93 s | 347 | 7 | pelvis minimum 0.393 m; waist_pitch above hard limit on 61 frames, above soft on 100 |

Other data notes:
- Locomotion clips stay within the soft limits.
- The reference foot-capsule centres sit at z = 0.002–0.004 m with radius 0.01 m (`g1_realsense.xml:10-11`), so the reference feet are about 6–8 mm into the floor. That causes a small push-out at reset; the reset already adds z noise of ±0.01.

### 2.4 Exact conversion recipe

**Option A: stock converter, mjlab G1 only.** Write each npy as a CSV with the quaternion reordered to xyzw, so the converter's swap restores wxyz:

```python
# uv run --no-project --with numpy python -I to_csv.py in.npy out.csv
q = np.load(src).astype(np.float64)
np.savetxt(dst, q[:, [0,1,2, 4,5,6,3, *range(7,36)]], delimiter=",")
```

Then run:

```bash
WANDB_MODE=offline uv run python -m mjlab.scripts.csv_to_npz --input-file out.csv \
  --output-name <name> --input-fps 30 --output-fps 50 --device cuda:0
cp /tmp/motion.npz dataset/tracking/<name>.npz
```

Run it on a GPU node, because `/tmp` is node-local. If W&B raises after the save, ignore it. The result is valid **only** with `get_g1_robot_cfg()`, which is mjlab's G1.

**Option B (recommended, ours): a small local converter.** It copies `run_sim`'s loop but builds the scene from the env cfg actually used for training, writes into the repo, skips W&B, and can also record palm sites for §6:

```python
from mjlab.scripts.csv_to_npz import MotionLoader            # resample + velocities (:23-180)
from mjlab.scene import Scene; from mjlab.sim.sim import Simulation, SimulationCfg
scene_cfg = <training env cfg>.scene      # e.g. tracking scene with entities={"robot": get_g1_29dof_robot_cfg()}
scene = Scene(scene_cfg, device); model = scene.compile()
sim = Simulation(num_envs=1, cfg=SimulationCfg(), model=model, device=device)
scene.initialize(sim.mj_model, sim.model, sim.data); robot = scene["robot"]
# per frame: robot.write_root_state_to_sim([pos+env_origin_xy, quat_wxyz, lin_w, ang_w]);
#            robot.write_joint_state_to_sim(...); sim.forward(); scene.update(dt)
# log joint_pos/joint_vel/body_link_{pos,quat,lin_vel,ang_vel}_w[0] (+ site_pos_w for left/right_palm)
np.savez(out, fps=[50], joint_pos=..., ...)
```

Because the toaster is a separate entity, adding it to `scene_cfg` does not change the robot's body indexing. Plain CPU MuJoCo also works for the FK; I ran `mj_kinematics` on all clips without a GPU or GL. For velocities, though, you would have to reproduce `compute_velocity_from_cvel` (`MJ/entity/data.py:20-31`) after `mj_forward`. If you go that way, check it against one stock csv_to_npz output.

---

## 3. The tracking MDP

### 3.1 Order of operations within one step (matters for a world model's reward head)
`step()` runs as follows (`MJ/envs/manager_based_rl_env.py:418-464`):
1. 4 physics substeps.
2. Terminations, then rewards (`:436-441`). Derived quantities such as `xpos` are one substep stale here, as the docstring at `:387-401` says.
3. Auto-reset of finished envs (`:444-448`).
4. `sim.forward()` (`:454`).
5. `command_manager.compute`, which advances `time_steps` by 1 (`:456`).
6. Step and interval events, including pushes (`:458-461`).
7. Observations (`:464`).

So the observation shows reference frame k. The reward at the next step scores the post-step robot against frame k, then k moves to k+1. `evaluate.py:112-126` documents this pairing.

Auto-reset happens inside `step`. The returned observation is post-reset, so the terminal state is lost unless `cfg.auto_reset=False` (`:144`, `:381-385`).

### 3.2 Observations (`tracking_env_cfg.py:47-120`, functions in `MJ/tasks/tracking/mdp/observations.py`)

| Term | Dim | Actor noise (uniform) | Definition |
|---|---|---|---|
| `command` | 58 | none | reference `joint_pos` and `joint_vel` at the current frame (`commands.py:126-127`) |
| `motion_anchor_pos_b` | 3 | ±0.25 | reference torso position in the robot torso frame (`observations.py:18-28`) |
| `motion_anchor_ori_b` | 6 | ±0.05 | first two columns of the relative rotation matrix (`:31-41`) |
| `base_lin_vel` | 3 | ±0.5 | sensor `robot/imu_lin_vel` |
| `base_ang_vel` | 3 | ±0.2 | sensor `robot/imu_ang_vel` |
| `joint_pos` | 29 | ±0.01, plus encoder bias (`biased=True`) | relative to `default_joint_pos` (`MJ/envs/mdp/observations.py:51-61`) |
| `joint_vel` | 29 | ±0.5 | |
| `actions` | 29 | none | last raw action (`:80-83`) |

- Actor: **160** dims; **154** for No-State-Estimation.
- Critic (no noise): the same terms plus `body_pos` 14×3 and `body_ori` 14×6 of the **robot's** tracked bodies in the robot torso frame (`observations.py:44-69`), giving **286** dims.
- Groups `actor` and `critic` map one-to-one (`MJ/rl/config.py:92-94`).

### 3.3 Goal representation
The goal is one reference frame. The policy sees target joint position and velocity plus the relative torso pose. Rewards use the reference body poses re-anchored to the robot: `update_relative_body_poses` (`commands.py:377-405`) takes the robot torso xy, the reference torso z and the yaw difference only. No future frames are observed. The SuperMimic summary says its encoder sees K=10 future frames. Adding them is easy: index `motion.joint_pos[clamp(time_steps+k)]` and the body poses the same way.

### 3.4 Rewards
Defined at `tracking_env_cfg.py:209-251`, functions in `MJ/tasks/tracking/mdp/rewards.py`. All exp terms are `exp(-err/std²)`; body terms average the squared error over the 14 bodies.

| Term | Weight | std | Error |
|---|---|---|---|
| `motion_global_root_pos` | 0.5 | 0.3 | ‖reference torso − robot torso‖², world frame (`:26-33`) |
| `motion_global_root_ori` | 0.5 | 0.4 | quaternion angle² (`:36-41`) |
| `motion_body_pos` | 1.0 | 0.3 | re-anchored body positions (`:44-59`; takes an optional `body_names` subset) |
| `motion_body_ori` | 1.0 | 0.4 | (`:62-77`) |
| `motion_body_lin_vel` | 1.0 | 1.0 | world frame (`:80-95`) |
| `motion_body_ang_vel` | 1.0 | 3.14 | world frame (`:98-113`) |
| `action_rate_l2` | −0.1 | – | ‖a_t − a_{t−1}‖² on raw actions (`MJ/envs/mdp/rewards.py:58-66`) |
| `joint_limit` | −10 | – | L1 distance outside the soft limits (`:81-96`) |
| `self_collisions` | −10 | – | number of substeps (out of 4) with self-contact force above 10 N (`rewards.py:116-135`) |

- Per-step reward is the weighted sum × dt (0.02). The positive maximum is 5 × 0.02 = 0.1 per step, so 50 over a 10 s episode.
- Logged `Episode_Reward/*` values are episode sums divided by `episode_length_s` (`reward_manager.py:106-110`).
- Per-term weighted values are available as `reward_manager._step_reward` (`:132`).

**SuperMimic deltas, per the summary:** joint_limit −2, self-collision removed; the other weights and stds are the same.

### 3.5 Terminations (`tracking_env_cfg.py:257-279`, `MJ/tasks/tracking/mdp/terminations.py`)
- `time_out`
- `anchor_pos`: |reference torso z − robot torso z| > 0.25 (`:27-34`)
- `anchor_ori`: |difference in the z-component of projected gravity| > 0.8 (`:37-53`)
- `ee_body_pos`: z error > 0.25 for any ankle or wrist_yaw link (`:73-86`, with bodies set at `env_cfgs.py:64-69`)

The summary's |e_z| > 0.25 / tilt > 0.8 matches this.

### 3.6 Reset (RSI) and clip wrap
`MotionCommand._resample_command` (`commands.py:319-375`) runs on episode reset through `CommandTerm.reset` (`MJ/managers/command_manager.py:99-107`). It:
1. picks a start frame (`start`, `uniform` or `adaptive`);
2. writes the reference root state, perturbed by `pose_range` (xy ±0.05, z ±0.01, roll and pitch ±0.1, yaw ±0.2) and `velocity_range` (xy ±0.5, z ±0.2, roll and pitch ±0.52, yaw ±0.78);
3. writes joints with ±0.1 rad noise (`tracking_env_cfg.py:139-159`; the cfg default is ±0.52, `commands.py:589`), clipped to the **soft** limits (`:311-312`);
4. calls `robot.reset` (`:317`).

`resampling_time_range=(1e9, 1e9)` means the command is never resampled on a timer.

**Clip wrap.** When `time_steps` reaches the clip length, `_update_command` calls `_resample_command` again (`:407-411`). That teleports the robot to a newly sampled frame, with noise, with no termination and no episode reset. It happens after `sim.forward()` in `step`, so that step's xpos-based observations are stale. With 10 s episodes this happens about 2.5 times per episode on the walking clips (a ~4 m jump back) and about once on the lift clips.

### 3.7 Domain randomization and pushes
All DR runs at startup only, so it is fixed per env for the whole run (`tracking_env_cfg.py:165-203`):
- `base_com`: `body_ipos` of `torso_link` += U(±0.025, ±0.05, ±0.05) (`dr/body.py:330-352`)
- `encoder_bias`: U(±0.01) per joint, applied only to biased observations (`dr/joint.py:209+`, `entity/data.py:368-371`)
- `foot_friction`: absolute U(0.3, 1.2), shared across the 14 foot geoms (`dr/geom.py:117-145`, geoms at `env_cfgs.py:59-61`)

These are exactly the summary's ξ (friction, joint offsets, torso CoM). The summary calls ξ per-episode; switching these events to `mode="reset"` would do that (ours).

`push_robot` is an interval event every U(1, 3) s that adds `VELOCITY_RANGE` to the root velocity (`tracking_env_cfg.py:166-171`, `MJ/envs/mdp/events.py:316-337`). The summary says SuperMimic removes root-velocity pushes.

### 3.8 Adaptive sampling (`commands.py:249-290`, defaults at `:590-594`)
- Bins are 1 s each: `bin_count = T//50 + 1`.
- Per-bin failure counts come from `termination_manager.terminated`, which excludes time-outs, and are smoothed by an EMA with α = 0.001.
- Probability = failures + 0.1 / bin_count, convolved with kernel size 1 (so no smoothing), then normalized, then multinomial sampling.
- It logs entropy and top-1 bin metrics.

The summary describes SuperMimic as using **return-based** sampling over 1 s bins. The exact mapping is UNSPECIFIED in my summary. It would replace this hook.

### 3.9 Play and evaluate
- **Play** (`env_cfgs.py:87-98`): effectively infinite episode, no observation noise, no pushes, pose and velocity reset ranges cleared, `sampling_mode="start"`. The ±0.1 joint reset noise is **kept**.
- **Evaluate** (`evaluate.py:71-75,158-164`):
  - It uses the *training* cfg with `sampling_mode="start"` and pushes removed, but **leaves observation noise on**.
  - Success means the episode ended by time-out without a termination.
  - Metrics are MPKPE, root-relative MPKPE, joint-velocity RMS error and end-effector position/orientation error (`MJ/tasks/tracking/mdp/metrics.py:13-79`).
  - Episodes are 10 s and wrap the clip. For a clean comparison, set `episode_length_s` to the clip duration (ours).

---

## 4. Robot asset and actuators: mjlab's G1 vs the user's

| | mjlab `asset_zoo/robots/unitree_g1` | user `REPO/src/mjlab_g1/assets/g1_constants.py` |
|---|---|---|
| MJCF | `xmls/g1.xml`, 30 bodies (`g1_constants.py:20-27`) | `g1/g1_realsense.xml`, 31 bodies (+ realsense camera, + foot sites) (`:19-36`) |
| Joints, ranges, collision geoms | the 29 joints in identical order with identical ranges; same collision geom names | same |
| Gains | kp = armature·(2π·10)², kd = 2·2·armature·2π·10 (`:112-123`) | same formula (`:122-133`) |
| **wrist_roll** | 5020 group: kp 14.25, kd 0.907, 25 Nm (`:125-137`) | **4010 group** via `.*_wrist_.*_joint`: kp 16.78, kd 1.068, **5 Nm** (`:177-183`) |
| Other groups | 7520-14: kp 40.18, 88 Nm; 7520-22 (hip roll, knee): kp 99.10, 139 Nm; waist roll/pitch and ankles: kp 28.50, 50 Nm | same |
| Action scale 0.25·τ/kp | hip pitch/yaw, waist yaw 0.548; hip roll, knee 0.351; shoulder, elbow, waist roll/pitch, ankle, wrist_roll 0.439; wrist pitch/yaw 0.075 (`:278-286`) | same except wrist_roll **0.075** (`:237-245`). The dual-arm task overrides these (knee 1.79, …) and clips actions to ±1.5 (`tasks/dualarm/config/g1/env_cfgs.py:19-33`) |
| Default pose (action and observation offset) | `KNEES_BENT` (`:198-211`) | all zeros, root at (−0.03, 0.1, 0.78) (`:188-192`) |
| Foot friction | 0.6 (`:220-225`), then DR U(0.3, 1.2) | 1.0 (`:198-203`) |
| Soft limit factor | 0.9 | 0.9 |

- Tracking has no action clip. Reaching the mocap knee of 2.2 rad from 0.669 needs about a = 4.4 with scale 0.351, which is fine without a clip.
- On the user's model, the dual-arm wrist roll of up to 1.49 rad would need |a| of about 20 at scale 0.075.
- **Recommendation (ours):** run both arms of the comparison on **mjlab's G1**. It is the BeyondMimic setting, which the summary calls "low BeyondMimic gains", and the stock converter targets it. If you need the user's model, build the npz with Option B and copy the wrist_roll actuator group from mjlab.

---

## 5. Simulator state for a world model

### 5.1 Dimensions (CPU MuJoCo FK)

| | nq | nv | Contents |
|---|---|---|---|
| G1, either XML | 36 | 35 | qpos = root pos 3 + quat wxyz 4 + 29 joints. qvel = root linear velocity (**world**) 3 + root angular velocity (**body frame**) 3 + 29 (`MJ/entity/data.py:105-116`) |
| Toaster003 | 10 | 9 | free joint 7/6, plus `button_cancel` slide, `knob_doneness` hinge and `lever` slide (`REPO/src/mjlab_g1/assets/Toaster003/model.xml:51,73,79,86`) |
| G1 + toaster | **46** | **44** | 43 / 41 if you strip the three cosmetic joints from a box copy (ours) |

- No actuator activation state (na = 0, built-in position actuators).
- ctrl is set from the action, held for 4 substeps.
- The toaster XML also declares 3 motors (`model.xml:103-107`). I found no code in `entity.py` that deletes XML actuators, so they probably remain at ctrl 0. To verify, check `env.sim.mj_model.nu`.
- Solver warm-start affects results only slightly (10 iterations); I am treating it as not part of the state.

### 5.2 Reading state

```python
robot, box = env.scene["robot"], env.scene["toaster"]
qpos, qvel = env.sim.data.qpos, env.sim.data.qvel                   # [N, nq], [N, nv] torch views
root = robot.data.root_link_pose_w; root_v = robot.data.root_link_vel_w  # world-frame (ang vel via cvel)
jp, jv = robot.data.joint_pos, robot.data.joint_vel                 # [N,29]
box_pose, box_v = box.data.root_link_pose_w, box.data.root_link_vel_w
origin = env.scene.env_origins                                      # grid, spacing 2.0 (scene.py:31) → subtract from xy
xi = (env.sim.model.geom_friction[:, foot_geom_gids, 0],            # expanded per-env DR fields
      env.sim.model.body_ipos[:, torso_gid] - env.sim.get_default_field("body_ipos")[torso_gid],
      robot.data.encoder_bias)                                      # global ids via robot.indexing.{geom_ids,body_ids}
term, trunc = env.termination_manager.terminated, env.termination_manager.time_outs
k = env.command_manager.get_term("motion").time_steps              # reference index (detect wraps: k_new != k_old+1)
```

Derived quantities (xpos, cvel, sensors) are valid only after `sim.forward()` (`MJ/entity/data.py:36-42`).

### 5.3 Setting state (for resets, or to seed imagination from real states)
- `robot.write_root_state_to_sim([N,13], env_ids)` takes pos, quat wxyz and **world** linear and angular velocity (`MJ/entity/entity.py:868-882`).
- `write_joint_state_to_sim(pos, vel, env_ids=...)` (`:932-952`).
- The same calls exist for the box.
- Alternatively write `env.sim.data.qpos[ids] = ...` in place, then `env.sim.forward()`.
- Always write **in place**. CUDA graphs keep array pointers (`MJ/sim/sim.py:161-180`).
- `MotionCommand.reset_to_frame(env_ids, frame)` is a deterministic reset to a reference frame, followed by `update_relative_body_poses()` (`commands.py:555-578`).

### 5.4 Gotchas for collecting world-model data (ours, from the code above)
- **Terminal states.** Auto-reset discards them. Set `auto_reset=False` and call `env.reset(env_ids=...)` yourself, or snapshot state through the recorder's `record_pre_reset` hook (`manager_based_rl_env.py:446`).
- **Clip-wrap teleports** (§3.6) and **pushes** (qvel overwrites at `:461`) are discontinuities a deterministic model should not learn. Mask them, remove pushes as the summary says SuperMimic does, and set `episode_length_s` to the clip duration.
- **Reward head targets.**
  - Rewards are computed against one-substep-stale kinematics (`:387-401`). That is consistent, but a world model computing rewards analytically from the predicted next state will differ slightly.
  - `action_rate` and `joint_limit` are analytic in (a_t, a_{t−1}, q), so compute them outside the model (ours).
  - The tracking terms need FK of the 14 bodies plus their velocities. mjlab has no torch FK. Either use a learned reward head, which the summary says SuperMimic uses, or write a differentiable FK (ours).
- **No simulator gradients.** `wp.config.enable_backward = False` (`MJ/__init__.py:28`). That is irrelevant for a learned model, but it rules out a differentiable-sim shortcut.
- **SuperMimic items the summary gives:** 8-step autoregressive model loss; H = 32 imagination; GAE λ = 0.95 critic; 1000-step replay per env; exploration noise N(0, 0.1²); 3×2048 MLP.
- **UNSPECIFIED here:** the exact state vector s, the normalization, the observation-head targets, and the bin-sampling formula. Take them from the paper's Tables A.2–A.5.

---

## 6. Building a lifting variant

### 6.1 Scene
- Entities: `{"robot": ..., "toaster": get_toaster_cfg()}` (`REPO/src/mjlab_g1/assets/toaster_constants.py:56-63`; grasp markers have friction 2.0 at `:41-47`).
- Reuse the sensors `left/right_hand_toaster_contact`, `toaster_contact` and `illegal_toaster_contact` (`REPO/src/mjlab_g1/tasks/dualarm/config/g1/env_cfgs.py:121-191`). The geom names exist in both G1 XMLs.
- Raise the solver buffers. Tracking's 35 / 250 is far too small. The dual-arm task uses nconmax 256, njmax 1024, ccd_iterations 50 and contact_sensor_maxmatch 64 (`:43-46`).

### 6.2 Box reference from the clips (palm midpoint after grasp)
Run FK on the 50 Hz clip, which is the same pass as the conversion, and log the `left_palm` and `right_palm` sites (`g1_realsense.xml`, wrist_yaw_link +0.08 x). The algorithm (ours):
1. `t_g` = frame with the lowest palm-midpoint z.
2. `p0 = (mid_x(t_g), mid_y(t_g), 0.125)`, which puts the box at rest on the floor (`toaster_constants.py:34`).
3. `box(t) = p0` for t < t_g, else `p0 + mid(t) − mid(t_g)`. This is continuous at t_g.
4. Yaw = 90° (the default `rot`, `toaster_constants.py:35`, so the markers face the palms; keep it, see the user's pitfall #7) + pelvis yaw at t_g.
5. Velocities by `np.gradient`.
6. Store as extra npz keys (`object_pos_w`, `object_quat_w`, `object_lin_vel_w`, `object_ang_vel_w`). `MotionLoader` ignores them.

Measured on the user's model:

| | t_g | palm-mid at t_g (world) | pelvis yaw | 2 cm rise | 95 % height | Δz | final palm-mid z | max palm-mid speed |
|---|---|---|---|---|---|---|---|---|
| dual_arm1 | 2.17 s | (0.464, 0.017, 0.223) | 8.0° | 2.43 s | 3.57 s | 0.798 m | 1.021 m | 0.90 m/s |
| dual_arm2 | 2.23 s | (0.484, 0.039, 0.160) | 8.2° | 2.47 s | 3.97 s | 0.786 m | 0.946 m | 0.73 m/s |

- **Height fits.** Marker centres sit at z 0.18 m (`model.xml:54-57`); the palm midpoints are within the ±0.05 m marker half-height. The left-minus-right palm vector points along +y, matching the default toaster yaw.
- **Spawn distance differs.** The box would sit about 0.51–0.52 m ahead of the pelvis. The user's current spawn is about 0.405 m ahead (toaster x 0.375 vs robot −0.03), so it is about 11 cm closer.
- **Lift height differs.** The reference lift is about 0.79 m, against the user's task goal of +0.625 m.

### 6.3 The width mismatch
- To touch the markers, the palms need a separation of about 2 × (0.095 marker face + 0.035 hand capsule radius) ≈ **0.26 m** (`model.xml:56-57`; hand capsule `g1_realsense.xml:259-260`; this is approximate because the palm site lies on the capsule axis).
- The mocap separation is **0.426 / 0.472 m at grasp** and **0.355–0.395 m while holding**. It also shrinks by 6–9 cm *during* the lift, so no rigid box matches the clip.
- So a faithful wrist tracker hovers about 5–10 cm away from each marker, and resets mid-lift drop the box.

Options, all ours:
- **(a)** After t_g, track only legs, pelvis and torso (`motion_body_*` accept `body_names`, `rewards.py:44-49`) and let box terms drive the arms.
- **(b)** Edit the arm reference after grasp by solving IK so the palms sit on the marker faces.
- **(c)** Use a custom box about 0.29–0.32 m wide (hold-phase separation minus 0.07).

(a) is the least work. (a) plus (b) gives a physically consistent reference.

### 6.4 Wiring
Reset events run **before** the command reset (`manager_based_rl_env.py:558-562` vs `:581`), so an event cannot know the sampled frame. Subclass `MotionCommand` and override `_resample_command` to:
1. call `super()`, then
2. write the toaster's root state from the box reference at `time_steps`, plus zeros for its 3 internal joints.

This also covers the clip-wrap path (`commands.py:407-411`). Optionally restrict the reset to frames before t_g until the arm reference is fixed.

### 6.5 Rewards, observations and terminations to add (ours; weights and stds UNSPECIFIED; the paper has no object terms per the summary)
- **Rewards:**
  - box position, `exp(-‖e‖²/0.1²)`, weight 1.0
  - box orientation, std 0.4, weight 0.5
  - both-hands contact, weight 0.5, gated on t ≥ t_g
- **Observations:** box pose and reference box pose in the torso frame.
- **Termination:** box error > 0.25 m after t_g, mirroring the anchor threshold.

Portability of the user's terms:
- These port to a plain `ManagerBasedRlEnv` unchanged, because they use only `env.scene` and `episode_length_buf`: `hands_contact`, `hands_at_markers` and `marker_force` (`REPO/src/mjlab_g1/tasks/dualarm/mdp/rewards.py:258-320`).
- These need rewriting against `env.scene["toaster"]` and the new command: `object_trajectory_tracking`, `object_pose` and the other custom terms. They use `env.toaster`, `env.robot`, `env.get_object_trajectory_reference`, `env.success_hold_buf`, `env._lift_progress` and `env.grasp_site_ids` (`rewards.py:322-357`, `observations.py:38-82`).
- The virtual-PD assistance only exists in the custom env. The summary notes that external assistance forces are unmodeled unless placed in ξ or the state.

### 6.6 Implications for SuperMimic on the lift
- The state grows to 46 / 44 (or 43 / 41). The box's 13-d root state must be predicted, and contact flags are useful extra targets (ours).
- The summary says SuperMimic has "no terrain or object interaction" and filtered out box clips. A deterministic MLP model of friction-only two-palm grasp contact is therefore outside the paper's demonstrated scope.
- Treat the lift row as exploratory. The walking row is the like-for-like comparison: mjlab tracking PPO vs SuperMimic on the same env cfg, same robot and same reward. The user's AMP-PPO velocity task and the dual-arm task should be reported as separate rows, not as the "walking baseline".