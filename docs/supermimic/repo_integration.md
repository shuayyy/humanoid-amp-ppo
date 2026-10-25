# SuperMimic + PPO side by side: integration audit of `humanoid-amp-ppo`

**Scope.** This was a read-only audit. Repo files were not edited, and no `uv sync`, training run or job submission was done. The only cluster calls were `sbatch --test-only`, which validates a request without creating a job (checked afterwards with `squeue`).

**How to read the citations.**
- Repo paths are relative to `<repo>`.
- `MJ/` means `<mjlab v1.4.0>/src/mjlab`.
- Paper citations come from the arXiv TeX source (v1). Eq. 1 is the world-model loss, Eq. 2 the policy objective, Eq. 3 the critic loss.
- Paper section map: §3.1 training loop, §3.2 world model, §3.3 policy, §4.1 scaling/PPO-Sim, §4.2 world-model properties.
- Appendix map: App. A spaces, B multi-interface, C networks, D.1 algorithm and reset sampling, D.2 hyperparameters, E data filter, F evaluation, G velocity task.
- Appendix tables: A.1 observation, A.2 reward, A.3 DR, A.4 networks, A.5 hyperparameters, A.6 velocity reward.

---

## 0. Bottom line (blockers first)

1. **The scripts only accept AMP configs.**
   - `train.py:39` types `agent` as `RslRlAMPOnPolicyRunnerCfg` and `train.py:61` asserts it.
   - `train.py:98-111`, `play.py:99-108` and `render_video.py:71-84` each pick the env class from a two-way isinstance chain.
   - A new learner or a plain tracking env fails before the env is built. The fix is small and touches 3 files (§1).
2. **The Slurm launcher's account is invalid.**
   - `scripts/train_slurm.sbatch:3` sets `--account=cchamzas`.
   - `sbatch --test-only` returns "Invalid account or account/partition combination specified".
   - Your associations are only `rbe577` (default) and `mmaghelih`, and every job in your sacct history used one of those two.
3. **mjlab's tracking task is not visible to your `train.py`, and its motion `.npz` must be rebuilt with your robot.**
   - mjlab keeps its own task registry, separate from yours.
   - Your `g1_realsense.xml` has 31 bodies; `realsense_d435_link` sits right after `torso_link`. mjlab's `g1.xml` has 30 bodies.
   - `MotionLoader` indexes the npz by the training robot's body indices, so every arm body would be off by one.
4. **The reference poses are out of reach whenever actions are clipped.** I computed this from the clips and the actuator gains (§4.4).
   - With BeyondMimic action scales and your all-zero default pose, the lift clips need |a| up to 6.4 on the knees and up to 20 on wrist roll. The walking clips need 4.2–4.6.
   - Any `clip_actions` makes these poses unreachable. The dual-arm task uses 1.5. This is the same failure class as the v15b reachability root cause in your notes.
5. **The env hides several things a world model needs** (§3):
   - Auto-reset throws away terminal states. Your two custom envs always auto-reset.
   - Reward terms are functions of the live env, not of a state vector.
   - A noise-free observation only exists if you add an uncorrupted observation group.
   - Domain-randomization parameters are drawn once per env for the whole run (startup events). The paper calls them per-episode.
6. **"Lifting" splits into two different things.**
   - The paper excludes object interaction (§6 Limitations; App. E filters out box clips).
   - The like-for-like comparison is tracking the robot-only squat-grasp-lift clip.
   - A toaster lift trained with SuperMimic is a research extension, not a config change (§7).
7. **The vendored rsl_rl `OnPolicyRunner`/`PPO` do work with mjlab 1.4's tracking config.**
   - Both are the same 5.2.0 API.
   - Conditions: run them through mjlab's `MjlabOnPolicyRunner`/`MotionTrackingOnPolicyRunner`, and match `obs_groups` to the env's group names (§4).
8. **Side-by-side runs must go on `short`.**
   - The `quick` QoS allows 1 GPU per user, so four runs would execute one after another.
   - `short` allows 24 h and 12 GPUs per user, and has H100, H200, A100 and rtx_pro_6000_b.
   - `long` has no H100 or H200.

---

## 1. How the scripts pick config types and runner classes, and the minimal change

### Where AMP is assumed today

| Location | Behaviour |
|---|---|
| `src/mjlab_g1/scripts/train.py:12-21` | Imports only the two custom env classes and the AMP config. `:31-33` and `:60` are the commented "enable for just PPO" lines. |
| `train.py:36-39` | `TrainConfig.env: G1LocomotionManagerBasedRlEnvCfg \| G1DualarmManagerBasedRlEnvCfg`; `agent: RslRlAMPOnPolicyRunnerCfg`. tyro builds the CLI from these annotations (`:261-270`). |
| `train.py:61` | `assert isinstance(agent_cfg, (RslRlAMPOnPolicyRunnerCfg))`. This is the only runtime type gate; the registry checks nothing (`tasks/registry.py:16-18` imports types under `TYPE_CHECKING` only). |
| `train.py:98-111` | isinstance chain on the env config, otherwise `TypeError`. Both custom configs subclass `ManagerBasedRlEnvCfg` (`g1_dualarm_rl_env.py:31`, `g1_locomotion_rl_env.py:26`), so a generic branch must come last. |
| `train.py:137-148` | The video interval uses `agent.save_interval * agent.num_steps_per_env`. The wrapper uses `agent.clip_actions`. |
| `train.py:153-160` | `load_runner_cls(task_id)` and `assert ... is not None`; the `or OnPolicyRunner` fallback is commented out at `:153`. Then `runner_cls(env, agent_cfg_dict, str(log_dir), device)`. |
| `train.py:164-186` | Resume calls `runner.load(path)`. Warm start calls `runner.load(path, load_cfg={actor, critic, optimizer, iteration, rnd, discriminator}, strict=True)`. |
| `train.py:188-191` | Dumps `params/agent.yaml` after the runner is built, but `PPO.construct_algorithm` pops `class_name` in place (`rsl_rl/rsl_rl/algorithms/ppo.py:412-414`). The saved YAML is therefore lossy. mjlab dumps before construction (`MJ/scripts/train.py:161-165`). |
| `train.py:193-196` | `runner.learn(num_learning_iterations=..., init_at_random_ep_len=...)`. A new runner must accept this keyword. |
| `play.py:121, 141-152` | `load_runner_cls(task_id) or OnPolicyRunner`. Calls `runner_cls(env, asdict(agent_cfg), device=device)` with no `log_dir`. `load(..., map_location)` has a fallback for a "discriminator" size-mismatch string (`load_cfg={"discriminator": False}`). Then `get_inference_policy(device)` is called as `policy(obs_TensorDict)` by the viewers (`:162-165`). |
| `render_video.py:57-62, 95-118` | Same pattern, with an actor-only `load_cfg` that includes `"discriminator": False`, and a `policy(obs)` loop. (The cluster has no GL, so rendering does not work there anyway; see CLAUDE.md.) |
| `tasks/registry.py:21-26, 39-57, 86-92` | `_TaskCfg(env_cfg, play_env_cfg, rl_cfg, runner_cls)`. Registering a duplicate ID raises. |
| `tasks/__init__.py:1, 7-20` + `MJ/utils/lab_api/tasks/importer.py:69, 76-82` | Every subpackage of `mjlab_g1.tasks` is auto-imported. The blacklist `["utils", ".mdp"]` is a substring match on the full dotted module name, so a package such as `tracking/motion_utils/` would be skipped silently. Import errors are raised. Registration pattern to copy: `tasks/dualarm/config/g1/__init__.py:1-16`. |

### Minimal changes (our proposal)

1. **Agent config.** Add `SuperMimicRunnerCfg(RslRlBaseRunnerCfg)` (see §6). Subclassing the base inherits every field the scripts read: `seed`, `num_steps_per_env`, `max_iterations`, `save_interval`, `experiment_name`, `run_name`, `logger`, `wandb_*`, `resume`, `load_run`, `load_checkpoint`, `clip_actions` (`src/mjlab_g1/rl/config.py:100-134`).
2. **`train.py` types.** Use `env: ManagerBasedRlEnvCfg` and `agent: RslRlBaseRunnerCfg`, which is mjlab's own pattern (`MJ/scripts/train.py:27-28, 42-46`). Change the assert at `:61` to the base class.
3. **One env factory** (`src/mjlab_g1/envs/factory.py::make_env(cfg, device, render_mode)`), used by all three scripts. It replaces the three copies of the isinstance chain. Custom configs map to their classes; anything else falls back to `ManagerBasedRlEnv`.
4. **Move the YAML dump before runner construction** (from `train.py:188-191` to before `:160`).
5. **Register a `runner_cls` for every new task**, so the `or OnPolicyRunner` fallback in `play.py:141` / `render_video.py:97` is never used. A new runner must provide:
   - `__init__(env, train_cfg: dict, log_dir=None, device="cpu")`
   - `learn(num_learning_iterations, init_at_random_ep_len)`
   - `save(path)`
   - `load(path, load_cfg=None, strict=True, map_location=None)`, tolerating the keys `actor/critic/optimizer/iteration/rnd/discriminator`
   - `get_inference_policy(device)`, returning a callable on a TensorDict
   - `add_git_repo_to_log(path)`
6. **Leave the AMP tasks alone.** They keep `RslRlAMPOnPolicyRunnerCfg`, `DualArmOnPolicyRunner`/`LocomotionOnPolicyRunner`, the custom envs and `rsl_rl/`.

---

## 2. RSL-RL 5.2: what the new learner can reuse and what must be written

### Reusable

| Piece | Where | Notes |
|---|---|---|
| `OnPolicyRunner` shell: multi-GPU setup, Logger, save/load, export | `rsl_rl/rsl_rl/runners/on_policy_runner.py:26-54, 136-161, 163-201, 207-249` | Subclass it and override `learn()`. `__init__` builds the algorithm via `resolve_callable(cfg["algorithm"]["class_name"]).construct_algorithm(obs, env, cfg, device)` (`:39-40`), so the SuperMimic algorithm class needs that staticmethod (or the subclass overrides `__init__`). `save` writes `alg.save() + iter + infos` (`:136-143`). `load` calls `alg.load(dict, load_cfg, strict)` (`:145-161`). |
| `Logger` | `utils/logger.py:24-34` | Reads `cfg["algorithm"]["rnd_cfg"]` (`:54, :207, :243`), so set it to `None` (the AMP runner does the same `setdefault`, `amp_on_policy_runner.py:22-25`). FPS uses `cfg["num_steps_per_env"]` (`:162`). `log()` requires an `action_std` tensor (`:151, :197, :250`): pass `torch.full((29,), 0.1)`, the collection noise σ (Table A.5), because a deterministic `MLPModel.output_std` dereferences a `None` distribution (`models/mlp_model.py:139-141`; `on_policy_runner.py:123`). Keys in `extras["log"]` containing "/" are logged as-is (`:184-189`), which suits WM loss and imagined-return scalars. `import git` (`:10`) needs `GIT_PYTHON_REFRESH=quiet` (the launcher sets it at `:35`). |
| Multi-GPU | `on_policy_runner.py:207-249`; `ppo.py:447-484` | NCCL setup, then broadcast and flat gradient all-reduce. Copy the all-reduce pattern for WM + actor + critic. Paper (Table A.5): "data parallel; gradients averaged", learning rate ×√2 on 2 GPUs. `train.py:222-248` already launches torchrunx when `--gpu_ids` lists more than one GPU. |
| `MLP`, `EmpiricalNormalization` | `modules/mlp.py`; `modules/normalization.py:15-69` | `forward` (`:46`) is differentiable. `update` (`:51`) is separate, and `until` (`:18`) freezes the statistics. Use these for the WM, actor and critic, with normalizers for s, a, ξ and ō. |
| `MLPModel` | `models/mlp_model.py:30-80` | With `distribution_cfg=None` it is deterministic (`:67-73, :103-108`) and exportable to JIT/ONNX (`:163-169`). Its input is a concatenation of TensorDict groups (`:110-119`). Fine for the critic or a flat actor. The paper's encoder/decoder actor (Table A.4: K=10 future-frame encoder (2048,1024,512,256) → tanh 32-d latent; decoder (2048,1024,512,256) over z plus the last 10 observations) is not an `MLPModel`: it needs a new module plus a TensorDict-forward wrapper for play/render. |
| `resolve_callable` | `utils/utils.py:97-174` | Short names only search inside `rsl_rl` (`:164-168`), so use the colon form (`:127-136`): `"mjlab_g1.rl.supermimic.algorithm:SuperMimic"`. |
| `VecEnv` ABC | `env/vec_env.py:14-93` | Only `get_observations`/`step` plus attributes. No `reset(env_ids)`, no terminal observation, no state. Extend it in our wrapper (§3). |

### Must be written

- **Sequence replay buffer.** A per-env ring of 1000 steps (Table A.5) holding:
  - s_t
  - the executed noisy a_t, clipped exactly as the wrapper clips it (`src/mjlab_g1/rl/vecenv_wrapper.py:84-85`)
  - ξ, ō_t, r_t (total and per term), d_t, a timeout flag, and the goal index (clip id, frame)

  It samples k=8 windows that never cross a reset. Nothing existing fits: `ReplayBufferMulti` (`storage/replay_buffer_multi.py:8-35`) is an AMP-window ring with no time order, and `RolloutStorage` is cleared after every update (`ppo.py:339`). Memory estimate (ours): 4096 × 1000 × about 230 floats × 4 B ≈ 3.8 GB.
- **World model.** Dynamics MLP (2048,2048,2048) (Table A.4), plus observation, reward and termination heads. The k=8 autoregressive loss of Eq. 1. Adam at 2e-4, 10 batches × 2048 windows per update, global gradient clip 1.0 (Table A.5).
- **BPTT actor-critic.**
  - H=32, γ=0.99, survival weight ĉ_t (Eq. 2). Critic GAE with λ=0.95 (Eq. 3), 5 epochs × 4 minibatches.
  - Policy Adam with β=(0.9, 0.95); gradient clip at global norm 1.0, then element-wise 0.1; 4096 imagined rollouts per update.
  - Cosine LR decay to 0.03× over 250k iterations, then constant; 320k iterations in total.
  - Sensor noise η is re-added to predicted observations (§3.3; Table A.1).
- **Imagined observation history.** Built in mjlab's per-term layout: each term's history is flattened (`MJ/managers/observation_manager.py:352-358`), then terms are concatenated (`:378`). The deployed policy must see exactly the layout it trained on.
- **Return-based reset sampling** (App. D.1) and the reward/termination heads (§3).
- **Checkpoint format.** Keep PPO's keys (`ppo.py:357-391`: `actor_state_dict`, `critic_state_dict`, `optimizer_state_dict`, `load_cfg.get(...)`) and add `world_model_state_dict` and the WM/critic optimizers. The replay buffer is not checkpointed, so a resumed run re-bootstraps it (the same as AMP).
- **Location.** Put this in `src/mjlab_g1/rl/supermimic/`. Not in `rsl_rl/`, which CLAUDE.md says is the AMP implementation and should not be edited casually. Not under `tasks/`, which the registry walker imports.

---

## 3. Does the env expose what a world model needs?

| Need | Where it is | What is hidden or missing | Recommendation |
|---|---|---|---|
| **Full state, read** | `robot.data.root_link_{pos,quat,lin_vel,ang_vel}_w` (`MJ/entity/data.py:448-463`); `joint_pos`/`joint_vel` (`:364, :374`); `joint_pos_biased` = pos + `encoder_bias` (`:369-371`); `body_link_{pos,quat,lin_vel}_w` (`:489-499`); zero-copy `sim.data.qpos/qvel` (`MJ/sim/sim_data.py:15-77`, `MJ/sim/sim.py:364-369`) | `RslRlVecEnvWrapper` returns only observation groups (`vecenv_wrapper.py:73-75, 94`) | Add a `wm_state` observation group (no noise, no history); it then appears in the TensorDict without wrapper changes. |
| **Full state, write** | `Entity.write_root_state_to_sim` (`MJ/entity/entity.py:868`), `write_joint_state_to_sim` (`:932`), `reset` (`:824`); `sim.reset(env_ids)` (`sim.py:464-475`) | — | SuperMimic never writes world-model states back into the simulator; imagination stays inside the WM (§3.1, Fig. 2). Sim resets only go to reference frames, which `MotionCommand._write_reference_state_to_sim` already does (`MJ/tasks/tracking/mdp/commands.py:300-317`). |
| **DR context ξ** | Paper Table A.3: friction U[0.3,1.2], joint calibration offset U[±0.01], torso CoM offset (±0.025, ±0.05, ±0.05). mjlab tracking has exactly these events: `foot_friction`/`encoder_bias`/`base_com` (`MJ/tasks/tracking/tracking_env_cfg.py:172-202`; torso and foot geoms set at `MJ/tasks/tracking/config/g1/env_cfgs.py:59-62`). Read back from `sim.model.geom_friction[:, foot_geoms, 0]` (written by `MJ/envs/mdp/dr/geom.py:117-145`), `sim.model.body_ipos[:, torso] − sim.get_default_field("body_ipos")[torso]` (`dr/body.py:330-352`, `sim.py:418-434`), and `robot.data.encoder_bias` (`dr/joint.py:209-241`, `data.py:74`). Your tasks write the same fields via `randomize_field` (`tasks/common_events.py:71-105`). Dual-arm adds object mass/CoM/friction and robot friction (`dual_arm_env_cfg.py:163-243`), and its `foot_friction` event is dead (CLAUDE.md pitfall 4). | (a) None of this reaches the wrapper's outputs. (b) All of it is `mode="startup"`, so ξ is fixed per env for the whole run, while the paper says "per-episode" (§3.2): 4096 envs give the WM 4096 ξ samples in total. (c) Pushes: the paper removes root-velocity pushes (App. A, "Sim context"), but mjlab tracking has `push_robot` (`tracking_env_cfg.py:166-171`), as do your tasks (`dual_arm_env_cfg.py:219-222`, `locomotion_env_cfg.py:158-161`). | Add a `wm_context` observation group that reads these fields. Start with startup DR; per-reset resampling would need `mode="reset"` plus a `set_const` recompute for `body_ipos` (`common_events.py:71-76`). Pop `push_robot` in both arms; PPO-Sim "shares … domain randomization" (§4.1). |
| **Termination flags** | `termination_manager.terminated` / `time_outs` / `get_term(name)` (`MJ/managers/termination_manager.py:79-84, 115-116`); `env.reset_terminated`/`reset_time_outs` (`MJ/envs/manager_based_rl_env.py:436-438`; `g1_dualarm_rl_env.py:411-413`) | The wrapper merges both into a long-typed `dones` (`vecenv_wrapper.py:87-90`) and only puts `time_outs` in extras when the task is not finite-horizon (`:91-92`) | Use `d = terminated` from `env.unwrapped` directly. |
| **Terminal state s_T** | Base env: with `auto_reset=True` the reset happens inside `step()`, so the returned observation is post-reset (`manager_based_rl_env.py:443-454`). mjlab 1.4 has `cfg.auto_reset=False` (`:144-155`): `step` returns the true terminal observation, and the caller must `reset(env_ids=...)` before stepping again (`:356-376, :411-416, :466-469`). | The custom G1 envs ignore `auto_reset` and always reset in `step` (`g1_dualarm_rl_env.py:432-436`; `g1_locomotion_rl_env.py:203-207`). The wrapper's `reset()` resets every env (`vecenv_wrapper.py:77-79`). This is CLAUDE.md pitfall 2 in general form. | New tracking tasks build on the base `ManagerBasedRlEnv`: use `auto_reset=False` plus a wrapper `reset_done(env_ids)`. For the custom envs, mask k-windows that cross a reset and make the termination head analytic (closed form: `bad_anchor_pos_z_only`/`bad_anchor_ori`, `MJ/tasks/tracking/mdp/terminations.py:27-55`). |
| **Per-term reward** | `RewardManager._step_reward` holds `[num_envs, n_terms]` unscaled rates (`MJ/managers/reward_manager.py:46-48, 73-74, 132`). Dual-arm uses `dualarm_reward_manager` + `reg_reward_manager` and leaves the base manager empty (`g1_dualarm_rl_env.py:353-361, 416-419`). | The wrapper only exposes episodic sums in `extras["log"]` at reset, and the custom envs never clear that dict (pitfall 5). More importantly, terms are computed as `term_cfg.func(self._env, ...)` (`reward_manager.py:125`) from live sim and command state, so they cannot be evaluated on a predicted ŝ. Forward kinematics is not differentiable: `MJ/__init__.py:28` sets `wp.config.enable_backward=False`. | Choose one: a learned reward head trained on `_step_reward` targets, or the 6 tracking + 2 regularization terms of Table A.2 re-implemented as pure torch functions of (ŝ, g, a, a_prev). The second needs the tracked-body kinematics inside ŝ. The paper allows either: "each head is either learned or computed analytically" (§3.2). |
| **Noise-free observation ō** | Noise is applied inside the observation manager when `enable_corruption` is on (pipeline at `observation_manager.py:21`). Group history setting: `:91-98`. | Only the noisy actor group exists | Add a `wm_obs` group (no corruption, no history), and give the `actor` group `history_length=10`. Paper observation (Table A.1): q−q0 (29), q̇ (29), IMU angular velocity (3), projected gravity (3), a_{t−1} (29) = 93 per frame, 10 frames. This differs from mjlab tracking's actor terms (`tracking_env_cfg.py:47-80`), which add command, anchor position/orientation and base linear velocity and have no projected gravity. |
| **Goal g** | `MotionCommand.command` = the current reference joint position and velocity only (`commands.py:125-127`) | The paper uses K=10 future reference frames (App. A, "Goal"; Table A.4) | New observation term reading the motion arrays at `time_steps+1..+10`, clamped. |
| **Reset sampling** | mjlab "adaptive" sampling is failure-count based over 1 s bins (`commands.py:93, 249-290`) | Paper: return-based mixture, 50% uniform / 40% softmax(−25·G_b/G_max) / 10% past weight (App. D.1) | Subclass `MotionCommand` and add a `"return"` sampling mode. |
| **Multiple clips** | `MotionLoader` loads one npz (`commands.py:36-60, 81-83`); at the end of the clip `_update_command` resamples and teleports the robot (`:407-411`) | Each task has 2 clips | Multi-clip loader. Keep (clip, frame) in the replay buffer. |
| **Wrapper `unwrapped`** | Returns `self.env` (`vecenv_wrapper.py:48-50`); mjlab's version returns `env.unwrapped` | With `VideoRecorder` in between (`train.py:137-146`), the setters for `episode_length_buf` and `reset_env_ids` (`:62-68`) write onto the VideoRecorder, whose `__getattr__` only forwards reads (`MJ/utils/wrappers/video_recorder.py:72-79`). The writes are silently lost. This is latent today because the launcher passes `--video False`. | Return `self.env.unwrapped`. This changes behaviour only in video mode. |

---

## 4. PPO baseline: where it lives and whether it works

### 4.1 Compatibility verdict

Yes, under conditions.
- **Same API.** mjlab pins `rsl-rl-lib==5.2.0` (`<mjlab checkout>/pyproject.toml:49`). `uv.lock:1812-1814` resolves it to the vendored editable `rsl_rl` (`pyproject.toml:39-41`). The AMP code is in separate files, so `ppo.py` and `on_policy_runner.py` are stock 5.2.
- **mjlab's model config passes keys `MLPModel` rejects.** mjlab's `RslRlModelCfg` (`MJ/rl/config.py:7-38`) passes `cnn_cfg` and `rnn_*`, which the vendored `MLPModel.__init__` does not accept (`mlp_model.py:30-40`). So go through `MjlabOnPolicyRunner`, which strips them (`MJ/rl/runner.py:23-31`). The short names `"MLPModel"`/`"GaussianDistribution"` resolve fine (`utils.py:164-168`).
- **`upload_model` is missing from your config.** `MjlabOnPolicyRunner.save` reads `cfg["upload_model"]` (`MJ/rl/runner.py:80`). mjlab's config has it (`MJ/rl/config.py:123`); yours does not (`src/mjlab_g1/rl/config.py:100-134`). Use mjlab's config classes for the PPO tracking tasks, or add the field.
- **`obs_groups` defaults differ.** mjlab uses `{"actor": ("actor",), ...}` (`MJ/rl/config.py:92-94`); yours uses `{"actor": ("policy",), ...}` (`config.py:108-110`). The tracking env names its groups `actor`/`critic` (`tracking_env_cfg.py:109-120`).
- **Plain path also works.** A bare `OnPolicyRunner` + `PPO` with your `RslRlOnPolicyRunnerCfg` is fine too:
  - `rnd_cfg`/`symmetry_cfg` resolve to `None` (`extensions/rnd.py:240-241`, `extensions/symmetry.py:135-136`).
  - `share_cnn_encoders` is popped with a default (`ppo.py:431`).
  - `GaussianDistribution` accepts `learn_std`/`std_range` (`modules/distribution.py:144-150`).

### 4.2 Where to put it (recommended)

Create a user task package `src/mjlab_g1/tasks/tracking/` that builds on `MJ/tasks/tracking/tracking_env_cfg.py::make_tracking_env_cfg` (`:40-313`):
- Swap in your robot `get_g1_29dof_robot_cfg()` (`assets/g1_constants.py:221-232`) and `G1_29Dof_ACTION_SCALE` (`:237-245`; same 0.25·τ_max/k formula as `MJ/asset_zoo/robots/unitree_g1/g1_constants.py:278-286`).
- Apply the paper's MDP:
  - no pushes;
  - no end-effector termination;
  - pelvis height/tilt termination;
  - joint-limit weight −2;
  - no self-collision term;
  - Table A.1 observations.
- Register two IDs per motion set that share one env-config function: a `…-PPO` ID (`runner_cls=MotionTrackingOnPolicyRunner`, rl config from `MJ/tasks/tracking/config/g1/rl_cfg.py:10-46` with a new `experiment_name`) and a `…-SuperMimic` ID.
- PPO-Sim parity per §4.1: same motions, observations, reward, termination, DR and evaluation; BeyondMimic PPO hyperparameters; 8192 envs per GPU (twice SuperMimic's 4096).
- The paper also matches actor and critic network sizes. That is our choice to make: run mjlab's (512,256,128) as the standard baseline and add a size-matched PPO variant if compute allows.

Why not reuse mjlab's own task directly:
- `Mjlab-Tracking-Flat-Unitree-G1` sits in mjlab's separate registry (`MJ/tasks/registry.py:19`; `MJ/tasks/tracking/config/g1/__init__.py:7-21`), so your `train.py` cannot list it.
- It uses mjlab's G1 (knees-bent default pose, 30 bodies).
- Running it via `python -m mjlab.scripts.train …` works as a walking smoke test only.
- Both pyprojects define a `train` console script (`pyproject.toml:23-25` and `<mjlab checkout>/pyproject.toml:59-60`), so use `python -m` or the file path, never `uv run train`.

### 4.3 Motion data conversion (required for both learners)

- **What `MotionLoader` needs.** An npz with `joint_pos`, `joint_vel`, `body_{pos,quat,lin_vel,ang_vel}_w` at the control rate. It indexes bodies with the training robot's indices (`commands.py:36-60, 75-79`).
- **Why mjlab's converter can't be used as-is.** `MJ/scripts/csv_to_npz.py`:
  - reads CSV (`:47`) and converts quaternions from xyzw to wxyz (`:59-63`), but your `.npy` files are already wxyz;
  - runs forward kinematics with mjlab's G1 scene (`:366`) and stores every body in that robot's order (`:268-277`);
  - writes `/tmp/motion.npz` and requires W&B (`:308-325`).
- **Fix: write `scripts/npy_to_motion_npz.py`.** Load the wxyz `.npy`, resample 30 → 50 Hz (lerp/slerp as `csv_to_npz` does), run forward kinematics with your G1 entity (`G1_XML = g1_realsense.xml`, `g1_constants.py:19-21`), and save locally.
- **Where to save.** Put the outputs in `dataset/tracking/...`, never in `dataset/{locomotion,dualarm}`. The AMP loader opens every file there with `os.listdir` (`rsl_rl/rsl_rl/utils/motion_loader_g1.py:129, 158`).
- **Clip facts (parsed with the stdlib).**
  - Walking: 120 and 125 frames = 3.97 s and 4.13 s; pelvis height 0.735–0.783 m.
  - Lift: 209 frames = 6.93 s each; pelvis minimum 0.455 m and 0.393 m.
  - With 1 s bins (`commands.py:93`) that is only 5–8 bins per clip.

### 4.4 Reachability check (dataset combined with the actuator gains)

Required |a|max = max|q_ref − q0| / α, with α = 0.25·τ_max/k. That gives α = 0.351 for the knees and hip roll, 0.548 for hip pitch/yaw and waist yaw, 0.439 for shoulders, elbows, ankles and waist roll/pitch, and 0.075 for the wrists.

| Clip | Worst joints | q0 = your zero pose | q0 = mjlab knees-bent pose |
|---|---|---|---|
| locomotion1/2 | knees (≤1.51 rad), wrist roll | 4.2–4.6 | knees 2.3–2.4 |
| dual_arm1 | r wrist roll 13.2; knee (2.03 rad) | 5.8 (knee) | 3.9 (knee) |
| dual_arm2 | r wrist roll 20.0, l wrist roll 17.4; knees (2.24 rad) | 6.2–6.4 (knee) | 4.3–4.5 (knee) |

- These are static setpoint offsets. Under gravity, the low-gain PD needs a larger setpoint still.
- So both learners need `clip_actions=None`, which is the mjlab default. Never copy the dual-arm `clip_actions=1.5` (`tasks/dualarm/config/g1/rl_cfg.py:48`); see `tasks/dualarm/config/g1/env_cfgs.py:18-32` for why that task rescaled its actions.
- Both learners must use the same robot asset, default pose and action scale.

### 4.5 Evaluator

`MJ/tasks/tracking/scripts/evaluate.py` cannot be used here as written:
- it needs an online W&B run for both the motion and the checkpoint (`:63-69, :81-83`);
- it reads mjlab's registry (`:18`);
- it requires the `ee_body_pos` termination (`:92`), which the paper's MDP drops.

Write `scripts/eval_tracking.py`. It should load any registered `runner_cls`, use `sampling_mode="start"`, no pushes and nominal ξ, and report:
- the paper's **normalized return**: the undiscounted sum of per-step reward divided by its maximum (§4 intro). The maximum per step is Σ tracking weights 5.0 × dt 0.02 = 0.1 (Table A.2; all terms are ×0.02 per the Table A.6 caption);
- the failure rate under SuperMimic's termination (App. F.1);
- MPJPE via `MJ/tasks/tracking/mdp/metrics.py:13, 25`.

The existing AMP tasks (`Mjlab-G1-Locomotion` = velocity tracking, `Mjlab-G1-DualArm` = task reward + style) optimize different objectives. Report them as a separate reference row, not as the PPO baseline.

---

## 5. Slurm

**Current launcher (`scripts/train_slurm.sbatch`):**
- `:3` account `cchamzas`: invalid (verified).
- `:4` `quick`; `:6` `gpu:rtx_pro_6000_b:1`; `:9` 6 h.
- `:56` runs `uv sync --python "$(command -v python)"` in every job.
- `:58-64` fixed flags, including `--init_at_random_ep_len False`. That flag is right for dual-arm, but mjlab passes `True` for tracking (`MJ/scripts/train.py:176`).

**Cluster facts (sinfo / sacctmgr / scontrol / `--test-only`, 2026-10-22):**

| Partition | Time limit | QoS per-user GPU limit | GPU GRES present |
|---|---|---|---|
| quick | 12 h | **1** | H100 (2 nodes × 4), H200 (1 × 8), A100 (4 × 4 at 80 GB + 1 × 2 at 40 GB on whitehill-0-02), L40S, rtx_pro_6000_b (20 × 2, 96 GB) |
| short (default) | 1 day | 12 | same as quick |
| long | 7 days | 4 | A100, L40S, rtx_pro_6000_b; **no H100/H200** (test-only: "Requested node configuration is not available") |

- Validated: `mmaghelih` + `quick` + `rtx_pro_6000_b` is accepted; `mmaghelih` + `short` + `H100` is accepted.
- GRES strings are `H100`, `H200`, `A100`, `L40S` and `rtx_pro_6000_b`. To avoid the 40 GB A100, add `--exclude=whitehill-0-02`.

**Required changes:**
1. Use `--account=mmaghelih` (or `rbe577`), either in the file or as a CLI override.
2. Use `short` for the side-by-side runs, 4 runs × 1 GPU. The paper's costs are measured on H100s (§4 intro), so give both arms the same GPU type. There are only 8 H100s in the cluster, so expect queueing; rtx_pro_6000_b is the plentiful option. Compare at equal wall-clock on the same GPU type.
3. `uv sync`: the first sync downloads CUDA torch, so ask the user before running it. Run it once by hand, then guard line 56 with `if [ "${SKIP_SYNC:-0}" != 1 ]`, so that N concurrent jobs do not all sync the same `.venv`. Line 58 already uses `uv run --no-sync`.
4. Add variables `INIT_RANDOM` (default `False`) and `EXTRA_ARGS`. Tracking tasks pass `INIT_RANDOM=True`.
5. Use `/cm/shared/apps/slurm/current/bin/sbatch` if `sbatch` is not on `PATH`.
6. Wall clock: the paper's 1-H100 curve runs to about 9.4 h for the full library (Fig. A.1), which fits in `short`. Resume does not restore the replay buffer.

---

## 6. File plan

**New files (ours):**
```
src/mjlab_g1/rl/supermimic/__init__.py
src/mjlab_g1/rl/supermimic/config.py        # SuperMimicRunnerCfg(RslRlBaseRunnerCfg) + nested cfgs
src/mjlab_g1/rl/supermimic/replay.py        # SequenceReplayBuffer [num_envs, 1000, ...], k-window sampler with reset masks
src/mjlab_g1/rl/supermimic/world_model.py   # dynamics (2048x3), obs/reward/termination heads, normalizers, Eq. 1 loss
src/mjlab_g1/rl/supermimic/heads.py         # analytic BeyondMimic reward (Table A.2) + pelvis height/tilt termination on (s_hat, g)
src/mjlab_g1/rl/supermimic/actor_critic.py  # motion encoder / decoder actor, critic V(s, g, xi); TensorDict-forward wrapper for play/export
src/mjlab_g1/rl/supermimic/algorithm.py     # SuperMimic: construct_algorithm, wm_update, imagine+BPTT (Eq. 2), critic (Eq. 3), save/load
src/mjlab_g1/rl/supermimic/runner.py        # SuperMimicRunner(OnPolicyRunner): learn(); reuses Logger, multi-GPU, save/load shell
src/mjlab_g1/rl/supermimic/vecenv.py        # SuperMimicVecEnvWrapper: reset_done(env_ids), terminated/time_outs, state/xi/goal groups
src/mjlab_g1/envs/factory.py                # make_env(cfg, device, render_mode)
src/mjlab_g1/tasks/tracking/__init__.py
src/mjlab_g1/tasks/tracking/tracking_env_cfg.py  # make_g1_tracking_env_cfg(motion_files, ...) on top of MJ make_tracking_env_cfg
src/mjlab_g1/tasks/tracking/mdp/__init__.py      # ".mdp" is skipped by the registry walker (tasks/__init__.py:1) -- intended
src/mjlab_g1/tasks/tracking/mdp/commands.py      # MultiClipMotionCommand + "return" sampling (App. D.1)
src/mjlab_g1/tasks/tracking/mdp/observations.py  # future_ref_frames(K=10), wm_state, wm_context(xi), clean obs terms
src/mjlab_g1/tasks/tracking/config/g1/{__init__,env_cfgs,rl_cfg}.py   # registrations below
scripts/npy_to_motion_npz.py                # wxyz .npy @30 Hz -> FK with the user's G1 -> .npz @50 Hz, local, no W&B
scripts/eval_tracking.py                    # learner-agnostic: normalized return, failure rate, MPJPE
dataset/tracking/{walk,lift}/*.npz          # generated; NOT under dataset/locomotion or dataset/dualarm
```

**Changed files:**
- `src/mjlab_g1/scripts/train.py`: `:38-39` types, `:61` assert, `:98-111` replaced by the factory, YAML dump moved before `:160`.
- `src/mjlab_g1/scripts/play.py:99-108` and `src/mjlab_g1/scripts/render_video.py:71-84`: replaced by the factory.
- `src/mjlab_g1/rl/__init__.py`: export the new config.
- `src/mjlab_g1/rl/vecenv_wrapper.py:48-50`: `unwrapped` returns `self.env.unwrapped`; add a `reset(env_ids)` passthrough.
- `scripts/train_slurm.sbatch`: account, `SKIP_SYNC`, `INIT_RANDOM`, `EXTRA_ARGS`.
- Unchanged: `rsl_rl/`, the AMP task packages, `envs/g1_*_rl_env.py`.

**Task IDs (ours):**

| ID | Clips | Runner |
|---|---|---|
| `Mjlab-G1-Track-Walk-PPO` | `dataset/locomotion` → npz | `MotionTrackingOnPolicyRunner`, mjlab `RslRlOnPolicyRunnerCfg` |
| `Mjlab-G1-Track-Walk-SuperMimic` | same | `SuperMimicRunner`, `SuperMimicRunnerCfg` |
| `Mjlab-G1-Track-Lift-PPO` | `dataset/dualarm` → npz (robot only, no toaster) | as above |
| `Mjlab-G1-Track-Lift-SuperMimic` | same | as above |
| (extension) `…-Track-LiftBox-{PPO,SuperMimic}` | + toaster entity | see §7 |

**Config dataclass sketch** (paper values cite Table A.5 / A.4; `# ours` marks our defaults):
```python
@dataclass
class SuperMimicWorldModelCfg:
  hidden_dims: tuple[int, ...] = (2048, 2048, 2048)   # Table A.4
  window_k: int = 8                                   # §3.2, Table A.5
  learning_rate: float = 2e-4                         # Table A.5 (1 GPU)
  num_batches: int = 10; batch_windows: int = 2048    # Table A.5
  max_grad_norm: float = 1.0                          # Table A.5
  reward_head: Literal["analytic", "learned"] = "analytic"      # ours (paper: either, §3.2)
  termination_head: Literal["analytic", "learned"] = "analytic" # ours

@dataclass
class SuperMimicPolicyCfg:
  horizon: int = 32; gamma: float = 0.99              # Table A.5
  num_imagined: int = 4096                            # Table A.5
  betas: tuple[float, float] = (0.9, 0.95)            # Table A.5
  max_grad_norm: float = 1.0; elementwise_clip: float = 0.1     # Table A.5
  encoder_dims: tuple[int, ...] = (2048, 1024, 512, 256)        # Table A.4
  decoder_dims: tuple[int, ...] = (2048, 1024, 512, 256)        # Table A.4
  latent_dim: int = 32; future_frames: int = 10; obs_history: int = 10  # Table A.4, App. A

@dataclass
class SuperMimicCriticCfg:
  hidden_dims: tuple[int, ...] = (256, 128); lam: float = 0.95  # Table A.4, A.5
  num_epochs: int = 5; num_mini_batches: int = 4                 # Table A.5

@dataclass
class SuperMimicRunnerCfg(RslRlBaseRunnerCfg):
  class_name: str = "SuperMimicRunner"
  num_steps_per_env: int = 1                          # ours (UNSPECIFIED, see §8)
  max_iterations: int = 320_000                       # Table A.5
  action_noise_std: float = 0.1                       # Table A.5
  replay_steps_per_env: int = 1000                    # Table A.5
  lr_cosine_iters: int = 250_000; lr_final_frac: float = 0.03   # Table A.5
  clip_actions: float | None = None                   # required, §4.4
  world_model: SuperMimicWorldModelCfg = field(default_factory=SuperMimicWorldModelCfg)
  policy: SuperMimicPolicyCfg = field(default_factory=SuperMimicPolicyCfg)
  critic: SuperMimicCriticCfg = field(default_factory=SuperMimicCriticCfg)
  algorithm: dict = field(default_factory=lambda: {
      "class_name": "mjlab_g1.rl.supermimic.algorithm:SuperMimic",
      "rnd_cfg": None})                               # Logger reads rnd_cfg (logger.py:54)
```

**CLI examples (after implementation):**
```bash
# One-time: convert clips (FK with the user's G1)
PYTHONPATH=src uv run python scripts/npy_to_motion_npz.py --input dataset/dualarm/dual_arm1.npy \
  --output dataset/tracking/lift/dual_arm1.npz --input-fps 30 --output-fps 50

# PPO baseline (BeyondMimic hyperparameters, 8192 envs per §4.1)
PYTHONPATH=src uv run python src/mjlab_g1/scripts/train.py Mjlab-G1-Track-Lift-PPO \
  --env.scene.num_envs 8192 --agent.max_iterations 30000 --agent.run_name ppo_s0 \
  --video False --init_at_random_ep_len True

# SuperMimic (4096 envs, Table A.5)
PYTHONPATH=src uv run python src/mjlab_g1/scripts/train.py Mjlab-G1-Track-Lift-SuperMimic \
  --env.scene.num_envs 4096 --agent.max_iterations 320000 --agent.run_name sm_s0 --video False

# Slurm (after the launcher edits)
/cm/shared/apps/slurm/current/bin/sbatch --account=mmaghelih --partition=short --gres=gpu:H100:1 \
  --time=23:59:00 --export=ALL,TASK=Mjlab-G1-Track-Walk-SuperMimic,NUM_ENVS=4096,MAX_ITERS=320000,\
EXP_NAME=g1_track_walk,RUN_NAME=sm_s0,INIT_RANDOM=True,SKIP_SYNC=1 scripts/train_slurm.sbatch

# Evaluation
PYTHONPATH=src uv run python scripts/eval_tracking.py Mjlab-G1-Track-Walk-PPO \
  --checkpoint-file logs/rsl_rl/<exp>/<run>/model_<it>.pt --num-envs 1024
```

---

## 7. Scope: what "lifting" can mean

- **Tier A (in paper scope, the like-for-like comparison).** Track the robot-only squat-grasp-lift clips with the BeyondMimic reward, with no toaster in the scene. This tests whether the robot reproduces the 0.39 m squat and the arm motion. It is not a box lift.
- **Tier B (extension, outside the paper).** A real toaster lift needs all of the following:
  - the toaster pose and velocity in s;
  - object mass, CoM and friction in ξ (`dual_arm_env_cfg.py:190-243`);
  - an object/lift reward head, since the clips carry no object trajectory;
  - friction-only grasp contacts that a deterministic MLP must learn;
  - the virtual-PD wrench (`g1_dualarm_rl_env.py:655-674`) and the curricula treated as an unmodelled input, or added to ξ/s. The paper removes pushes precisely because "state discontinuities … the deterministic world model cannot capture" (App. A).
- **What the paper does not show.** The paper states "No terrain or object interaction … ongoing work" (§6). It also shows no SuperMimic advantage on one or two clips: its smallest settings are 1 h of data and LAFAN1 (40 clips, 2.45 h) (§4.1–4.2).

---

## 8. Hyperparameters and details the paper leaves UNSPECIFIED

| Item | Paper | Our default (labelled ours) | Justification |
|---|---|---|---|
| Contents of state s | Only "state s_t gives o_t and d_t" (§3.1) | Pelvis pos (relative to env origin) + quaternion + linear/angular velocity (13) + q, q̇ (58) + a_{t−1} (29) + the 14 tracked bodies' pos/quat/linear/angular velocity (182) = 282-d. Minimal alternative: 71-d with learned reward and observation heads. | The observation head needs a_{t−1}. The analytic reward needs body kinematics, and forward kinematics is not differentiable here (`MJ/__init__.py:28`). |
| Learned vs analytic heads | "either learned or computed analytically" (§3.2) | Analytic termination and reward; learned observation head | Closed forms exist (`terminations.py:27-55`; Table A.2) |
| Env steps per iteration; meaning of "UTD 0.4" | UTD 0.4, 1 policy update per WM update (Table A.5) | 1 vectorized step per iteration (`num_steps_per_env=1`) | Inferred, not stated: 320k × 4096 / 1.32e5 steps/s (MuJoCo Warp at 4096 envs, Fig. 5a) ≈ 2.8 h of pure simulation, consistent with the ~9.4 h single-GPU run (Fig. A.1) |
| G_max and the G_b update | "largest attainable discounted return"; "running estimate" (App. D.1) | G_max = 0.1/(1−γ) = 10; G_b as an EMA with α = 0.01 per finished visit | Maximum per-step reward = 5.0 × 0.02; with only 5–8 bins per clip, α needs to be faster than mjlab's 0.001 (`commands.py:593`) |
| Collection episode length | — | Clip length (≤ 7 s), resampled at clip end | mjlab's 10 s (`tracking_env_cfg.py:312`) exceeds every clip |
| Critic minibatch size | 5 epochs × 4 minibatches (Table A.5) | 4096 × 32 / 4 = 32,768 imagined transitions | Derived from Table A.5 |
| Observation history encoding | "last 10 observations" (App. A) | mjlab per-term flattened layout | Must match the deployed observation manager (`observation_manager.py:352-378`) |
| Action scale α and q0 | "q_sp = α a + q0", low BeyondMimic gains (App. A) | α = 0.25 τ_max/k (`g1_constants.py:237-245`); q0 = your zero pose; **no clip** | Same as your existing tasks; §4.4 shows clipping breaks reachability. If |a| ≈ 6 proves too hard, try mjlab's knees-bent q0 in **both** arms. |
| Reward anchor body | Reward says "root" (Table A.2); termination uses the pelvis frame (App. A) | Termination on pelvis; reward anchor stays torso_link (BeyondMimic, `MJ/tasks/tracking/config/g1/env_cfgs.py:41`) | Follows the paper's stated deviation only; flag it in results |
| Tracked body set 𝒯 | "set of tracked bodies" (Table A.2) | mjlab's 14 bodies (`MJ/tasks/tracking/config/g1/env_cfgs.py:42-57`) | BeyondMimic default |
| WM input/target normalization, activation, residual Δs | — | `EmpiricalNormalization` on inputs and targets; predict normalized Δs in the pelvis heading frame; ELU | rsl_rl defaults; heading frame removes the yaw/xy invariance the model would otherwise have to learn |
| Network sizes for 2-clip libraries | Table A.4 sized for 129k clips | Keep the Table A.4 sizes | Imagination compute is cheap; shrinking adds a confound |
| PPO-Sim hyperparameters | "BeyondMimic's PPO hyperparameters", 8192 envs/GPU, shared network sizes (§4.1) | `MJ/tasks/tracking/config/g1/rl_cfg.py:10-46` (LR 1e-3 adaptive, 5 × 4 minibatches, 24 steps, γ 0.99, λ 0.95, entropy 0.005, init_std 1.0, 30k iterations) plus an optional size-matched variant | mjlab's file is a BeyondMimic re-implementation (`tracking_env_cfg.py:1-10`) |
| Seeds | "several random seeds … figures do not report seed variation" (§4 intro) | 3 seeds per arm per task | Two arms × two tasks fits inside the 12-GPU `short` limit |

**Files I produced (scratchpad only, nothing in the repo):**
- `<scratch>/tools/npy_reach.py`: the stdlib reachability calculator behind §4.4.
- `<scratch>/paper_src/x/sections/`: the paper TeX source used for every paper citation (`method.tex`, `appendix.tex`, `results.tex`).