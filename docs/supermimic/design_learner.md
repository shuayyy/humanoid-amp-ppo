# SuperMimic learner design (G1 walk-tracking and box lift)

## 0. Decisions on points where the three reports disagree

| # | Topic | Spec report | Refs report | Integration report | **Decision (this design)** |
|---|---|---|---|---|---|
| D1 | State `s` | 310-d heading-local, carries body poses | generalized coordinates | 282-d, carries body poses | **Generalized coordinates (qpos/qvel, plus box) with a differentiable torch FK.** Warp FK has no backward pass (`MJ/__init__.py:28`). Body poses that the model predicts can drift away from `q`. A 31-body torch FK is exact and cheap. |
| D2 | UTD 0.4 | 2.5 vectorized steps per iteration | 2.5 | 1 | **2.5, from a fractional accumulator** (literal reading: 1 update per 2.5 vectorized steps). Config field `env_steps_per_iter`. The integration report's value of 1 is a listed ablation. |
| D3 | Termination head | learned BCE | – | analytic | **Learned BCE**, faithful to §3.1 "the learned model predicts d̂". Its inputs include the analytic e_pz and e_tilt, so it is smooth and easy to learn. |
| D4 | Clip end inside H | bootstrap | bootstrap | – | **Continuation 0 at clip end**, applied through a mask m_t (ours). Bootstrapping V at a state with no future is self-referential. The critic gets a `t_left` input. `"bootstrap"` is a config option. |
| D5 | G_max | per bin | – | constant 10 | **Per bin.** With clip-end episodes, a constant G_max makes late bins look permanently "hard". |
| D6 | Code location | – | – | `src/mjlab_g1/rl/supermimic/` | **Same.** CLAUDE.md marks `rsl_rl/` as the AMP implementation, so we do not add files under `rsl_rl/`. |

### Contract refinements the learner needs
Names are refined; the env/learner split is unchanged.
- **(a) Pre-reset terminal state.** `step()` must return the **pre-reset** robot and box qpos/qvel of `s_{t+1}`, either through `auto_reset=False` plus `reset(env_ids, ref_index)`, or through `terminal_qpos/qvel` fields. Without this, the k-windows and the BCE head never see a termination.
- **(b) Origin-relative root position.** Either `env_origins [N,3]`, or root position already relative to the env origin. The reference library is stored in env-local coordinates.
- **(c) Reference indexing.** Provide `ref_index [N]` (int, a global frame index) and a shared **`MotionLibrary`** handle with these tensors: `joint_pos/vel [F,29]`, `body_pos/quat/lin_vel/ang_vel_w [F,B,·]`, `box_ref_pos/quat [F,7]`, and `clip_start/clip_last [C]`. Imagination needs frames k…k+H+K, not just the env's next K.
- **(d) qvel[3:6] frame.** In MuJoCo, free-joint angular velocity is **body-local**, but the contract says "world". The learner assumes **world**, i.e. mjlab `root_link_ang_vel_w`. A parity test checks this against `root_link_ang_vel_b`.
- **(e) DR ranges.** Provide `xi_lo/xi_hi [D]` from the env's DR config for the [−1,1] encoding, instead of hard-coding Table A.3.
- **(f) Reward pairing offset.** Provide `reward_ref_offset ∈ {0,1}`. With the default 0 (mjlab), r_{t+1} and d_{t+1} are evaluated on s_{t+1} against the **pre-advance** frame ref[k_t] (`MJ/envs/manager_based_rl_env.py:436-456`). The goal frames and the analytic reward and termination all use this one offset.
- **(g) Assist forces.** Lift env: **no reference-dependent assist wrench and no curriculum force.** f_ω(ŝ,a,ξ) has no g input, so dynamics that depend on g are not Markov for the model.

---

## 1. Fidelity table
"Report followed" is given only where the reports disagreed.

| Component | Paper | This design | Status | Report followed |
|---|---|---|---|---|
| Loop (collect, then WM, then actor, then critic, every iteration) | Alg. A.1 | same | faithful | – |
| Sim envs per GPU | 4096 (Table A.5) | 4096 | faithful | – |
| Collection noise | ε~N(0,0.1²); stores the executed action (§3.1, Table A.5) | same | faithful | – |
| Replay | FIFO, 1000 steps/env (Table A.5) | ring [N,1000]; also stores pre-reset s_{t+1} | faithful + ours | – |
| Window sampling | UNSPECIFIED | uniform over valid starts; per-step mask after an episode end | ours (refs #11) | – |
| UTD 0.4 | ambiguous (Table A.5) | 2.5 vectorized steps per iteration | ours | spec (a), not integration |
| "10 batches × 2048" | Table A.5 | 10 sequential Adam steps | ours (spec #7) | – |
| f_ω width | MLP 2048×3 (Table A.4) | same; Linear→LayerNorm→SiLU | faithful width; activation ours (refs #5) | refs (spec said ELU) |
| Deterministic, ξ-conditioned, no ensemble | §3.2 | same | faithful | – |
| State content | UNSPECIFIED | generalized coordinates + torch FK features | ours (refs #1) | refs (D1) |
| Prediction target | UNSPECIFIED | Δ generalized velocity + small position residual, semi-implicit Euler, zero-init | ours (refs #2; SuperTrack §3.4, §6.3) | – |
| Normalization | UNSPECIFIED | running stats for inputs; **fixed** mocap-derived action scale; σ_nom for targets | ours | – |
| WM loss | Eq. 1: k=8, unit weights, logged actions, no detach (§3.2–3.3) | same; state MSE is the mean over normalized dimensions | faithful + ours | – |
| Observation head | "learned or analytic" (§3.2) | analytic | ours (all three agree) | – |
| Reward head | same | analytic torch (Table A.2 + box terms) | ours | – |
| Termination head | "learned model predicts d̂" (§3.1) | learned BCE, MLP (256,128) | faithful | spec (D3) |
| WM optimizer | Adam 2e-4, global clip 1.0 (Table A.5) | same; β=(0.9,0.999), eps 1e-8 | faithful + ours (refs #17) | – |
| Policy obs | Table A.1, 93/frame × 10 | same, computed from raw tensors | faithful | – |
| Goal | K=10 future frames, content UNSPECIFIED | q_ref, q̇_ref, ref-anchor orientation (6D) relative to the robot; offsets tied to `reward_ref_offset` | ours (spec §1.3) | – |
| Actor | enc and dec (2048,1024,512,256); z=32, tanh (Table A.4) | same; ELU; final layer ×0.01; linear unclipped output | faithful + ours (spec #24) | – |
| Imagination | H=32, 4096 rollouts, η re-added, no action noise (§3.3) | same | faithful | – |
| Start states | from B with recorded ξ (§3.3) | uniform over non-terminal slots with m=1 | faithful + ours | – |
| Survival weight | ĉ (§3.1, Eq. 2) | same | faithful | – |
| Actor loss reduction | UNSPECIFIED | −mean_batch(J)/H (SHAC Eq. 5) | ours | – |
| Actor optimizer | Adam (0.9,0.95), LR=α_ω, global 1.0 then element-wise 0.1 | same | faithful | – |
| Critic | V(ŝ,g,ξ) (256,128), no target net, λ=0.95, 5×4 (Eq. 3) | adds a `t_left` input; zero-init output | faithful + adapted | – |
| Clip end in imagination | UNSPECIFIED | continuation 0 + mask | ours | D4 |
| Schedule | cosine to 0.03× over 250k iterations, then constant; 320k iterations | same, on all three optimizers | faithful | – |
| Warm-up | none (Alg. A.1; FAQ) | prefill 32 steps/env; no WM-only phase | faithful + ours | – |
| Reset sampling | App. D.1 mixture (0.5 uniform / 0.4 softmax / 0.1 past weight, temperature 25) | mixture faithful; G_b and G_max rules ours | faithful + ours | D5 |
| Precision | TF32 (Fig. 5) | TF32 on | faithful | – |
| Box in s / ξ / obs / goal | none (§6: "no object interaction") | added | adapted (outside paper) | – |
| Velocity interface | App. B, G | not implemented (walking = clip tracking) | omitted | – |

---

## 2. Features (all computed in torch from raw contract tensors)

### Notation
- **Raw step.** `qpos[N,36] = [p_root (env-local), quat wxyz, q(29)]`; `qvel[N,35] = [v_root (world), ω_root (world), q̇(29)]`. For lift, add `bqpos[N,7]` and `bqvel[N,6]` (world).
- **Frames.**
  - R = pelvis rotation.
  - R_H = yaw-only rotation of the pelvis (heading frame).
  - ω_b = Rᵀω.
- **`TorchFK`** (built once from `g1_realsense.xml` using CPU `mujoco`: `body_parentid`, `body_pos`, `body_quat`, `jnt_axis`, `jnt_pos`, `jnt_bodyid`, `site_pos`).
  - Returns world pos/quat/lin-vel/ang-vel of the requested bodies and sites by recursive spatial-velocity propagation, with link-origin velocity (as in mjlab `body_link_lin_vel_w`).
  - Bodies at the same tree depth are batched together.

### 2.1 World-model state `s` (stored and integrated; not a network input)

| Part | Walk | Lift |
|---|---|---|
| Robot qpos | 36 | 36 |
| Robot qvel | 35 | 35 |
| Box qpos / qvel | – | 7 + 6 |
| **Total s** | **71** | **84** |
| Carried, not predicted: a_{t−1} | 29 | 29 |

a_{t−1} is not an f_ω input: the mjlab PD actuators have no action delay or filter.

### 2.2 WM input features φ(s), heading-invariant

| Feature | Dim |
|---|---|
| pelvis height z | 1 |
| rot6d(R_Hᵀ R) | 6 |
| R_Hᵀ v_root | 3 |
| ω_b | 3 |
| q − q0 | 29 |
| q̇ | 29 |
| 14 tracked bodies (`env_cfgs.py:42-57`): R_Hᵀ(p_b − p_pelvis) | 42 |
| 14 tracked bodies: R_Hᵀ v_b | 42 |
| 2 ankle_roll links: rot6d(R_Hᵀ R_foot) | 12 |
| 2 ankle_roll links: world z | 2 |
| **φ_robot** | **169** |
| Lift: R_Hᵀ(p_box − p_pelvis) 3, rot6d(R_Hᵀ R_box) 6, R_Hᵀ v_box 3, R_Hᵀ ω_box 3, box z 1, both palm sites in the box frame 6 | +22 → **191** |

- **f_ω input** = `[norm(φ), a / σ_a, enc(ξ)]`: 169 + 29 + 33 = **231** (walk) and 191 + 29 + 35 = **255** (lift).
- **σ_a (ours):** a fixed per-joint scale equal to the std over the motion library of (q_ref − q0)/α, floored at 0.5. The actions need |a| up to about 6 (knees) and 20 (wrist roll) (integration §4.4). Running stats from a near-zero initial policy would therefore be degenerate.

### 2.3 Policy observation o (Table A.1), per frame, 10-frame history

| Term | Dim | Formula from raw | Noise η (uniform) |
|---|---|---|---|
| q − q0 + Δq0 | 29 | Δq0 = encoder bias from ξ | ±0.01 |
| q̇ | 29 | qvel[6:] | ±0.5 |
| ω_b | 3 | Rᵀ qvel[3:6] | ±0.2 |
| projected gravity | 3 | Rᵀ[0,0,−1] | ±0.05 |
| a_{t−1} | 29 | carried (0 at reset) | none |
| **Per frame** | **93 → 930** | | |
| Lift (ours): Rᵀ(p_box − p_pelvis) 3, rot6d(Rᵀ R_box) 6 | +9 → 102 → **1020** | | ±0.01 m / ±0.02 (ours) |

- **`PolicyObsHistory`** (one class, used in both sim and imagination; spec #13):
  - layout [N,10,F] with frames concatenated oldest to newest (a frame-major layout we choose; the learner owns this layout, so the deployed policy sees the same one);
  - at reset, all 10 slots are filled with the first noisy frame;
  - η is drawn fresh for each new frame.

### 2.4 Goal g, K = 10 frames
- **Frames used:** k_t + `reward_ref_offset` + {0..9}, clamped to `clip_last`. Under mjlab pairing, this includes the frame the immediate reward is computed against.

| Per frame | Dim |
|---|---|
| q_ref | 29 |
| q̇_ref | 29 |
| rot6d(R_pelvisᵀ R_ref_pelvis,k), recomputed from ŝ at every imagined step (differentiable) | 6 |
| **Walk** | **64 → 640** |
| Lift: R_Hᵀ(p_boxref,k − p_pelvis) 3, rot6d(R_pelvisᵀ R_boxref,k) 6 | +9 → 73 → **730** |

### 2.5 Domain-randomization context
- **Encoding:** enc(ξ) = 2(ξ − lo)/(hi − lo) − 1. Dimensions with hi = lo are set to 0.
- **Walk ξ:** [μ_foot (1), Δq0 (29), Δcom_torso (3)] = **33**.
- **Lift ξ:** adds box mass (1) and box friction (1) = **35**, plus 3 for box CoM if the env randomizes it.
- **Storage:** ξ is stored per step, so both startup-mode and reset-mode DR work.

---

## 3. World model

### 3.1 Dynamics f_ω
- **Network:** MLP 231/255 → 2048 → 2048 → 2048 → out, each layer Linear → LayerNorm → SiLU (activation ours). The final layer is **zero-initialized**, so at init the model behaves as constant velocity.
- **Output y** (de-normalized as y ⊙ σ_out, with σ_out = std of the residual targets, cumulative and frozen within an update):
  - Δν: R_H-frame Δv_root 3, Δω_b 3, Δq̇ 29, and for lift box Δv (heading frame) 3 and Δω 3;
  - δx: R_H-frame δp_root 3, δθ_root (body axis-angle) 3, δq 29, and for lift box δp 3 and δθ 3.
  - Size: 70 (walk) / 82 (lift).
- **Integration** (dt = 0.02):
  - velocities first: v′ = v + R_H Δv, ω_b′ = ω_b + Δω_b, q̇′ = q̇ + Δq̇;
  - then positions with the new velocities: p′ = p + dt·v′ + R_H δp, quat′ = quat ⊗ exp(dt·ω_b′ + δθ), q′ = q + dt·q̇′ + δq;
  - the box uses the same scheme.
- **Lineage:** SuperTrack §3.4 and §6.3; the δx residual is ours. It absorbs the 4-substep integration mismatch.

### 3.2 Heads

| Head | Type | Definition |
|---|---|---|
| Observation ō | analytic | §2.3 table applied to ŝ, ξ, a_{t−1}; no loss term (Eq. 1) |
| Reward r̂_{t+1} | analytic | §4.3 on (ŝ_{t+1}, ref[k_t + off], a_t, a_{t−1}); no loss term |
| Termination d̂_{t+1} | learned | σ(MLP(256,128, ELU) over [norm φ(ŝ_{t+1}), e_pz, e_tilt, ref anchor z, (lift) ‖e_box‖]); BCE against the logged `terminated` (time-outs give d = 0) |

### 3.3 Loss (Eq. 1)
For a window with start s_0, logged actions a_0..a_7 and per-step mask m_j (0 after an episode end; a terminal step itself counts):

```
ŝ_0 = s_0;  ŝ_j = f_ω(ŝ_{j−1}, a_{j−1}, ξ)            # no detach (§3.3)
e_j = [R_H,trueᵀ(p̂−p), log(R̂Rᵀ), q̂−q, R_Hᵀ(v̂−v), ω̂_b−ω_b, q̂̇−q̇, (box 12)] / σ_nom
L_WM = mean_batch Σ_{j=1..8} m_j [ mean_d(e_jd²) + BCE(d̂_j, d_j) ]
```

- **σ_nom (ours):** the cumulative std of the one-step change in each dimension. The same normalized 8-step error is logged as the Fig. 3c metric.

### 3.4 Optimization and data

| Item | Value |
|---|---|
| Optimizer | Adam 2e-4, β = (0.9, 0.999), eps 1e-8, no weight decay |
| Gradient clip | global norm 1.0 per step (Table A.5) |
| Updates per iteration | 10 Adam steps × 2048 windows × k = 8 |
| Normalizers | `EmpiricalNormalization` on φ (cumulative, updated on every collected transition, frozen inside each update) |
| Collection | a_exec = π(o, g) + N(0, 0.1²) per dimension; the buffer stores a_exec, unclipped (`clip_actions=None`) |

### 3.5 Replay buffer `SequenceReplay`
GPU ring [N = 4096, T = 1000]. FIFO is per env by write pointer.

| Field | Dtype / dim |
|---|---|
| `s` (qpos, qvel, box) | f32, 71 / 84 |
| `s_next` (pre-reset) | f32, 71 / 84 |
| `a_exec`, `a_prev` | f32, 29 + 29 |
| `xi` | f32, 33 / 35 |
| `r` (total), `r_terms` | f32, 1 + ≤12 |
| `term`, `trunc` | bool |
| `ep_step` (steps since reset) | int16 |
| `ref_idx` | int32 |

- **Size:** about 255 floats per slot, so 4096 × 1000 × ~1 KB ≈ **4.3 GB** (lift ≈ 4.7 GB).
- **`sample_windows(B, k)`:** uniform over (env, t0) with t0 in valid history. m_j = 1 while no `term`/`trunc` occurred at steps < j. `s_next` supplies the target at the last valid step.
- **`sample_starts(B)`:** uniform over slots with `term = 0` and the reference not past the clip end. It returns s, a_prev, ξ, ref_idx and the **clean** 10-frame history, rebuilt from slots t − min(i, ep_step) (this pads with the episode's first frame, matching the sim's reset padding).

---

## 4. Actor and critic via BPTT

### 4.1 Imagination (Eq. 2), 4096 starts, H = 32
WM and critic parameters are set `requires_grad_(False)` for the actor step; gradients still flow through their inputs.

```
s, a_prev, Hst, ξ, k = start;  Hst += η (fresh per frame);  c = 1;  J = 0
for t in 0..31:
    g  = goal(s, k+off)                        # 640/730, depends on s
    a  = actor(norm(Hst), norm(g))             # deterministic, no action noise
    s1 = wm.step(s, a, ξ)
    r  = reward(s1, lib.frame(k+off), a, a_prev)          # analytic
    d  = wm.term_prob(s1, lib.frame(k+off))
    m  = lib.alive(k)                          # 0 once k > clip_last (truncation parity-tested vs env)
    J += γ^t · c · m · r
    c  = c · (1 − d)
    Hst = push(Hst, obs_frame(s1, a, ξ) + η);  a_prev = a;  s = s1;  k += 1
J += γ^32 · c · m_32 · V(φ(s), g_32, ξ, t_left)
loss_π = −mean(J) / 32                         # SHAC Eq. 5 reduction (ours)
backward → clip_grad_norm(actor, 1.0) → grad.clamp_(−0.1, 0.1) → Adam(β=(0.9,0.95))
```

- **Gradient paths into θ:**
  - r̂ through ŝ_{t+1} = f_ω(ŝ_t, a_t) (WM Jacobians with respect to state and action);
  - the smoothness term through a_t − a_{t−1};
  - ĉ through d̂;
  - ō (built from ŝ) into later actions;
  - g through the robot-relative orientation;
  - V through ŝ_32.
- **No gradient into the env:** every raw tensor is detached at ingestion, and the policy step uses only buffer tensors.

### 4.2 Analytic reward (Table A.2, every term × dt = 0.02, weights from the env config)
Re-anchoring, as in mjlab `commands.py:396-405`:
- Δq = yaw(q_robot,anchor ⊗ q_ref,anchor⁻¹)
- p_off = [p_robot,anchor,xy, p_ref,anchor,z]
- p̃_b = p_off + Δq·(p_ref,b − p_ref,anchor)
- q̃_b = Δq ⊗ q_ref,b

Terms:
- **Body tracking:** body pos/ori use p̃ and q̃ with σ = 0.3 / 0.4. Body lin/ang velocity use global references with σ = 1.0 / 3.14. Weight 1 each.
- **Anchor ("root") pos/ori:** global, weight 0.5 each. The anchor body is read from the env config (`anchor_body`).
- **Action rate:** −0.1·‖a_t − a_{t−1}‖².
- **Joint limit:** −2 × soft-limit violation, using the env's limit factor.
- **Lift:** the env team's box terms re-implemented one-to-one in torch. Palm sites come from `TorchFK.sites`. Terms that need contact forces are **not allowed**; the env team should use distance-based proxies.
- **Mandatory startup parity check:** on about 10k real transitions, compute max over terms of |r_analytic − r_env| and abort if it exceeds 1e-4. The same check compares the analytic e_pz/e_tilt thresholds with `terminated`.

### 4.3 Critic (Eq. 3)
- **Input:** V(norm φ(ŝ), norm g, enc ξ, t_left), with t_left = clamp(clip_last − k, 0, 64)/64. The `t_left` input is adapted.
- **Network:** MLP (256,128), ELU, zero-init output.
- **Targets** (computed once on the detached imagined rollout):
  - R̂_32 = V_32
  - R̂_t = r̂_{t+1} + γ(1 − d̂_{t+1})·m_{t+1}·[(1−λ)V_{t+1} + λR̂_{t+1}], with λ = 0.95 and γ = 0.99
  - loss masked by m_t
- **Fit:** 5 epochs × 4 shuffled minibatches of 32,768; Adam 2e-4; clip 1.0; no target network.
- **Ordering:** the actor step uses the previous iteration's critic (Alg. A.1, lines 7–8).

### 4.4 Return-based resets (App. D.1)
`ReturnBasedResetSampler` is a shared, learner-agnostic module; the protocol engineer decides whether PPO uses it too.
- **Bins:** 1 s bins per clip.
- **G_b:** EMA with rate 0.01 (ours) of the discounted, episode-truncated return of episodes that **start** in bin b.
- **G_max,b:** 0.1·(1 − γ^{T_b})/(1 − γ), where T_b is the number of steps from the bin's midpoint to `clip_last` (ours).
- **Past weight w̄_b:** EMA with rate 0.001 per iteration of the softmax term (ours).
- **Mixture:** P = 0.5·U + 0.4·softmax(−25·G_b/G_max,b) + 0.1·w̄/Σw̄.
- **Start frame:** uniform within the bin. The env writes the reference state plus its reset noise.

---

## 5. Training loop and ratios

```python
prefill: 32 vec steps/env with the init actor (≈0 actions) + ε    # ours: windows + normalizer/σ_nom stats
acc = 0.0
for it in range(320_000):                                           # Table A.5
    acc += 2.5; n = int(acc); acc -= n                              # UTD 0.4 → 2,3,2,3,… (ours)
    for _ in range(n): collect_step()                               # noisy act, B.add, sampler.update, reset(done, sampler.sample)
    for _ in range(10): wm_step(B.sample_windows(2048, 8))          # Eq. 1
    rollout = actor_step(B.sample_starts(4096))                     # Eq. 2, H=32
    critic_fit(rollout)                                             # Eq. 3, 5×4
    sched.step()                                                    # cosine→0.03 over 250k, then const (all three)
```

- **Per-iteration ratio:**
  - env: 2.5 × 4096 = 10,240 transitions;
  - WM: 10 updates (20,480 windows, 163,840 predicted steps);
  - actor: 1 update (131,072 imagined steps);
  - critic: 20 minibatch steps.
- **Buffer turnover:** every 400 iterations.
- **Warm-up:** no WM-only phase (faithful).
  - Fallback (ours, refs #13): if the actor's gradient norm or |a| diverges in the first 2k iterations, restart with 2,000 WM-only iterations.
  - Horizon-ramp fallback (refs #14): ramp H from 8 to 32 only if the pre-clip gradient norm explodes.

---

## 6. Module and file plan (`src/mjlab_g1/rl/supermimic/`)

```
config.py        SuperMimicRunnerCfg(RslRlBaseRunnerCfg) + WM/Policy/Critic/Sampler sub-cfgs
features.py      quat utils (wxyz); TorchFK.from_xml(path, bodies, sites)
                   .forward(qpos[*,36], qvel[*,35]) -> BodyKin(pos[*,B,3], quat[*,B,4], lin_vel, ang_vel, site_pos[*,S,3])
                 wm_features(s, kin) -> [*,169|191]; obs_frame(s, a_prev, xi) -> [*,93|102]
                 goal(s, kin, lib, k, off) -> [*,640|730]; encode_xi(xi, lo, hi); sensor_noise(shape)
                 PolicyObsHistory(N, 10, F): reset(ids, frame), push(frame), get() -> [N,10*F]
reference.py     MotionLibrary(npz tensors, clip_start, clip_last): frame(k), alive(k), t_left(k)   # shared with env/PPO
reward.py        TrackingReward(env_reward_cfg): __call__(s1, kin1, ref, a, a_prev) -> (r[*], terms{})
                 term_errors(kin1, ref) -> (e_pz, e_tilt[, e_box])
world_model.py   WorldModel(nn.Module): step(s[*,71|84], a[*,29], xi_enc) -> s'; term_prob(s', ref) -> [*]
                 window_loss(W) -> (loss, metrics); normalizers φ, σ_out, σ_nom
actor_critic.py  MotionEncoder(640|730 -> 32, tanh), Decoder(32+930|1020 -> 29), Actor(hist, goal)
                 Critic(phi, goal, xi_enc, t_left) -> [*]; ActorTensorDictWrapper (play/ONNX)
replay.py        SequenceReplay(N, 1000, spec): add(dict), sample_windows(B,k) -> Window, sample_starts(B) -> Start
reset_sampler.py ReturnBasedResetSampler(lib, bin_s=1.0, mix=(.5,.4,.1), temp=25, ema=0.01, slow_ema=0.001)
imagination.py   imagine(start, actor, wm, reward, critic, lib, H, gamma) -> Rollout(J, s, g, r, d, m, V)
algorithm.py     SuperMimic: construct_algorithm(obs, env, cfg, device) [staticmethod]; act(raw); wm_update();
                 actor_update(); critic_update(rollout); save() -> dict; load(d, load_cfg, strict)
runner.py        SuperMimicRunner(OnPolicyRunner): learn(num_learning_iterations, init_at_random_ep_len),
                 get_inference_policy(device), save/load (Logger with rnd_cfg=None, action_std=full(29, 0.1))
tests/supermimic/test_*.py
```

```python
@dataclass
class SuperMimicRunnerCfg(RslRlBaseRunnerCfg):
  class_name: str = "SuperMimicRunner"
  max_iterations: int = 320_000                 # Table A.5
  env_steps_per_iter: float = 2.5               # ours (UTD 0.4, reading a); ablation 1.0
  prefill_steps: int = 32                       # ours
  action_noise_std: float = 0.1                 # Table A.5
  replay_steps_per_env: int = 1000              # Table A.5
  lr: float = 2e-4; lr_final_frac: float = 0.03; lr_cosine_iters: int = 250_000   # Table A.5
  clip_actions: float | None = None             # required (integration §4.4)
  reward_ref_offset: int = 0                    # from env (mjlab pairing)
  clip_end: Literal["stop", "bootstrap"] = "stop"   # ours (D4)
  wm: WMCfg = WMCfg(hidden=(2048,)*3, k=8, n_batches=10, batch=2048, max_grad_norm=1.0, act="silu_ln")
  policy: PolicyCfg = PolicyCfg(H=32, n_imag=4096, gamma=0.99, betas=(0.9,0.95), max_grad_norm=1.0,
                                elem_clip=0.1, enc=(2048,1024,512,256), dec=(2048,1024,512,256),
                                z=32, K=10, hist=10, out_init_scale=0.01)
  critic: CriticCfg = CriticCfg(hidden=(256,128), lam=0.95, epochs=5, minibatches=4)
  sampler: SamplerCfg = SamplerCfg()
  memory: MemCfg = MemCfg(imag_microbatches=1, checkpoint_steps=False, compile=True)
  save_replay: bool = False                     # ours: exact resume across 24 h jobs
  algorithm: dict = {"class_name": "mjlab_g1.rl.supermimic.algorithm:SuperMimic", "rnd_cfg": None}
```

- **Checkpoint contents:**
  - state dicts: actor, critic, world model (including φ normalizer, σ_out, σ_nom), and the 3 optimizers;
  - scheduler step, iteration, sampler state (G_b, w̄), RNG states, cfg;
  - optional `replay_<it>.pt` (about 4.5 GB).
  - PPO-compatible keys are kept (`actor_state_dict`, `critic_state_dict`).
- **Logged metrics:**
  - `wm/loss_state`, `wm/loss_bce`, `wm/err_norm_step{1..8}` (on fresh, not-yet-trained windows), `wm/grad_norm`;
  - `actor/J`, `actor/grad_norm_preclip`, `actor/frac_elem_clipped`, `actor/|a|_p50,p99`;
  - `imag/c_H`, `imag/r_mean` against `sim/r_mean`;
  - `critic/loss`, `critic/expl_var`;
  - `sim/norm_return`, `sim/term_rate`, `sim/r_<term>`;
  - `parity/r_maxabs`, `parity/term_acc`;
  - `sampler/P_b`, `sampler/G_b`;
  - `perf/{sim,wm,actor,critic}_ms`, `perf/mem_gb`.

---

## 7. GPU memory and time (4096 × H = 32)
These figures are our estimates; profile 200 iterations before committing to a schedule.

### 7.1 Memory
**Activations per imagined sample-step:**

| Source | Floats |
|---|---|
| WM (3 × LayerNorm/SiLU layers of 2048, saved tensors) | ≈19k |
| Actor encoder + decoder | ≈17k |
| History + goal tensors | ≈1.6k |
| FK (31 bodies) | ≈1.7k |
| Reward / termination | ≈1k |
| **Total** | **≈40k floats ≈ 160 KB** |

| Item | Memory |
|---|---|
| Imagination activations (×131,072 sample-steps) | **≈21 GB** |
| Replay | 4.3–4.7 GB |
| WM fit (2048 × 8) | ≈1.3 GB |
| Parameters + Adam (≈19M params) | ≈0.3 GB |
| MuJoCo Warp, 4096 envs | measure (assume ≤10 GB) |
| **Total** | **≈30–40 GB**; fits H100 80 GB and RTX PRO 6000 96 GB |

- **Sanity check:** scaling to H = 128 gives about 84 GB, consistent with §4.3 ("H = 128 is the largest that fits on an H100").

### 7.2 Time
- **FLOPs per iteration:** WM fit ≈ 8.9 TFLOP, imagination ≈ 11.7 TFLOP, critic ≈ 1 TFLOP, so ≈ **21.6 TFLOP**. At 40–60% of the H100's TF32 dense peak this is ≈ 70–110 ms.
- **Simulation:** 2.5 × 4096 steps at roughly 1–1.3·10⁵ steps/s (read off Fig. 5a by the reports) is ≈ 80–100 ms.
- **Kernel launches:** 32 sequential steps × FK tree depth × small ops cost ≈ 50–150 ms in eager mode. Wrap the imagined step with `torch.compile` or a CUDA graph.
- **Total:** ≈ **0.2–0.3 s per iteration**, so 320k iterations take ≈ 18–27 h. That is above the 24 h `short` limit. Either resume once with `save_replay=True`, or compare at equal wall-clock (protocol's call).
- **D2 trade-off:** the 1-step reading would take ≈ 13 h, closer to the paper's implied ≈ 110 ms per iteration.

### 7.3 Fallback order if memory does not fit
1. Gradient accumulation over 2–4 imagined micro-batches, with clipping applied after accumulation. The gradient is exact.
2. `torch.utils.checkpoint` per imagined step: memory drops to about 2 GB at ≈ 1.3× compute.
3. bf16 autocast for MLPs, flagged as a deviation from TF32.

Never reduce H or N.

---

## 8. CPU unit tests (`tests/supermimic/`)

| Test | Assertion |
|---|---|
| `test_fk_matches_mujoco` | Random qpos/qvel on `g1_realsense.xml`: TorchFK pos/quat/lin-vel/ang-vel vs CPU `mj_kinematics` + `mj_objectVelocity` (link origin, world), max error < 1e-5 |
| `test_feature_shapes` | s, φ, o, g and enc(ξ) have dims 71/84, 169/191, 930/1020, 640/730, 33/35; φ is invariant to world xy and yaw |
| `test_obs_parity` | `obs_frame` on mjlab CPU data equals `joint_pos_rel(biased)`, `base_ang_vel`, `projected_gravity` (catches the qvel[3:6] frame convention) |
| `test_history_padding_parity` | Sim `PolicyObsHistory` after a reset plus n steps equals the `sample_starts` rebuild from replay (clean) for n < 10 |
| `test_window_sampler_no_cross_reset` | Synthetic resets/terminals: every window has m_j = 0 after the first `term`/`trunc`; the target at a terminal step is `s_next` |
| `test_wm_integration_identity` | With a zero-initialized output, `step` equals the constant-velocity integrator; quaternions stay unit to 1e-6 |
| `test_wm_toy_fit` | Toy damped point-mass chain (s 6-d, a 2-d, ξ = mass): 8-step loss falls ≥ 10× in 2k CPU steps with a small MLP |
| `test_bptt_toy_improves` | Toy tracking task through the fitted toy WM: imagined return rises monotonically over 200 actor steps (smoothed) |
| `test_grad_flow` | After the actor step: all actor params have non-zero gradients; WM and critic params have `grad is None`; buffer and env tensors have `requires_grad == False` |
| `test_clip_order` | A gradient with a huge norm and outliers is clipped first to global norm 1.0, then every element to ≤ 0.1 |
| `test_gae_targets` | Eq. 3 recursion with d̂ and m equals the brute-force λ-return on random tensors, including the clip-end mask |
| `test_survival_weight` | ĉ_t = Π(1 − d̂) and J match a hand computation; d̂ = 1 at step 3 zeroes later terms |
| `test_reset_sampler` | Mixture probabilities sum to 1; a bin with low G_b/G_max,b gets more mass; per-bin G_max matches the closed form |
| `test_reward_formulas` | Analytic terms at an identical robot/reference state reach their maxima (exp terms = 1); re-anchoring is invariant to robot xy/yaw offsets |
| `test_checkpoint_roundtrip` | save → load restores identical actor outputs, WM predictions and sampler state |

The GPU-only startup checks (§4.2 parity, truncation-timing parity) run inside `SuperMimicRunner.learn()` before iteration 0.

No tool calls were made; this design was written from the three verified reports.