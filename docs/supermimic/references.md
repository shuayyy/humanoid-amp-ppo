# SuperMimic gap-filling: what the learner builds on, from primary sources, and defaults for the unspecified details

**Sources.** The SuperMimic text is from the arXiv PDF (2610.28480v1). The HTML version (`arxiv.org/html/2610.28480v1`) returned 404 on 2026-10-22, so the citation check below uses the PDF reference list on pp. 10–12. Every reference PDF was downloaded and read. Text extracts are in `<scratch>/refs/<name>/paper.txt`, and the code files I read are in `.../scratchpad/refs/{shac,mbpo,rwm}/code/`. mjlab paths below are relative to `<mjlab v1.4.0>/src/`.

**Labels.**
- **SPECIFIED**: SuperMimic states it.
- **UNSPECIFIED**: SuperMimic does not state it.
- **ours**: a default I propose, not taken from any paper.

---

## 1. What SuperMimic actually cites (PDF refs [1]–[28])

| Ref | Work | How SuperMimic uses it |
|---|---|---|
| [19] | SHAC (Xu et al., ICLR 2022) | "motivating SuperMimic's critic-bootstrapped objective" (§2). The actor–critic is "in the spirit of SHAC" (§3.3). |
| [12] | SuperTrack (Fussell et al., TOG 2021) | A learned differentiable surrogate. It is the "optimize only the H-step reward sum, no critic" lineage (§2, §3.3). |
| [6], [13], [14] | ControlVAE, MoConVQ, SuperDyno | Learned-dynamics motion control (§2). |
| [15], [16], [17], [18] | Dreamer, DreamerV3, TD-MPC, DayDreamer | Model-based policy optimization (§2). |
| [20] | Song, Kim, Scaramuzza (CoRL 2024) | A simplified differentiable surrogate plus a high-fidelity forward simulator (§2). |
| [21], [22] | RWM, RWM-U (Li, Krause, Hutter) | Learned neural simulators with PPO in imagination (§2). |
| [23], [24] | Dyna, MBPO | Rollouts seeded from replay-buffer states (§3.1). |
| [25] | GAE | Critic targets (§3.3, Eq. 3). |
| [3] | BeyondMimic | Reward, observations, actions, domain randomization, and the PPO-Sim hyperparameters (§3.3, §4.1, App. A). |

**Not cited:**
- AHAC (Georgiev et al., ICML 2024).
- Schwarke et al. (arXiv 2404.02887; v1 "Learning Quadrupedal Locomotion via Differentiable Simulation", v2 "Learning Deployable Locomotion Control via Differentiable Simulation", CoRL 2025).
- DiffSim2Real (arXiv 2411.02189, CoRL 2024 workshop).
- TD-MPC2. Only TD-MPC is cited.
- No humanoid-tracking model-based RL work beyond [6], [12], [13], [14].

These are summarized below for gap-filling only.

---

## 2. What SuperMimic already pins down (these should not be re-invented)

| Item | Value | Where |
|---|---|---|
| World-model window k | 8 autoregressive steps on logged actions, seeded at the logged s₀ | §3.2, Table A.5 |
| World-model loss | Σⱼ ‖ŝⱼ−sⱼ‖² + (r̂ⱼ−rⱼ)² + ‖ô̄ⱼ−ōⱼ‖² + BCE(d̂ⱼ,dⱼ). An analytic head has no term. | Eq. 1 |
| World-model network and optimizer | f_ω(ŝ,a,ξ): MLP 2048×3. Adam, LR 2e-4 on 1 GPU (√2·2·2e-4 on 2 GPUs). 10 batches × 2048 windows per update. Global-norm clip 1.0. | Tables A.4, A.5 |
| Imagination | H = 32. The model is frozen during the policy step. Sensor noise η is re-added each step. The policy is deterministic. | §3.3, Table A.5 |
| Policy optimizer | Adam β = (0.9, 0.95), LR = α_ω. 4096 imagined rollouts per update. Clip global norm 1.0, then element-wise 0.1. γ = 0.99. | Table A.5 |
| Survival weight | c₀ = 1, c_{t+1} = c_t(1−d_{t+1}), with d̂ ∈ [0,1] predicted | §3.1 |
| Critic | Eq. 3 GAE(λ = 0.95) on imagined transitions only. Targets are computed once and held fixed. No EMA target network. Adam, LR = α_ω, clip 1.0, 5 epochs × 4 minibatches. Input (ŝ, g, ξ), MLP (256, 128). | §3.3, Tables A.4, A.5 |
| Data collection | 4096 envs per GPU. ε ~ N(0, 0.1²) per dimension; the buffer stores the noisy action. Buffer of 1000 steps per env, FIFO. | §3.1, Table A.5 |
| Schedule | Update-to-data ratio 0.4. One policy update per world-model update. Cosine LR to 0.03× over 250k iterations, then constant. 320k iterations. | Table A.5 |
| Policy network | Last 10 observations (Table A.1). Motion encoder MLP(2048,1024,512,256) over K = 10 future reference frames gives z ∈ ℝ³², tanh-bounded. Decoder MLP(2048,1024,512,256). | App. A, Table A.4 |
| Action, reward, termination | q_sp = α·a + q₀ with BeyondMimic gains. Table A.2 reward, every term ×Δt = 0.02. Terminate if \|e_{p,z}\| > 0.25 or tilt > 0.8 (gravity-z difference in pelvis frames). No end-effector termination. | App. A, Tables A.2 and A.6 caption |
| Domain randomization ξ | Friction U[0.3, 1.2]; joint offsets U[±0.01]; torso CoM offset. No velocity pushes. | Table A.3 |
| Simulator resets | 1 s bins. 50% uniform, 40% softmax(−25·G_b/G_max), 10% past softmax weight. | App. D.1 |
| Precision | TF32 | Fig. 5 caption |

**Lineage.**
- k = 8, H = 32 and noise σ = 0.1 match SuperTrack Table 1 exactly.
- β₂ = 0.95 is SHAC's (0.7, 0.95) with β₁ raised to 0.9.
- The 8-step forecast and the 1000-step buffer match RWM Tables S10 and S11.
- Return-based resets replace BeyondMimic's failure-based sampler (App. D.1 vs `mjlab/tasks/tracking/mdp/commands.py:249-290`).

---

## 3. Method by method

### 3.1 SHAC — Xu et al., ICLR 2022 (cited [19])

**Key equations.**
- Policy loss (Eq. 5): L_θ = −(1/Nh) Σᵢ [Σ_{t=t₀}^{t₀+h−1} γ^{t−t₀} R(sᵢₜ, aᵢₜ) + γʰ V_φ′(sᵢ_{t₀+h})]. Gradients come from BPTT through the simulator (App. A.1, Eqs. 9–12) with reparameterized actions.
- Critic target (Eq. 7) is TD(λ): Ṽ(s_t) = (1−λ) Σ_{k=1}^{h−t−1} λ^{k−1} G_t^k + λ^{h−t−1} G_t^{h−t}, held constant during the critic fit (Eq. 6).
- Target critic: V_φ′ ← αV_φ′ + (1−α)V_φ (Alg. 1). The target critic is used both in the policy loss and in Eq. 7.

**Hyperparameters.** From the general setting (Table 5) and Table 4:
- h = 32, N = 64 envs.
- Actor LR 2e-3, critic LR 5e-4, linear decay to 1e-5 (`DiffRL/algorithms/shac.py:434-441`).
- γ = 0.99, λ = 0.95.
- Target α = 0.995 (0.2 for the easy tasks).
- 16 critic iterations × 4 minibatches.
- Adam β = (0.7, 0.95) for both actor and critic (`shac.py:128-129`).
- Humanoid actor [256, 128], critic [128, 128] (Table 1).
- ELU with LayerNorm on every hidden layer (App. A.2.2; `models/actor.py:32,65`).
- Running observation normalization (§3.3; `shac.py:185-212`). Return normalization is off for humanoid (`ret_rms: False` in `cfg/shac/humanoid.yaml`).
- Stochastic policy with initial log-std −1.0 (`actor.py:71-73`). A deterministic policy also works and is sometimes better (App. A.4.4).

**Contact and exploding gradients.**
- Smoothed contact model (§3.1).
- Short windows, with gradients cut between windows (§3.3).
- The terminal critic smooths the discontinuities caused by early termination (§3.3, Fig. 2).
- Actor and critic gradient-norm clip 1.0 (`humanoid.yaml: grad_norm 1.0, truncate_grads True`; `shac.py:415-417`, `:473-474`).
- The run aborts if the gradient norm is NaN or above 1e6 (`shac.py:421`).
- The critic gradient has NaNs zeroed (`shac.py:466-468`).

**Termination and time-out handling** (`shac.py:228-239`):
- Early termination: the bootstrap value is 0.
- Time-out: the target critic is evaluated on `obs_before_reset`.
- The discount restarts after a reset (`:257`). Every env is cut at the end of the window (`:250`).
- Actions are tanh-squashed (`:198`).

**What SuperMimic changes** (§3.3, Table A.5):
- Rollouts through a frozen learned model, seeded from replay states (Dyna). SHAC instead continues live simulator states.
- A soft survival weight instead of hard reset handling.
- No target critic.
- β = (0.9, 0.95), LR 2e-4.
- 5 × 4 critic passes.
- A deterministic policy.
- An additional element-wise 0.1 gradient clip.

**Code:** https://github.com/NVlabs/DiffRL

### 3.2 SuperTrack — Fussell, Bergamin, Holden, TOG 2021 (cited [12])

**Model** (Alg. 1, §3.4, Eqs. 8–12):
- The world model predicts root-local rigid-body linear and angular accelerations, rotates them to world, and integrates semi-implicitly: v′ = v + dt·a; x′ = x + dt·v′; rotation q′ = exp(dt/2·ω′) ⊗ q.
- Loss: L1 on position, velocity, rotation (quaternion log difference) and angular velocity over an N_W = 8 window. Weights give "roughly equal contribution" at the start of training.

**Inputs** (§3.3, Eqs. 1–7):
- Body positions and velocities in the root frame, two-axis (6D) rotations, body heights, and the up vector in the root frame. Heights and up vector are there "to infer contact information".
- Every input is normalized by mean/std "computed from a database of kinematic data offline before training".

**Policy** (Alg. 2, §3.5):
- Rolls N_Π = 32 steps through the world model from a buffer start state.
- Gaussian noise σ = 0.1 on the PD offsets, offset scale α = 120.
- Local-frame L1 losses plus small L2/L1 regularizers on the offsets.

**Table 1:**
- World model: 5×1024 ELU, batch 2048, LR 1e-3.
- Policy: 5×1024 ELU, batch 1024, LR 1e-4.
- RAdam. One world-model step and one policy step per iteration. Buffer of about 150k samples (§3.6).

**Ablations:**
- Policy window 64 is unstable; 32/8 is "consistently stable" (§6.1).
- Predicting accelerations beats predicting velocities (§6.3).
- No critic, so look-ahead is limited to N_Π. Gradient clipping "help[s] somewhat" (§7).

**Code:** no official code; the author's publications page lists none.

### 3.3 ControlVAE — Yao et al., TOG 2022 (cited [6]); MoConVQ [13]

**World model** (§3.4.2):
- "formulated and trained in the same way as SuperTrack": 4×512 ELU.
- Predicts the *change* of body velocity and angular velocity in the root frame, then forward Euler.

**State** (§3.4.1, Eq. 15): root-local positions, 6D rotations, velocities, heights, and the up axis.

**Training process** (§3.3.3):
- World-model windows T_w = 8, N_w = 512, 8 updates per epoch.
- Policy synthetic rollouts T = 24 from start states "randomly selected from the simulation buffer".
- Buffer 5×10⁴.

**Optimizer** (§5.2): RAdam (0.9, 0.999), LR policy 1e-5 / world model 2e-3, gradients clipped to [−1, 1].

**Code:** https://github.com/heyuanYao-pku/Control-VAE and https://github.com/heyuanYao-pku/MoConVQ (I checked that the repos exist; I did not read them).

### 3.4 Dreamer v1 — Hafner et al., ICLR 2020 (cited [15]): the analytic-gradient precedent

**Objective.**
- The actor maximizes Σ V_λ(s_τ) by backpropagating "analytic gradients through the learned dynamics" (Eqs. 5–8).
- V_λ (Eq. 6) is the same λ-return as SuperMimic's Eq. 3 and SHAC's Eq. 7.
- With early termination, the model predicts a discount and "weigh[s] the time steps … by the cumulative product of the predicted discount factors" (§3). This is SuperMimic's survival weight c_t.

**Hyperparameters** (App. A):
- H = 15, γ = 0.99, λ = 0.95.
- Adam LRs: model 6e-4, value 8e-5, actor 8e-5.
- Gradient norms above 100 are scaled down.
- No target value network, no entropy bonus.
- ELU 3×300. Actor: tanh-Gaussian with the mean scaled ×5.

**Code:** https://github.com/danijar/dreamer

### 3.5 DreamerV3 — Hafner et al., Nature 2025 / arXiv 2301.04104v2 (cited [16])

**Caveat:** the actor uses **REINFORCE for both discrete and continuous actions** (Eq. 6). It is not a backprop-through-model method.

**Robustness tricks** (Methods, Table 4):
- symlog on vector inputs and targets (Eqs. 8–9).
- twohot critic and reward.
- Return normalization S = EMA(Per95 − Per5, 0.99), dividing by max(1, S) (Eq. 7).
- 1% unimix on categoricals.
- Critic regularized toward its EMA (decay 0.98).
- Zero-initialized output weights for reward and critic.
- AGC(0.3) with LaProp (ε = 1e-20).
- RMSNorm + SiLU.
- LR 4e-5, H = 15, γ = 0.997, λ = 0.95.

**What transfers to SuperMimic:**
- Zero-init of the critic output.
- symlog or standardization for heavy-tailed inputs such as joint velocities.
- Percentile return normalization, but only if reward scales diverge.

**What does not transfer:**
- unimix and categorical latents (the model is a deterministic MLP).
- AGC as a replacement for the specified global 1.0 / element-wise 0.1 clip.

**Code:** https://github.com/danijar/dreamerv3

### 3.6 RWM and RWM-U — Li, Krause, Hutter (cited [21], [22])

**RWM training.**
- Autoregressive multi-step training (Eqs. 1–2): L = (1/N) Σₖ αᵏ [L_o + L_c]. Predicted observations are fed back. Privileged contact heads serve as an auxiliary loss.
- GRU 256×2 with MLP heads of 128 units (Table S7). The model predicts a Gaussian mean and std and uses reparameterization (§3.2).

**Table S10:**
- History M = 32, forecast N = 8, decay α = 1.0.
- LR 1e-4, weight decay 1e-5, batch 1024.
- The ablation (Fig. S8) shows N = 1 (teacher forcing) is poor and N = 8 is a good trade-off.

**MBPO-PPO** (Table S11): 4096 imagination envs, 100 imagination steps, buffer 1000.

**Termination:** the model predicts terminations and imagination treats them as episode ends (App. A.4.3).

**Negative result (important):**
- RWM §4.4 trained SHAC through RWM on velocity tracking (ANYmal D, G1). It "fails to converge … not well-suited for discontinuous dynamics". Neither SHAC nor Dreamer gave a deployable policy.
- Confounds: a stochastic GRU model fit on single-environment data and a velocity-tracking reward. SuperMimic uses a deterministic MLP, 4096 envs, and a dense tracking reward.

**RWM-U:** an ensemble of 5 observation heads; the uncertainty penalty is r̃ = r − λu (Eq. 5).

**Code:** https://github.com/leggedrobotics/robotic_world_model. It is an Isaac Lab extension covering RWM and RWM-U. Configs use `history_horizon 32`, `forecast_horizon 8`, `ensemble_size 5` (`anymal_d_flat_cfg.py:68-70`).

### 3.7 Song, Kim, Scaramuzza — CoRL 2024 (cited [20])

**Method.**
- The forward pass comes from IsaacGym. Gradients come from a single-rigid-body surrogate through **state alignment** (§2.4): x̂ = x_nondiff + α(x_diff − sg(x_diff)). The forward value is exact, and the backward pass scales each step's Jacobian by α.
- α = 0.9, LR 1e-3, MLP [256, 256] (Table 4). Horizon segments N = 24 (§2.5). No critic.
- Without alignment, nothing is learned (App. 5).
- A terminal penalty has zero gradient for a differentiable objective (App. 4). This is an argument for SuperMimic's critic and survival weight.

**Code:** none found in one search (not definitive).

### 3.8 MBPO / Dyna — Janner et al., NeurIPS 2019 (cited [24])

- Algorithm 2: sample s_t **uniformly from D_env**, then run a k-step branched model rollout.
- k schedule: thresholded-linear 1→15 or 1→25 over epochs 20→100 (App. C, Table 1).
- Ensemble B = 7.
- In code, the model predicts **Δobs** (`mbpo/models/constructor.py:20-28`), and the input scaler is refit on the training inputs (`mbpo/models/bnn.py:335`).

**Code:** https://github.com/jannerm/mbpo

### 3.9 TD-MPC2 — Hansen et al., ICLR 2024 (not cited; TD-MPC [17] is)

- Every component is an MLP with LayerNorm + Mish. SimNorm on the latent is "essential to training stability" (§3.1).
- Model horizon H = 3, temporal weight λ = 0.5. Policy loss normalized by moving 5%/95% percentiles of Q.
- Adam 3e-4, gradient clip 20 (Table 8, Eqs. 3–4).

**Code:** https://github.com/nicklashansen/tdmpc2

### 3.10 GAE (cited [25])

SuperMimic Eq. 3, R̂_t = r̂_{t+1} + γ(1−d̂_{t+1})[(1−λ)V_{t+1} + λR̂_{t+1}] with R̂_H = V_H, is the same recursion as DreamerV3 Eq. 5 and SHAC Eq. 7, with the continuation set to 1 − d̂.

### 3.11 BeyondMimic (arXiv 2508.08241, cited [3]) and the mjlab re-implementation

mjlab's version is pinned to `HybridRobotics/whole_body_tracking@f8e20c8` (`mjlab/tasks/tracking/tracking_env_cfg.py:6-9`).

| Element | mjlab file:line | SuperMimic change |
|---|---|---|
| Reward: anchor pos/ori 0.5 (σ 0.3/0.4); body pos/ori/lin vel/ang vel 1.0 (σ 0.3/0.4/1.0/3.14); action rate −0.1; joint limit −10; self-collision −10 | `tracking_env_cfg.py:209-251`; formulas `tracking/mdp/rewards.py:26-113` | Joint limit becomes −2; self-collision removed (App. A, Table A.2) |
| Reference body positions re-anchored to the robot (xy + yaw) | `tracking/mdp/commands.py:396-405` | Must be reimplemented in torch for an analytic reward |
| Termination: anchor z > 0.25; gravity-z tilt > 0.8; end-effector z > 0.25 | `tracking_env_cfg.py:257-279`, `tracking/mdp/terminations.py:162-188` | End-effector term dropped (App. A). The BeyondMimic paper text says ‖e_R,anchor‖ > 0.8 rad, but mjlab and SuperMimic use the gravity-z difference. |
| Anchor body | `tracking/config/g1/env_cfgs.py:41` (`torso_link`) | SuperMimic says root = pelvis for termination. Whether the Table A.2 "Root" terms use pelvis or torso is UNSPECIFIED. |
| Domain randomization: torso CoM (±0.025/0.05/0.05), encoder bias ±0.01, foot friction abs U(0.3, 1.2), pushes every 1–3 s | `tracking_env_cfg.py:165-203`; feet and torso in `env_cfgs.py:59-62` | Identical to Table A.3 except pushes, which are removed |
| Actor observations: command (reference q, q̇), anchor pos/ori, base lin vel, ang vel, joint pos/vel, last action | `tracking_env_cfg.py:47-80`; command `commands.py:126-127` | Table A.1 has no base lin vel or anchor terms; it uses projected gravity (±0.05); 10-frame history; K = 10 future goal frames |
| Action scale α = 0.25·τ_max/k_p; k_p from ω = 10 Hz, ζ = 2 | `asset_zoo/robots/unitree_g1/g1_constants.py:112-116,278-286`, `env_cfgs.py:37` | Same: "BeyondMimic gains" (App. A) |
| Adaptive sampling: failure EMA α = 0.001, uniform 0.1/S, kernel λ = 0.8 | `commands.py:249-290`, defaults `:590-594` | Replaced by return-based resets (App. D.1). mjlab's default `adaptive_kernel_size = 1`, whereas the BeyondMimic paper uses K = 3 (u ∈ {0,1,2}). |
| PPO: actor/critic [512, 256, 128] ELU; LR 1e-3 adaptive (KL 0.01); clip 0.2; entropy 0.005; 5 epochs × 4 minibatches; 24 steps per env; γ 0.99; λ 0.95; max grad norm 1.0 | `tracking/config/g1/rl_cfg.py:13-45` (= BeyondMimic Table S5) | **This is PPO-Sim** (§4.1), but with 8192 envs per GPU and SuperMimic's actor and critic sizes |
| Rewards ×dt | `managers/reward_manager.py:34-41` | Same (Table A.6 caption) |

### 3.12 Not cited, but useful

**AHAC** (ICML 2024):
- Constrained objective (Eq. 9) and Lagrangian (Eq. 10): truncate the horizon where the contact-force Jacobian ‖∇f‖ > C = 500, with contact forces normalized by q̂ = max(q, 1) (§4.2).
- Double critic without a target network, trained until convergence (at most 64 iterations) (App. C).
- Actor LR 2e-3; horizon LR 1e-5 (2e-4 for Hopper); gradient norm 1.0; initial H = 32 (Tables 3–4).
- The optimal horizon tracks the gait period: H ≈ 28 for Ant (Fig. 6).
- It needs simulator contact Jacobians, so it does not apply directly to an MLP world model. The gait-period idea is the transferable part.
- Code: https://github.com/imgeorgiev/DiffRL (the README says it is the AHAC code).

**Schwarke et al.** (CoRL 2025; v1 2404.02887):
- SHAC in Warp with sigmoid-smoothed hard contact, κ = 300 (§3.3, Eq. 5).
- On ANYmal, the gradient norm "increases exponentially and prohibits successful learning" when backpropagating more than 16 steps, so they use H = 12 (§4, Fig. 5).
- Table 9: actor LR 5e-3→1e-5, critic 2e-3→1e-5; actor gradient norm 1.0, critic 10.0; 128 envs; 16 × 4 critic passes.
- DiffSim2Real (2411.02189) is the workshop version.
- No code found in one search.

**SuperDyno** (cited [14]): the full text is behind OpenReview's bot check (forum `FPgKt7bA8w`). The abstract says it combines supervised learning and RL through neural dynamics and scales to tens of thousands of clips. Nothing else could be verified.

**Open-DiffLoco** (arXiv 2608.02069): a SHAC implementation in MJX that it describes as open-source, deployed on a Go2. I read the abstract only; I did not verify a code link.

---

## 4. Recommended defaults for the UNSPECIFIED details

| # | Detail | Status | Default | Source / justification |
|---|---|---|---|---|
| 1 | Contents of state s_t | UNSPECIFIED. "state" only; the Fig. 3 error "counts the state only, normalized by a predefined nominal error". | Generalized coordinates: root height, root orientation, 29 joint q, root linear and angular velocity, 29 q̇ (mjlab qpos/qvel). Network inputs exclude world x, y and yaw (heading-invariant) and include root height and up vector. World x, y and yaw are kept only for integration and reward. | (q, q̇) plus ξ is Markov for MuJoCo position actuators (ours). Local frame plus heights plus up vector follow SuperTrack §3.3 Eqs. 1–7 and ControlVAE Eq. 15. Alternative: SuperTrack-style maximal coordinates (§3.2). |
| 2 | Delta vs absolute prediction | UNSPECIFIED | Predict Δq̇ (acceleration·dt) in the yaw-removed root frame, then integrate semi-implicitly: q̇′ = q̇ + Δq̇, q′ = q ⊕ dt·q̇′ (quaternion exp for the root). Zero-init the output layer so the model starts as constant velocity. | SuperTrack §3.4 Eqs. 8–12 and §6.3 (acceleration beats velocity); ControlVAE §3.4.2; MBPO `constructor.py:20-28`. Zero-init: ours, by analogy with DreamerV3 Methods. |
| 3 | Input normalization | UNSPECIFIED | World model: per-dimension mean/std of (state features, a, ξ) computed from the buffer after prefill, then frozen or very slow EMA, with a std floor. Policy: running normalizer, the same as the PPO baseline (mjlab `rl_cfg.py:16`). Do not use mocap statistics: 2 clips give degenerate std. | SuperTrack §3.3 (fixed statistics); MBPO `bnn.py:335`; SHAC §3.3 and `shac.py:185-212` (running statistics). |
| 4 | Target normalization and loss weights | UNSPECIFIED (Eq. 1 is an unweighted MSE) | Apply the Eq. 1 MSE to targets normalized per dimension by their buffer std. Weight all 8 steps equally. | SuperTrack §3.4 ("roughly equal contribution"); RWM Table S10 (forecast decay α = 1.0); SuperMimic Fig. 3 caption (nominal-error normalization). |
| 5 | Activations and layer norm | UNSPECIFIED for all networks | World model: Linear→LayerNorm→SiLU (or Mish) ×3. Actor and critic: ELU, matching the PPO baseline for a fair comparison. Zero-init the critic output layer. | TD-MPC2 Table 8 (LN + Mish); DreamerV3 Table 4 (RMSNorm + SiLU); SHAC App. A.2.2 (ELU + LN); BeyondMimic Table S5 and `rl_cfg.py:15,25`. |
| 6 | Which heads are analytic | UNSPECIFIED ("either learned or computed analytically", §3.2) | Observation head analytic: every Table A.1 term is closed-form in (q, q̇, root quaternion, ξ offsets, a_{t−1}). Reward head analytic on ŝ_{t+1}: needs a torch FK for the 14 tracked bodies (`env_cfgs.py:42-57`), body velocities via J(q)q̇ (autograd JVP), and the anchor alignment from `commands.py:396-405`. Termination head learned with BCE (Eq. 1). Fallback if FK is too costly: a learned reward head, at the cost of exploitable reward errors. | ours. The exploitation risk is the motivation for RWM-U's uncertainty penalty (Eq. 5). |
| 7 | Terminal transitions in the buffer | UNSPECIFIED | Store the pre-reset state as the terminal next state with d = 1. Windows never cross a reset. | SHAC `shac.py:228-239` (`obs_before_reset`). The repo env auto-resets inside `step()` (repo `CLAUDE.md`, pitfall 2), so the buffer writer has to capture the state before the reset. |
| 8 | Clip end / time-out in imagination | UNSPECIFIED | Treat clip end (fewer than K = 10 future frames) as a time-out, not a termination: mask later rewards and bootstrap V there, with no survival decay. | SHAC `shac.py:232-239` (early termination → 0, time-out → critic). This matters because the walking clips are about 200 steps and H + K = 42. |
| 9 | Observation history at the imagination start | UNSPECIFIED | Store the noise-free ō history. At the start, load the 9 logged previous ō and re-add fresh η. Starts fewer than 9 steps after a reset: pad by repeating the first frame, or exclude them. | ours. The need comes from Dyna-style seeding (§3.1; MBPO Alg. 2 line 7). SHAC continues live simulator states (Alg. 1), so it never meets this. |
| 10 | Start-state distribution | Partly specified (a replay state with its ξ) | Uniform over valid (env, t) in the buffer, excluding terminal states. | MBPO Alg. 2 line 7; ControlVAE §3.3.3. Simulator resets are already return-weighted (App. D.1). |
| 11 | Sampling of world-model windows | UNSPECIFIED | Uniform over 8-step windows that lie inside one episode. | SuperTrack Alg. 1; ControlVAE §3.3.3; RWM §3.2. |
| 12 | Meaning of update-to-data 0.4 | UNSPECIFIED (ambiguous) | 2.5 simulator steps per env per iteration (alternate 2 and 3), i.e. about 10k transitions per iteration at 4096 envs. The 1000-step buffer then spans about the last 400 iterations. Make it a config value. | ours. Check: 320k iterations (Table A.5) in about 9.8 h (§4.1, Fig. A.1) is about 110 ms per iteration. Readings with 25 or more simulator steps per env per iteration (≥10⁵ simulator steps per iteration) would need ≥0.9 M simulator steps/s for the simulation alone, which you can benchmark against MuJoCo Warp on your GPUs (Fig. 5(a)). |
| 13 | Prefill and warm-up | UNSPECIFIED (Alg. A.1 updates everything from iteration 1) | Prefill 64 steps per env with the initial policy plus ε. Then world-model-only updates for about 1–2k iterations, until the 8-step error flattens. Then start the policy and critic. | Dreamer v1 App. A (5 random seed episodes); MBPO/SAC seed steps (AHAC Table 3). The numbers are ours. |
| 14 | Horizon warm-up | SPECIFIED: H = 32 fixed | No ramp by default. Log the pre-clip gradient norm (as `shac.py:414-418` does). Only if it explodes, ramp H 8→32 over a few thousand updates using MBPO's thresholded-linear schedule. | MBPO App. C Table 1; Schwarke §4 Fig. 5. Do not stay at H ≤ 16: H = 8 never reaches the target and H ≤ 16 needs the critic (SuperMimic §4.3, Fig. 4c). |
| 15 | Exploding-gradient fallback | Clipping SPECIFIED | Optional: per-step Jacobian decay ŝ ← sg(ŝ) + 0.9·(ŝ − sg(ŝ)), which leaves the forward pass unchanged. Skip, rather than abort, an update with a NaN gradient. | Song §2.4 and Table 4 (α = 0.9); SHAC `shac.py:421`. |
| 16 | Value or return normalization | UNSPECIFIED | None. Rewards are Δt-scaled, so tracking terms are ≤ 0.02·(4·1 + 2·0.5) = 0.1 per step and V ≤ 10. | Table A.6 caption; SHAC `humanoid.yaml` (`ret_rms: False`). Percentile normalization (DreamerV3 Eq. 7) only if multi-task reward scales diverge. |
| 17 | Adam betas for the world model and critic; weight decay | UNSPECIFIED | (0.9, 0.999), no weight decay. | ours (PyTorch default). SHAC shares (0.7, 0.95) across actor and critic (`shac.py:128-129`); RWM uses weight decay 1e-5 (Table S10). |
| 18 | Which optimizers the LR schedule covers | UNSPECIFIED | All three (world model, actor, critic). They share α_ω. | Table A.5 ("same as α_ω"); SHAC decays both actor and critic (`shac.py:434-441`). |
| 19 | Action output range | UNSPECIFIED | Linear output, no tanh, as in BeyondMimic ("intentionally not clipped", Methods, "Action"). If the model gets exploited, clamp the world-model action input to the range seen in the buffer. | SHAC uses tanh (`shac.py:198`) because its tasks are torque-bounded. |
| 20 | Encoding of ξ | UNSPECIFIED | Concatenate [μ, Δq₀ (29), Δcom (3)] = 33 dimensions, each scaled to [−1, 1] by its randomization range. | ours. Matches the mjlab randomization set (`tracking_env_cfg.py:172-202`). |
| 21 | Initial-state noise at simulator resets | UNSPECIFIED | Keep BeyondMimic's pose and velocity noise (`tracking_env_cfg.py:144-153`). It is an initial-state perturbation, not a mid-episode discontinuity. | SuperMimic App. A removes only pushes. |
| 22 | Content of a future goal frame | K = 10 SPECIFIED; content UNSPECIFIED | Per frame: reference joint q and q̇ (`commands.py:126-127`) plus the reference anchor pose relative to the robot (`tracking_env_cfg.py:51-60`). Stride of 1 control step. | ours |

---

## 5. Implications for the side-by-side plan

**Walking as clip tracking.** This is SuperMimic's own setting: a dense BeyondMimic reward with the loop of §3. The datasets are tiny, though: 2 clips of about 4 s for walking and 2 × 6.93 s for lifting. SuperMimic's smallest experiments use 1 h (Fig. 3b–c) and LAFAN1, 2.45 h (Fig. 4). Its advantage on a single clip is not shown.

**Walking as velocity commands** (the repo's `Mjlab-G1-Locomotion`). Expect SHAC-through-model to be fragile:
- RWM §4.4 failed at exactly this.
- SuperMimic trained a velocity *encoder* only, with the decoder frozen (App. B, G). It never trained velocity tracking from scratch through the model.

**Lifting.**
- SuperMimic has "No terrain or object interaction" (§6) and removes box clips (App. E).
- The faithful comparison is tracking the robot-only squat-grasp-lift clip.
- Adding the toaster puts object pose and velocity plus friction-only grasp contacts into the world model. No cited method covers that.

**PPO baseline.** `rl_cfg.py:28-45` equals BeyondMimic Table S5, which equals SuperMimic's PPO-Sim (§4.1). To mirror the paper, use 8192 envs per GPU and SuperMimic's network sizes.

**Metrics.** Use SuperMimic's normalized return (§4) and failures under the training termination (App. F.1), plus wall-clock time and GPU-hours.

---

## 6. Code links

I checked that each repo exists (HTTP 200 on 2026-10-22). NVlabs/DiffRL, robotic_world_model and jannerm/mbpo were also read in part.

| Method | Repository |
|---|---|
| SHAC | https://github.com/NVlabs/DiffRL |
| AHAC | https://github.com/imgeorgiev/DiffRL |
| RWM / RWM-U | https://github.com/leggedrobotics/robotic_world_model |
| ControlVAE | https://github.com/heyuanYao-pku/Control-VAE |
| MoConVQ | https://github.com/heyuanYao-pku/MoConVQ |
| Dreamer | https://github.com/danijar/dreamer |
| DreamerV3 | https://github.com/danijar/dreamerv3 |
| TD-MPC2 | https://github.com/nicklashansen/tdmpc2 |
| MBPO | https://github.com/jannerm/mbpo |
| BeyondMimic | https://github.com/HybridRobotics/whole_body_tracking |
| mjlab | https://github.com/mujocolab/mjlab |

No public code found for SuperTrack, Song et al., Schwarke et al., SuperDyno or SuperMimic.

**Pages used:**
- [mlanthology SuperDyno entry](https://mlanthology.org/iclrw/2025/lei2025iclrw-scalable)
- [ICLR 2025 SuperDyno listing](https://iclr.cc/virtual/2025/37555)
- [DiffSim2Real arXiv](https://arxiv.org/abs/2411.02189v1)
- [Schwarke PMLR](https://proceedings.mlr.press/v305/schwarke25a.html)
- [SuperTrack page](https://theorangeduck.com/page/supertrack-motion-tracking-physically-simulated-characters-using-supervised-learning)
- [Open-DiffLoco arXiv](https://arxiv.org/pdf/2608.02069)
- [Song et al. arXiv](https://arxiv.org/html/2403.14864v2)