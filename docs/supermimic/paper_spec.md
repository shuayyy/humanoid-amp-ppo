# SuperMimic: implementation spec (arXiv:2610.28480v1, CoRL 2026)

## 0. Sources, conventions and caveats

- **Primary source.** The arXiv PDF (19 pages): text extracted with pypdf, saved at `<scratch>/pdfout/paper.txt`. I also rendered pages 5 and 14–16 to images and checked Eqs. 1–3 and Tables A.2–A.6 visually, because extracted text drops glyphs (for example, "|e_p,z|>0.25 **more** tilt>0.8" is really "**or**").
- **arXiv HTML.** Not available. arXiv returns "No HTML for '2610.28480v1'".
- **Code.** No public code. A web search for an OpenReview or rebuttal page found nothing.
- **Project page.** I used the prose of https://supermimic.github.io (FAQ, compute notes, failed-motion table). I did **not** use anything from the in-browser demo. Its LICENSE forbids downloading for retention and deriving its architecture or internals, so I deleted the demo files I had fetched.
- **Notation.**
  - `(§x)`, `(Eq. n)`, `(Table A.n)`, `(App. X)` cite the paper.
  - `MJ/...:Lx-y` cites mjlab v1.4.0 source under `<mjlab v1.4.0>/src/mjlab/`. mjlab calls its tracking task "a re-implementation of BeyondMimic", based on whole_body_tracking@f8e20c8 (`MJ/tasks/tracking/tracking_env_cfg.py:6-9`).
  - **UNSPECIFIED** means the paper does not state it. Every proposed default is marked **(ours)**.

---

## 1. Problem setup

### 1.1 POMDP and notation (§3.1)
- Goal-conditioned POMDP M with state s_t, goal g_t and action a_t.
  - s_t gives the observation o_t and the termination d_t ∈ {0,1}.
  - (s_t, a_t) gives r_{t+1} and s_{t+1}.
- **Policy.** Deterministic: a_t = π_θ(o_t, g_t). a_t "parameterizes a normalized joint setpoint". For motion tracking, g_t is read from the reference motion.
- **Observation noise.** o_t = ō_t + η_t, where ō_t is noise-free and η_t ~ ρ_η is known noise sampled outside the world model.
- **Collection noise.** During collection the simulator executes a_t + ε_t, with ε_t ~ ρ_ε = N(0, 0.1²) per action dimension (Table A.5). The buffer stores the executed noisy action. The noise "serves only the model fit and is optional" (§3.1).
- **Soft survival weight.** c_0 = 1 and c_{t+1} = c_t(1 − d_{t+1}). The objective is J(θ) = E[Σ_{t≥0} γ^t c_t r_{t+1}]. The simulator logs hard d, while the model predicts d̂_t ∈ [0,1] (§3.1).
- **Simulator and timing.**
  - MuJoCo Warp; 50 Hz control = 4 substeps of 5 ms (Fig. 5 caption).
  - mjlab tracking already matches: timestep 0.005 and decimation 4 (`MJ/tasks/tracking/tracking_env_cfg.py:306,311`).
  - 4096 simulator envs per GPU (Table A.5).

### 1.2 Observation o_t (App. A, Table A.1): 93 per frame, last 10 frames (930)

| Term | Dim | Noise η (uniform, per dim) |
|---|---|---|
| Joint positions q − q0 | 29 | U[−0.01, 0.01] rad |
| Joint velocities q̇ | 29 | U[−0.5, 0.5] rad/s |
| Base angular velocity (IMU) | 3 | U[−0.2, 0.2] rad/s |
| Projected gravity (base frame) | 3 | U[−0.05, 0.05] |
| Previous action a_{t−1} | 29 | none |

- **What is missing compared with mjlab's tracking actor.** The actor has no base linear velocity, no reference-anchor position or orientation in the base frame, and no current-frame joint command. mjlab's tracking actor has all of these (`MJ/tasks/tracking/tracking_env_cfg.py:47-80`).
- **Consequence.** The reference enters only through g_t, and the actor needs no state estimator.
- **Joint encoder offsets.** The calibration offset Δq0 in ξ biases the observed q. In mjlab, `encoder_bias` shifts both the observation (`MJ/entity/data.py:369-371`, `joint_pos_rel(biased=True)` at `MJ/tasks/tracking/tracking_env_cfg.py:71-75`) and the PD target (`MJ/envs/mdp/actions/actions.py:222-223`).
- **Per-frame ordering and history padding at reset.** UNSPECIFIED. (ours: Table A.1 order; pad the history with the first frame.)

### 1.3 Goal g_t (App. A "Goal", Table A.4)
- K = 10 **future** reference frames feed a motion encoder. BeyondMimic uses a single-frame target.
- **Per-frame quantities.** UNSPECIFIED.
- **Frame spacing, and whether the current frame is included.** UNSPECIFIED.
- **Robot-relative quantities.** UNSPECIFIED. If g_t contains any (for example, the reference anchor orientation expressed in the robot frame, like mjlab's `motion_anchor_ori_b`, `MJ/tasks/tracking/mdp/observations.py:31-41`), it must be recomputed from ŝ_t at every imagined step, and it is then differentiable with respect to ŝ_t.
- **Proposed default (ours).** Frames t+1 … t+10 at 50 Hz. Per frame: [q_ref (29), q̇_ref (29), reference-anchor orientation in the robot-anchor frame as 6D (6)] = 64, so 640 in total. This mirrors mjlab's no-state-estimation actor (`MJ/tasks/tracking/config/g1/env_cfgs.py:74-84`), which suits an actor that has no linear-velocity input.

### 1.4 Action (App. A "Action")
- q_sp = α·a_t + q0, about the default pose q0.
- "Following BeyondMimic, the low PD gains make this setpoint a proxy for joint torque."
- **Numeric α, gains and q0.** UNSPECIFIED in the paper. BeyondMimic values in mjlab:
  - α_j = 0.25·τ_max,j / K_p,j (`MJ/asset_zoo/robots/unitree_g1/g1_constants.py:278-286`).
  - K_p = I_armature·(2π·10 Hz)² and K_d = 2·2.0·I_armature·(2π·10 Hz) (`g1_constants.py:112-123`).
  - q0 = KNEES_BENT keyframe (`g1_constants.py:198-211`, used at `:270-275`).
- **Output squashing or clipping.** UNSPECIFIED.

### 1.5 State s_t
- **Contents and representation.** UNSPECIFIED. The paper says only that s_t "gives" o_t and d_t (§3.1).
- s_t is the input to f_ω and V_ψ.
- The "8-step prediction error counts the state only, is normalized by a predefined nominal error, and is measured online on policy rollouts" (Fig. 3 caption).
- Proposal in §9, items 1–3.

### 1.6 Episodes, resets and termination
- **Termination (App. A).** |e_{p,z}| > 0.25 m **or** e_tilt > 0.8.
  - e_tilt = |z-component of the gravity direction in the reference root frame − the same in the robot root (pelvis) frame|. For an upright reference this is about 78°.
  - **No end-effector termination**, unlike BeyondMimic.
- **mjlab differences.**
  - mjlab's `bad_anchor_pos_z_only` and `bad_anchor_ori` (`MJ/tasks/tracking/mdp/terminations.py:162-188`) have exactly this form, but on the **anchor torso_link** (`MJ/tasks/tracking/config/g1/env_cfgs.py:41`). The paper says pelvis.
  - The `ee_body_pos` term (`tracking_env_cfg.py:271-278`, `env_cfgs.py:64-69`) must be removed.
- **Episode length / time-out.** UNSPECIFIED. mjlab uses 10 s (`tracking_env_cfg.py:312`) and, when a clip ends, resamples the clip mid-episode instead of ending the episode (`MJ/tasks/tracking/mdp/commands.py:407-411`).
- **Reset-state perturbations.** UNSPECIFIED. mjlab adds pose noise, root-velocity noise and joint noise of ±0.1 rad (`tracking_env_cfg.py:144-153`, applied at `commands.py:333-365`).

### 1.7 Return-based reset sampling (App. D.1)
- The motion timeline is divided into **1 s bins**. Each bin b keeps a running estimate G_b of the discounted return from it, updated on the policy's own rollouts.
- Reset distribution:
  - P(b) = 0.5·Uniform(b) + 0.4·softmax_b(−25·G_b/G_max) + 0.1·(proportional to each bin's past softmax weight).
  - G_max is "the largest attainable discounted return".
  - The 10% term exists "so a hard motion keeps some practice after it is learned".
- **UNSPECIFIED.** The G_b update rule (EMA rate; whether it uses episode-start bins or every visited step), the definition and window of "past softmax weight", the closed form of G_max, and the start time within a bin.
- **mjlab differs.** Its sampler is failure-count based: EMA α = 0.001, 10% uniform mix, one bin per second (`MJ/tasks/tracking/mdp/commands.py:93,249-290,590-594`). It is not the paper's sampler.

### 1.8 Sim context ξ / domain randomization (App. A, Table A.3)
- **Per-episode** ξ, fed to f_ω and g_ω (and V_ψ):
  - friction μ ~ U[0.3, 1.2];
  - joint calibration offsets Δq0_j ~ U[−0.01, 0.01] rad;
  - torso CoM offset Δx ~ U[−0.025, 0.025], Δy ~ U[−0.05, 0.05], Δz ~ U[−0.05, 0.05] m.
- Dimension: 1 + 29 + 3 = **33 (our count)**.
- **Encoding and normalization.** UNSPECIFIED.
- **Removed.** BeyondMimic's root-velocity pushes, because "such pushes inject state discontinuities that the deterministic world model cannot capture".
- **mjlab mapping.**
  - Same distributions: `base_com`, `encoder_bias`, `foot_friction` with shared_random (`tracking_env_cfg.py:172-202`, foot geoms at `env_cfgs.py:59-61`, torso at `:62`).
  - But mjlab samples them with `mode="startup"`, i.e. once per env for the whole run (`MJ/managers/event_manager.py:108-112`). The paper says per episode, which needs `mode="reset"`.
  - `push_robot` (`tracking_env_cfg.py:166-171`) must be removed.

---

## 2. Reward

### 2.1 Motion-tracking reward (Table A.2): BeyondMimic minus self-collision, joint limit −10 → −2

| Term | Formula | Weight |
|---|---|---|
| Body position | exp(−(1/\|T\|)Σ_b ‖p_b^des − p_b‖² / 0.3²) | 1.0 |
| Body orientation | exp(−(1/\|T\|)Σ_b ‖log(R_b^des R_bᵀ)‖² / 0.4²) | 1.0 |
| Body linear velocity | exp(−(1/\|T\|)Σ_b ‖v_b^des − v_b‖² / 1.0²) | 1.0 |
| Body angular velocity | exp(−(1/\|T\|)Σ_b ‖ω_b^des − ω_b‖² / 3.14²) | 1.0 |
| Root position | exp(−‖p_root^des − p_root‖² / 0.3²) | 0.5 |
| Root orientation | exp(−‖log(R_root^des R_rootᵀ)‖² / 0.4²) | 0.5 |
| Action smoothness | ‖a_t − a_{t−1}‖² | −0.1 |
| Joint limit | Σ_j [max(0, q_j − q_j^max) + max(0, q_j^min − q_j)] | −2 |

- **Δt scaling.** Every term is multiplied by Δt = 0.02 s (Table A.6 caption: "here and in Table A.2"). mjlab's RewardManager does the same by default (`MJ/managers/reward_manager.py:34-44`).
  - Maximum per-step reward = (4×1.0 + 2×0.5)×0.02 = **0.1** (our arithmetic; penalties are ≥ 0 in magnitude).
- **Why self-collision was removed (App. A).** It guards a stochastic policy against pressing bodies together to suppress noise. The imagined policy is deterministic, so the exploit has no payoff.
- **Mapping to mjlab** (`MJ/tasks/tracking/tracking_env_cfg.py:209-251`, functions in `MJ/tasks/tracking/mdp/rewards.py:26-113`):
  - Same kernels and weights.
  - Body pos/ori use the reference **re-anchored** to the robot anchor's xy and yaw (`commands.py:377-405`). Body velocities and root terms are global.
  - Changes needed: joint_limit −10 → −2 (`:241-245`) and delete `self_collisions` (`:246-250`).
- **Tracked body set T.** UNSPECIFIED. mjlab/BeyondMimic G1 uses 14 bodies (`env_cfgs.py:42-57`).
- **"Root".** The paper names the pelvis as root only in the termination text. mjlab's root reward terms use torso_link.
- **Joint limits: hard or soft.** UNSPECIFIED. mjlab uses soft limits at 0.9× range (`MJ/envs/mdp/rewards.py:81-96`, `g1_constants.py:260`).
- **Reward/reference pairing.** mjlab computes r_{t+1} and d_{t+1} on the post-step state against the *pre-advance* reference frame. Order: termination, then reward, then reset, then command advance (`MJ/envs/manager_based_rl_env.py:436-456`; also noted at `MJ/tasks/tracking/scripts/evaluate.py:112-117`).

### 2.2 Analytic or learned, and where the actor gradient flows
- §3.2: "each head is either learned or computed analytically": r̂_{t+1} = r_ω(ŝ_t, a_t, g_t), and "an analytic head has no loss term" (Eq. 1).
- **Which heads are analytic for motion tracking is UNSPECIFIED.**
  - Reward head: UNSPECIFIED.
  - Observation head: UNSPECIFIED.
  - Termination head: §3.1 says "**the learned model predicts a termination probability** d̂_t ∈ [0,1]", which points to a learned BCE head.
- **Gradient path.** Either way, the actor gradient flows **through r̂_{t+1}** (Eq. 2), via the world model's Jacobians with respect to ŝ_t and a_t (§3.3). It also flows through ĉ_t (via d̂), through ō → future actions, and through V_ψ(ŝ_H).
- **Constraint for either choice.** The action-smoothness term and the observation's a_{t−1} both need the previous action. ŝ_t, or an auxiliary input carried next to it, must therefore include a_{t−1}.

### 2.3 Velocity interface reward (Table A.6; only for the App. B/G downstream study)
- Linear velocity: exp(−‖v_xy − v_xy^cmd‖² / 0.5²), weight 2.0.
- Yaw rate: exp(−(ω_z − ω_z^cmd)² / **0.5**), weight 2.0. The denominator is printed without a square.
- Action smoothness −0.3; joint limit −2; torso sideways lean sin²φ, −4; torso roll/pitch rate ω_roll² + ω_pitch², −0.05. All multiplied by Δt.
- Fall termination: h_root < 0.35 m or 1 + u_z > 0.7 (about 73°).

---

## 3. World model (§3.2, Table A.4, Table A.5)

- **Heads.** ŝ_{t+1} = f_ω(ŝ_t, a_t, ξ); r̂_{t+1} = r_ω(ŝ_t, a_t, g_t); ō̂_t = g_ω(ŝ_t, ξ); d̂_t = d_ω(ŝ_t, g_t).
- **Dynamics network.** MLP with hidden widths (2048, 2048, 2048).
  - Activation and normalization layers: UNSPECIFIED.
  - Absolute vs delta output: UNSPECIFIED.
  - Input/target normalization: UNSPECIFIED.
  - Architectures of the reward/obs/termination heads: UNSPECIFIED (Table A.4 lists only f_ω).
- **Deterministic, no ensemble.** "Since the simulator's randomness has known sources, the world model stays deterministic": it conditions on ξ and predicts the noise-free observation.
- **Precision.** TF32 in training (Fig. 5 caption).
- **Loss (Eq. 1).**
  - L_WM(ω) = E_B Σ_{j=1}^{k} [ ‖ŝ_j − s_j‖² + (r̂_j − r_j)² + ‖ō̂_j − ō_j‖² + BCE(d̂_j, d_j) ].
  - All weights are 1 as written. Analytic heads drop out.
  - k = 8 autoregressive windows, seeded at ŝ_0 = s_0 and rolled on **logged (noisy) actions**.
  - Per-dimension weighting or normalization of the MSE: UNSPECIFIED.
- **Autoregressive fit.** "the k-step autoregressive fit trains it to correct its own prediction errors", so the model "remains stable over indefinitely long rollouts" (§3.3). This implies gradients flow through the 8-step chain without detaching.
- **Optimization.**
  - Adam; learning rate 2×10⁻⁴ on 1 GPU and √2 × 2×10⁻⁴ on 2 GPUs.
  - Gradient clip: global norm 1.0.
  - "10 batches of 2048 windows per update" (whether that is 10 optimizer steps or one accumulated step is UNSPECIFIED; see §5).
  - Adam β/eps for the world model: UNSPECIFIED. β = (0.9, 0.95) is stated only for the policy.
- **Replay buffer.**
  - FIFO, 1000 steps per env, so 4.096 M transitions per GPU (our arithmetic). "new transitions replace the oldest" (Fig. 2).
  - Sampling distribution: UNSPECIFIED.
  - Handling of windows that cross a reset or termination: UNSPECIFIED.
  - Transitions from older policies stay valid "because the dynamics do not depend on the policy that collected them" (§3.1).
- **Data collection.** The current deterministic policy plus N(0, 0.1²) action noise (Table A.5).
- **Diagnostic.** 8-step prediction error of 0.007–0.011 across 1–264 h of data (Fig. 3c). The nominal-error normalizer is UNSPECIFIED.

---

## 4. Policy learning (§3.3, App. B, Table A.4, Table A.5)

### 4.1 Networks
- **Actor.** π(o, g) = dec_θ(enc_θ,motion(g), o_{t−9..t}).
  - Motion encoder: MLP (2048, 1024, 512, 256), K = 10 future frames → z.
  - z is 32-dimensional and tanh-bounded.
  - Shared decoder: MLP (2048, 1024, 512, 256), (z, last 10 frames of o) → a_t.
  - Deterministic.
  - Activations, input normalization and output layer: UNSPECIFIED.
- **Critic.** V_ψ(ŝ_t, g_t, ξ): MLP (256, 128), one critic per interface. **No EMA target network** (§3.3).

### 4.2 Imagination and objective
- **Start states.** Each rollout starts from a replay-buffer state s_0 ~ ρ_0 with its recorded ξ (§3.3; Alg. A.1 line 5). ρ_0 beyond "from B" is UNSPECIFIED.
- **Batch.** 4096 imagined rollouts per update, H = 32.
- **Each step.** ō̂_t = g_ω(ŝ_t, ξ); re-add sensor noise η_t ~ ρ_η (Table A.1); a_t = π_θ(ō̂_t + η_t, g_t); no action noise in imagination.
- **Objective (Eq. 2).** Ĵ(θ) = E[ Σ_{t=0}^{H−1} γ^t ĉ_t r̂_{t+1} + γ^H ĉ_H V_ψ(ŝ_H, g_H, ξ) ], with γ = 0.99.
  - Termination is soft, through ĉ_t; there is no hard truncation.
  - "SuperMimic replaces SHAC's hard episode termination with the survival weight ĉ_t."
- **Gradient.** First-order BPTT through H steps via Jacobians with respect to ŝ_t and a_t. ω is "held fixed (stop-gradient) only inside the policy's backward pass" (Alg. A.1 caption).
- **Actor optimization.** Adam with β = (0.9, 0.95); learning rate equal to α_ω.
  - **Clip global norm 1.0, then element-wise 0.1**, in that order.
- **Unspecified details.**
  - Loss reduction (sum or mean over H and batch): UNSPECIFIED. This matters because the clip thresholds are absolute.
  - Whether ψ is detached in the actor loss: UNSPECIFIED (the SHAC convention is yes).
  - Return or advantage normalization: none mentioned.
  - Obs-history initialization at the imagined start: UNSPECIFIED.
  - What happens when the reference clip ends inside H: UNSPECIFIED.

### 4.3 Critic (Eq. 3)
- **Loss.** L_V(ψ) = E Σ_{t=0}^{H−1} (V_t − R̂_t)², with
  - R̂_t = r̂_{t+1} + γ(1 − d̂_{t+1})[(1−λ)V_{t+1} + λR̂_{t+1}];
  - R̂_H = V_H; λ = 0.95;
  - V_t = V_ψ(ŝ_t, g_t, ξ).
- Targets are computed once per rollout, over imagined transitions only, and stay fixed while ψ is fit.
- 5 epochs, each over 4 shuffled minibatches. With 4096 × 32 = 131,072 samples, a minibatch is 32,768 (our arithmetic).
- Adam; learning rate equal to α_ω; clip global norm 1.0.
- **Order (Alg. A.1 lines 7–8).** The actor steps before the critic fit, so the actor uses the previous iteration's critic.

### 4.4 Horizon and critic findings (§4.3, Fig. 4b-c; LAFAN1)
- Without the critic, training fails at H ≤ 16. With the critic, H = 16 matches critic-free H = 32.
- H = 32 reaches 0.82 fastest.
- Going from H = 64 to H = 128 nearly triples the time to reach 0.82 and lowers the final return. H = 128 is the largest horizon that fits on an H100. H = 8 fails both with and without the critic.

### 4.5 Multi-interface (App. B, App. G; only for velocity)
- Objective: max_θ E_{i~p(I)} Ĵ_i(θ), with z = enc_θ,i(g) and a = dec_θ(z, o).
- On downstream rollouts the decoder weights are stop-gradiented. Only enc_θ,i updates from Ĵ_i (Alg. A.1 caption).
- Velocity encoder: MLP (256, 256) on the normalized (v_x, v_y, ω_z) plus "a feature of the last 10 frames of o". That feature is UNSPECIFIED.
- Latent-alignment MSE, weight 0.1, toward the stop-gradiented motion latent, on rollouts with reference commands.
- 100,000 iterations. Half of the 4096 envs and half of each imagined batch use the velocity interface.
- Commands:
  - Resampled at every reset and every 250 steps.
  - With p = 0.5, read from the reference: the centered 25-frame mean of the reference root velocity in the heading frame.
  - Otherwise uniform: v_x ∈ [−1.5, 2], v_y ∈ [−1.32, 1.32], ω_z ∈ [−3.35, 3.35], zeroed with p = 0.1.

---

## 5. Training schedule (Table A.5, §4, App. F)

- **Length.** 320,000 iterations.
- **Learning-rate schedule.** Cosine decay to 0.03× over 250,000 iterations, then constant. It applies to the world model, actor and critic, since all share α_ω. A warm-up is UNSPECIFIED.
- **Per iteration (Alg. A.1).**
  - Collect with noise and push to B.
  - Draw resets by return-based sampling.
  - One world-model update ("10 batches of 2048 windows").
  - **1 policy update per world-model update.**
  - One critic fit (5 × 4 minibatches).
- **Separate warm-up before the model is used.** None stated. "Both ω and θ update every iteration"; the FAQ says it is "No[t] … trained first, then frozen".
- **"Update-to-data ratio 0.4": definition UNSPECIFIED.** Two readings (both ours):
  - **(a)** 0.4 iterations per vectorized sim step, i.e. 2.5 sim steps per env per iteration = 10,240 transitions per iteration. Total ≈ 320k × 2.5 × 4096 ≈ 3.3×10⁹ transitions per GPU. The buffer turns over every 400 iterations.
  - **(b)** Updates per transition counted in windows: 20,480 windows / 0.4 = 51,200 transitions = 12.5 steps per env per iteration (≈1.6×10¹⁰ in total).
  - Reading (a) fits better with the ~10 h x-axis of Fig. A.1 and simulator throughput of order 10⁵ steps/s at 4k envs (Fig. 5a, read off a log axis). This rests on the unconfirmed assumption that a run of about 10 h covers the 320k iterations.
- **"10 batches of 2048 windows per update."** Either 10 sequential Adam steps or one step accumulated over 20,480 windows: UNSPECIFIED.
- **Multi-GPU.** Data parallel with averaged gradients; learning rate ×√2 on 2 GPUs.
- **Wall-clock claims.** All on the full BONES-SEED run (264.56 h, 129,514 motions; Fig. 3a, Table A.8, Fig. A.1):
  - 1 H100: surpasses the SONIC checkpoint's return (0.90) within **3 h**, with 25 failed motions. "From then on, every checkpoint is ahead of SONIC by at least half of the gap" (§4.1).
  - 2 H100: **0 failures after 9.8 h**.
  - Whether 3 h or 9.8 h marks the end of training: UNSPECIFIED.
  - Checkpoints are scored every 4000 iterations (Fig. A.1).
  - The 3 h covers the full loop; evaluations are not counted.
  - Separate LAFAN1 experiment: return 0.82 in 1.2 h on 1 GPU (Fig. 4a).
- **Iterations per second and total env steps.** Not stated (see the readings above).

---

## 6. Baselines and evaluation

### 6.1 SONIC (§4.1, App. F)
- Released checkpoint and v1.1.
- 128 L40 GPUs × 168 h ≈ 21,000 GPU-h, trained on 611 h of data. SuperMimic uses 264.56 h of BONES-SEED, 2.3× less.
- Ratios: ≈56× wall-clock (stated as >50×) and ≈7200× GPU-hours (stated as >1000×).
- An H100 has 0.74× the fp32 and 5.5× the tensor-core peak of an L40.

### 6.2 PPO-Sim (§4.1, Fig. 4a): the controlled baseline
- PPO in the simulator, on LAFAN1 (40 clips, 2.45 h).
- **Shares** motions, observations, reward, termination, domain randomization, **actor and critic network sizes**, and evaluation protocol.
- Uses **BeyondMimic's PPO hyperparameters** with **8192 envs per GPU**, on 1, 4 or 8 GPUs with an 18 h budget.
- Result: final return 0.60 / 0.70 / 0.72 and never reaches 0.82. SuperMimic reaches 0.82 in 1.2 h on 1 GPU.
- **Codebase.** Not named.
- **mjlab's BeyondMimic PPO values** (`MJ/tasks/tracking/config/g1/rl_cfg.py:110-146`):
  - Hyperparameters: clip 0.2, entropy 0.005, value coefficient 1.0, clipped value loss, 5 epochs × 4 minibatches, learning rate 1e-3 with adaptive KL 0.01, γ 0.99, λ 0.95, max_grad_norm 1.0, 24 steps per env, init_std 1.0, obs normalization.
  - Its (512, 256, 128) ELU networks must be **replaced by the Table A.4 sizes** to match PPO-Sim.

### 6.3 PPO-WM (§4.2, Fig. 5b)
- PPO on world-model rollouts, using walk/run/sprint clips from LAFAN1.
- Batch shapes (parallel rollouts × steps): 4k×32, 8k×24, 8k×32, 16k×32; 16k×32 is best.
- SuperMimic reaches the best PPO-WM return with about **64× fewer imagined steps**.
- Other PPO-WM hyperparameters: UNSPECIFIED.

### 6.4 Metrics (exact definitions)
- **Normalized return** (§4): the undiscounted per-step reward sum over the full episode, divided by its largest possible value. Early termination scores low.
  - "Training return" uses return-based reset starts, so it differs from checkpoint scores (Fig. 4 caption).
- **Failed motions** (App. F.1):
  - Each motion plays once from its first frame.
  - Foot friction 1.0, no domain randomization, no observation noise, deterministic physics.
  - Each policy uses its own gains, action scale and observation construction.
  - A motion fails if SuperMimic's training termination triggers.
  - SONIC's own criterion is different: root/end-effector height deviation > 0.25 m, with no orientation term.
- **8-step prediction error** (Fig. 3c), **GPU-hours**, **world-model steps/s vs MuJoCo Warp** (>10× at 2k–16k envs; 11× / 26× / 62×, the last at 64k; Fig. 5a).
- **Push-recovery settling time** (§4.3): 0.5–4 m/s pushes while holding a T-pose. With the critic the policy settles in 0.5–2.4 s; without it, it never settles and falls at 4 m/s.
- **Velocity step test** (App. G.2):
  - Relative steady-state error = ‖mean velocity error over the last 2 s‖ / ‖command‖, median over non-fallen trials out of 4.
  - 672 commands on 15×15 grids over 3 planes.
  - The RL encoder is better on 650 and distillation on 15; medians 0.16 vs 0.49; distillation falls on 7 commands (translation plus a turn of ≥ 2.8 rad/s).
- **No MPJPE or body-position error is reported.** mjlab's MPKPE and root-relative MPKPE (`MJ/tasks/tracking/mdp/metrics.py:82-105`) would be our addition.
- **mjlab evaluation differs from the paper's protocol.** mjlab's evaluate keeps observation corruption on (`MJ/tasks/tracking/scripts/evaluate.py:73`).

### 6.5 Datasets (App. E, F.3)
- **BONES-SEED (G1 retarget).**
  - 142,220 clips → 129,667 after the name filter (−12,553) → 129,514 after the feasibility filter (−153); 264.56 h.
  - Name filter terms: stair, ladder, box, chair, sit, climb, "jump off 50cm", "look over fence".
  - Feasibility filter at 50 Hz:
    - elevated support: z_min > 0.20 m for ≥ 0.6 s;
    - penetration: z_min < −0.12 m, or < −0.08 m for > 80% of the clip;
    - floating feet: both feet > 0.02 m (mj_geomDistance) for ≥ 0.5 s while upright, at standing height and nearly stationary.
  - Dataset-length sweep subsets: 1, 7, 40, 264 h.
- **LAFAN1.** 40 clips, 2.45 h.

### 6.6 Headline failed-motion counts (Table A.8: SONIC / SuperMimic 1 GPU × 3 h / SuperMimic 2 GPUs × 9.8 h)

| Group | SONIC | 1 GPU × 3 h | 2 GPUs × 9.8 h |
|---|---|---|---|
| Side rolls (22) | 22 | 2 | 0 |
| Flips (44) | 44 | 1 | 0 |
| Fire-in-the-hole turns (20) | 3 | 3 | 0 |
| Crossed legs (10) | 9 | 0 | 0 |
| Lying / resting / standing up (99) | 28 | 8 | 0 |
| Broad jumps (24) | 15 | 0 | 0 |
| Dance routine (8) | 8 | 0 | 0 |
| All other (129,287) | 50 | 11 | 0 |
| **Total** | **179** | **25** | **0** |

- SONIC v1.1 fails 239.
- The project page merges fire-in-the-hole turns into "all other" (129,307 motions; 53 / 14 / none). The numbers are consistent; the PDF is authoritative.

### 6.7 Sim-to-real (§4.4)
- One policy, zero-shot on a Unitree G1, no per-motion tuning: skipping jump, side roll, 90° flip, in-place 360° flip.
- Domain randomization over physical parameters only "suffices for transfer".
- Hardware stack (state estimation, latency, onboard compute): UNSPECIFIED.

---

## 7. Limitations stated by the authors
1. Multi-interface training is understudied. The velocity reward updates only the velocity encoder, so concurrent training equals sequential training (§6).
2. The world model learns only from simulator data; there is no real-robot refinement or system identification (§6).
3. **No terrain or object interaction; flat ground only.** This is ongoing work (§6). Clips with boxes, chairs, stairs and ladders were filtered out (App. E).

Other limitations stated elsewhere in the paper:
- Root-velocity pushes were removed because the deterministic model cannot capture discontinuities (App. A).
- Seed variation is not reported ("several seeds … very stable", §4).
- Benchmark motions are training motions, so the counts measure coverage, not generalization (App. F.2).
- The velocity study is preliminary, with one seed per policy (§4.5, G.2).
- The compute comparison mixes data and hardware differences (§4.1).
- H = 128 is the maximum horizon on an H100 (§4.3).

---

## 8. Training loop pseudocode
Paper-stated items are plain; **(ours)** marks fills.

```python
# per GPU: N=4096 sim envs; B = FIFO replay, 1000 steps/env (stores s, a_exec, r, d, timeout, ō, ξ, motion_id, frame, obs-history (ours))
f_w (MLP 2048x3) [+ learned heads; termination head learned, BCE];  pi = dec(enc_motion(g), o_hist);  V_psi (256,128)
opt_w   = Adam(lr=2e-4)                      # 2 GPUs: sqrt(2)*2e-4
opt_pi  = Adam(lr=2e-4, betas=(0.9, 0.95))
opt_V   = Adam(lr=2e-4)
sched   = cosine(1.0 -> 0.03 over 250_000 its), then const   # all three
sim.reset(all, start=return_based_sampler(), xi=sample_xi_per_episode())   # D.1; Table A.3; no pushes

for it in range(320_000):
    # (1) Collect  -- UTD 0.4 -> 2.5 vectorized steps / iteration (ours, reading (a))
    for _ in range(n_collect(it)):                       # fractional accumulator: 2,3,2,3,... (ours)
        a = pi(o_hist_noisy, g)                          # sim adds eta per Table A.1
        a_exec = a + Normal(0, 0.1**2)
        s_next, r, d, timeout = sim.step(a_exec)         # log PRE-reset s_next (ours)
        B.push(...)                                      # timeout != terminal (ours)
        sampler.update_bin_returns(finished_episodes)    # G_b running estimate (rule ours)
        # resets ~ 0.5*Uniform + 0.4*softmax(-25*G_b/G_max) + 0.1*past-softmax-weight
    # (2) Fit world model (all heads; analytic heads have no loss)
    for m in range(10):                                  # "10 batches of 2048 windows" (10 steps: ours)
        W = B.sample_windows(2048, k=8)                  # windows inside one episode (ours)
        s_hat, L = W.s[0], 0
        for j in range(1, 9):                            # autoregressive, no detach
            s_hat = f_w(s_hat, W.a[j-1], W.xi)
            L += sq(s_hat - W.s[j]) + [sq(r_hat - r_j)] + [sq(o_hat - ō_j)] + BCE(d_hat_j, d_j)
        opt_w.step(clip_global_norm(grad(L.mean()), 1.0))
    # (3) Improve policy through frozen f_w  (BPTT, H=32, gamma=0.99)
    S0 = B.sample_starts(4096)                           # s0, xi, ref index, 10-frame ō history (ours)
    s, c, J = S0.s, 1.0, 0.0
    for t in range(32):
        o = g_w(s, S0.xi) + eta_t                        # sensor noise re-added; no action noise
        a = pi(push_history(o), goal(ref, t, s))         # K=10 future frames
        s_next = f_w(s, a, S0.xi)                        # w frozen (stop-grad on params only)
        r = r_w(s, a, g_t);  d = d_w(s_next, g_{t+1})
        J += gamma**t * c * r;  c = c * (1 - d);  s = s_next
    J += gamma**32 * c * V_psi(s, g_32, S0.xi)           # psi frozen in this step (ours, SHAC)
    loss_pi = -J.mean() / 32                             # reduction: ours (SHAC)
    g = clip_global_norm(grad(loss_pi), 1.0); g = g.clamp(-0.1, 0.1)
    opt_pi.step(g)
    # (4) Critic: GAE(lambda=0.95) targets on the (detached) imagined rollout, Eq. 3
    R[32] = V[32];  R[t] = r[t+1] + gamma*(1-d[t+1])*((1-lam)*V[t+1] + lam*R[t+1])   # computed once
    for epoch in range(5):
        for mb in shuffle_split(4096*32, 4):
            opt_V.step(clip_global_norm(grad(mse(V_psi(s_t, g_t, xi), R_t)), 1.0))
    sched.step()
```

---

## 9. UNSPECIFIED items needed to implement, with proposed defaults (all ours)

1. **State s content.** A heading-local state:
   - pelvis height (1);
   - yaw-removed pelvis orientation, 6D (6);
   - pelvis linear velocity in the heading frame (3) and angular velocity in the pelvis frame (3);
   - q − q0 (29) and q̇ (29);
   - poses of the 14 tracked bodies relative to the pelvis in the heading frame: position (42), 6D orientation (84), linear velocity (42), angular velocity (42);
   - carry, without predicting, a_{t−1} (29) and world xy and yaw (3).
   
   This is about 310 input dimensions. Why: the analytic reward and observation then need no differentiable FK, and the representation is invariant to xy and yaw on flat ground. Alternative: qpos/qvel plus a differentiable G1 FK.
2. **Prediction target.** Normalized deltas in the heading frame. xy and yaw are integrated outside the network; quaternions are composed or re-orthonormalized. Why: residual targets are easier at 50 Hz.
3. **Normalization.** Running mean/std over the buffer for inputs and delta targets, frozen within an update. Eq. 1's MSE is computed on normalized targets. The same std serves as the "nominal error" for the prediction-error metric.
4. **World-model activation and normalization.** ELU (a smooth Jacobian for BPTT), no LayerNorm. Fall back to LayerNorm + SiLU if training is unstable.
5. **Heads.**
   - Reward analytic (Table A.2 on ŝ_{t+1} versus the pre-advance reference frame, as mjlab does at `manager_based_rl_env.py:436-456`).
   - Observation analytic: select from ŝ and add Δq0 from ξ.
   - Termination learned: MLP (256, 128), sigmoid output, input (ŝ, g), BCE, following the §3.1 wording.
6. **ξ encoding.** A 33-d concatenation, each component scaled to [−1, 1] by its Table A.3 range.
7. **"10 batches × 2048".** 10 Adam steps per iteration.
8. **UTD 0.4.** Reading (a): 2.5 sim steps per iteration via a fractional accumulator.
9. **Buffer sampling.** Uniform over valid 8-step windows within one episode. The terminal step stores the pre-reset state, captured via `recorder_manager.record_pre_reset` or `auto_reset=False` (`MJ/envs/manager_based_rl_env.py:444-448`, `:144`). The repo's CLAUDE.md pitfall #2 describes the same auto-reset trap for AMP.
10. **Time-outs.** Label d = 0. Only true terminations get d = 1.
11. **Warm-up.** None, as in Alg. A.1. Start updates once every env has at least 9 logged steps.
12. **Goal g_t.** See §1.3: frames t+1…t+10, 64 per frame; the robot-relative orientation is recomputed from ŝ at every imagined step.
13. **Imagined obs history.** Store the 10-frame noise-free history in B and add fresh η at the start.
14. **Clip end inside H.** Clamp the reference index to the last frame (as mjlab's ONNX export does, `MJ/tasks/tracking/rl/runner.py:208-211`). Mask rewards after the clip end and bootstrap at that step (truncation).
15. **Imagination start distribution ρ_0.** Uniform over non-terminal buffer states.
16. **Sim episode length.** End at the clip end as a time-out, not a terminal, matching "full episode" normalized return and App. F's play-once evaluation.
17. **Reset noise.** Keep mjlab's BeyondMimic reset perturbations (`tracking_env_cfg.py:144-153`). They are reset-time, so they create no intra-episode discontinuity.
18. **Return-based sampler.**
    - G_b = EMA (rate 0.01) of the discounted, episode-truncated return of episodes that start in bin b.
    - Past softmax weight = slow EMA (rate 0.001) of the softmax term.
    - G_max = 0.1·(1 − γ^{T_b})/(1 − γ), where T_b is the number of steps left in the clip.
    - Start uniformly within the bin (as `commands.py:278-282`).
19. **Domain-randomization frequency.** `mode="reset"`, i.e. per episode.
20. **Root body.** Pelvis for the root reward terms and for termination, following the paper's naming. Note that mjlab uses torso_link (`env_cfgs.py:41`).
21. **Tracked bodies T.** mjlab's 14 bodies (`env_cfgs.py:42-57`).
22. **Joint limit.** Soft limits at 0.9× range (`MJ/envs/mdp/rewards.py:81-96`).
23. **α, gains, q0.** mjlab's BeyondMimic G1 values (§1.4).
24. **Policy and critic.** ELU activations; running-mean/std input normalizers updated from sim data only and frozen in the backward pass. The final actor layer is scaled ×0.01 at init so it starts near q0. The actor output is linear and unclipped, matching PPO's Gaussian mean; monitor |a| for world-model exploitation.
25. **Actor loss.** Mean over the batch divided by H (SHAC). ψ is frozen in the actor step, and the gradient enters through ŝ_H.
26. **Critic loss.** Mean MSE; minibatch 32,768.
27. **Return normalization.** None (rewards are Δt-scaled and bounded).
28. **Optimizer details.** Adam eps 1e-8 and no weight decay for all; world-model and critic β at the PyTorch default.
29. **Precision.** TF32 on for all matmuls (stated for the world model).
30. **Velocity-only items.** The history "feature" for the velocity encoder: a linear projection of the flattened 10-frame o.

---

## 10. Repo-side notes for the side-by-side plan (nothing was run or edited)

- **Motion data.**
  - The repo's `.npy` clips are [frames, 36] qpos at 30 fps with wxyz quaternions: 2 locomotion clips (120 and 125 frames ≈ 4.0 and 4.2 s) and 2 dual-arm clips (209 frames ≈ 6.93 s each).
  - mjlab's tracking `MotionLoader` needs one 50 fps `.npz` (joint_pos/vel, body_pos/quat/lin_vel/ang_vel; `MJ/tasks/tracking/mdp/commands.py:36-60`). It loads **one file**, so multiple clips need concatenation or a multi-clip loader.
  - `MJ/scripts/csv_to_npz.py` expects CSV with **xyzw** quaternions (it reorders [3,0,1,2] at `:62`), 30→50 fps by default (`:342-343`), and its hardcoded joint order (`:394-423`). Verify that the repo's joint order matches.
  - It writes `/tmp/motion.npz` and then uploads to W&B (`:309-334`). Use `WANDB_MODE=offline` or a small converter. `render=False` is the default (`:345`), which matters because the cluster has no OpenGL.
- **Environment changes for both arms** (starting from `MJ/tasks/tracking`):
  - Table A.1 observation with 10-frame history, plus the K = 10 goal.
  - Remove the push, self-collision and end-effector terminations; joint limit −2.
  - Pelvis termination; domain randomization per reset, with ξ exposed.
  - Return-based sampler; episode ends at the clip end; pre-reset state logging.
- **PPO baseline (PPO-Sim).** Same environment; mjlab's BeyondMimic PPO hyperparameters (`rl_cfg.py:128-145`); 8192 envs; **Table A.4 network sizes**. Evaluate noise-free and without domain randomization at friction 1.0, unlike `evaluate.py:73`.
- **"Normal walking" needs a decision.** Either (a) track the 2 locomotion clips (an apples-to-apples tracking comparison), or (b) use the paper's velocity interface (App. B/G) and compare with the repo's existing `Mjlab-G1-Locomotion` PPO.
- **Lifting is outside the paper's scope** (§6; App. E removes "box" clips). The dual-arm clips have no object trajectory, so SuperMimic as published can track only the robot's squat-lift motion without the box. Adding the box would need object pose and velocity in s, object mass and friction in ξ, and grasp-contact dynamics in a deterministic MLP. None of this is in the paper.
- **Scale.** About 8 s (walking) and about 14 s (lift) of reference, compared with the paper's smallest sweep point of 1 h and its smallest controlled comparison, LAFAN1 at 2.45 h. The paper never shows an advantage over PPO on single clips.

Sources:
- [SuperMimic, arXiv:2610.28480 (PDF)](https://arxiv.org/pdf/2610.28480)
- [SuperMimic project page](https://supermimic.github.io)
- [arXiv HTML endpoint (returned "No HTML")](https://arxiv.org/html/2610.28480v1)
- Web search with no relevant hit: [papers.cool OpenReview listing](https://papers.cool/venue/C6VxzSpjrv@OpenReview), [HAIC arXiv 2602.11758](https://arxiv.org/pdf/2602.11758)