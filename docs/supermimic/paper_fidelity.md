# SuperMimic implementation vs the paper (audit, 2026-10-22)

Primary source: the arXiv LaTeX of 2610.28480 (`sections/method.tex`, `sections/appendix.tex`).
Bins: **match**; **documented deviation** (brief / `# ours`); **object extension** (the paper has no
objects, so anything box-related is necessarily ours); **undocumented** (audit findings).

## What the paper says about objects

- Limitations (`conclusion.tex`): "No terrain or object interaction. SuperMimic trains only on flat
  ground. Extending it to terrain and object interaction is ongoing work." Untested, not ruled out.
- The only "cannot capture" claim (`appendix.tex:45`) is about BeyondMimic's root-velocity
  **perturbation pushes** in training DR ("state discontinuities that the deterministic world model
  cannot capture"); the paper removes them, and so do we (`del cfg.events["push_robot"]`).
- Its data filter removes clips with boxes, stairs, climbing (`appendix.tex:282-295`): the box
  lift is outside the paper's tested regime.

## Env / MDP (checked against mjlab `tracking_env_cfg.py` and `smtrack/env_cfgs.py`)

| Paper | Ours | Verdict |
|---|---|---|
| Reward: body pos/ori/lin vel/ang vel std 0.3/0.4/1.0/3.14 w 1; root pos/ori 0.3/0.4 w 0.5; action rate −0.1; joint limit −2; no self-collision; × dt | mjlab values, joint limit −2, self-collision removed, dt-scaled | match |
| Termination: root height err > 0.25 m or tilt > 0.8; no end-effector termination | same thresholds, anchor = `torso_link` | documented (paper text says pelvis; mjlab/BeyondMimic use torso) |
| DR ξ: friction U[0.3, 1.2], encoder offset U±0.01, torso CoM ±(0.025, 0.05, 0.05), per episode | mjlab ranges, `mode="reset"` | match |
| No root-velocity pushes | `push_robot` deleted | match |
| Obs: q−q0, q̇, IMU ω, gravity, a_{t−1}, 10-frame history; noise ±0.01/0.5/0.2/0.05 | same | match |
| Goal: K = 10 future reference frames | same | match |
| Return-based reset sampling (1 s bins, 50 % uniform / 40 % softmax(−25 G_b/G_max) / 10 % past) | mjlab adaptive sampler + `adaptive_uniform_mix=0.5` | documented (same 50 % uniform floor; failure-driven rest instead of return-driven) |
| Box state, box ξ (mass, friction scale), box reward/termination/obs | ours | object extension |

## Hyperparameters (Table `tab:hyperparams` vs `rl/supermimic/config.py`)

| Paper | Ours | Verdict |
|---|---|---|
| WM MLP (2048, 2048, 2048), k = 8, Adam lr 2e-4, 10 × 2048 windows, clip 1.0 | same | match |
| H = 32, 4096 imagined, Adam β (0.9, 0.95), clip 1.0 then element-wise 0.1, γ 0.99 | same | match |
| Critic MLP (256, 128), λ 0.95, 5 epochs × 4 minibatches, clip 1.0 | same | match |
| Encoder/decoder MLP (2048, 1024, 512, 256), latent 32 tanh | same | match |
| 4096 envs, ε ~ N(0, 0.1²), replay 1000 steps/env, UTD 0.4 (2.5 env steps/iter), 1 policy update per WM update | same | match |
| Cosine LR to 0.03× over 250k of 320k iterations | same shape, compressed to 0.78 × max_iterations | documented |
| Activations, termination-head size | not given; ours SiLU+LayerNorm (WM), ELU (actor/critic) | paper-unspecified |

## Learner code: world model, replay, collection

No bug found. Code references are `rl/supermimic/` files.

| Paper | Ours | Verdict |
|---|---|---|
| Sim executes a + ε, buffer stores the executed noisy action | `algorithm.py:204` `a_exec = a + std * randn`, stored as `a_exec` | match |
| Ring buffer, oldest replaced | `replay.py` shared-pointer ring | match |
| Eq. 1: k = 8 autoregressive windows from logged s₀ on logged actions, gradient through the chain | `world_model.py:658-663` re-steps from the prediction, no detach | match |
| Eq. 1 state term ‖ŝ − s‖² | per-dimension error / σ_nom, mean over dims; local frames, quaternion log for orientation | documented (design §3.3); the paper must normalize somehow but does not say how |
| Heads: each learned or analytic | dynamics + termination learned (BCE); reward + observation analytic (exact FK) | documented; analytic heads have no loss term, as the paper allows |
| Episode boundaries | windows masked after the first done (`replay.py` cumprod mask), terminal step kept with its pre-reset target | match |
| ξ per episode with the window | logged per slot, steps from other episodes masked | match |
| Residual prediction, φ input normalization, σ_out output scale, LayerNorm, zero-init output | ours | documented (design §2.2, §3.1) |
| **Fig. 4c "8-step error, state only, normalized by a predefined nominal error, measured online on policy rollouts" (0.007–0.011)** | `wm/err_norm_step8` = squared, mean over dims, σ_nom = running std of the one-step state change, scored on windows drawn uniformly from the whole 1000-step replay ring | **undocumented: metric only, training unaffected.** Not comparable to the paper's number. Converged walk/squat log 0.017–0.025 (squared). |
| Constants `phi_eps 1e-2`, `out_floor 1e-4`, `nom_floor 1e-3`, `term_init_prob 0.01` | not in config | undocumented, inert (no floor binds in walk model_9000) |

## Learner code: imagination, policy, critic

No bug found.

| Paper | Ours | Verdict |
|---|---|---|
| Rollouts start from a replay state with its recorded ξ; H = 32; 4096 rollouts | `imagination.py:305-311`, starts uniform over valid slots | match (start filter: design §3.5) |
| WM frozen, gradient through its Jacobians w.r.t. ŝ and a | `algorithm.py:251` freeze; no detach in `wm.step` | match |
| Deterministic policy in imagination, sensor noise η re-added every step | `imagination.py:263, 270-272` | match |
| Survival weight c₀ = 1, c_{t+1} = c_t(1 − d̂_{t+1}), c_t multiplies r_{t+1}; + γ^H c_H V(ŝ_H) with gradient into θ | `imagination.py:327, 355-356, 369-371` | match |
| Critic: λ-returns (Eq. 3), R̂_H = V_H, targets fixed per rollout, no target network | `algorithm.py:402-411` | match |
| Critic input V(ŝ, g, ξ) | V(φ(ŝ), g, ξ, t_left) | documented (design §4.3, `config.py`). Side effect: φ is translation/yaw invariant, so V cannot see global anchor xy drift (critique M1, unresolved) |
| Actor loss reduction, normalizers, ELU, output init | ours | documented |
| Reward analytic = env reward | parity check at startup aborts above 1e-3 | match |
| Termination: learned d̂ (no hard threshold in imagination) | `world_model.py:582` | match |
| **Previous action a_{t−1} in the observation** | **sim/replay history uses the executed noisy a + ε (`runner.py:292`); imagination uses the clean a (`imagination.py:270`)** | **paper silent on which; our sim and imagination disagree with each other** (known: critique M4 proposed storing `a_pol` beside `a_exec`; never implemented). Fixing it changes the observation, so it needs reruns of every task it should be compared on. The actor's 29 previous-action inputs carry N(0, 0.1²) noise in sim data and in the first ≤ 10 imagined frames, none after; the sim action-rate term is ~1.2 % of max step reward lower than imagination's. Eval has no action noise, so it matches imagination. Small; not a candidate explanation for task-level failures |

## Verdict

The learner and env match the paper's equations and tables; the remaining differences are
documented design fills or necessary object extensions. Two undocumented items, neither a bug:
the logged WM-error metric is not the paper's Fig. 4c measurement (comparability only), and the
previous-action observation is noisy in sim but clean in imagination (small input shift). Our
lift results therefore test SuperMimic itself, outside the regime the paper evaluated.
