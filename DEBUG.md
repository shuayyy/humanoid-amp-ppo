# Humanoid Loco-Manipulation RL: Feet Swing Usage Check

This note compares open-source humanoid loco-manipulation RL projects and checks whether they use feet swing / gait phase rewards.

## Summary

| Repo                                 | Loco-Manipulation Task Checked      |              Feet Swing Used? | Notes                                                                                                            |
| ------------------------------------ | ----------------------------------- | ----------------------------: | ---------------------------------------------------------------------------------------------------------------- |
| `Humanoid-SkillBlender/SkillBlender` | `g1_task_lift`                      |                           Yes | Uses gait phase, stance/swing mask, feet air-time reward, foot contact number reward, and foot slip penalty.     |
| `NVlabs/GR00T-VisualSim2Real`        | `walk_stand_place_grasp_turn_homie` | No classic swing reward found | Uses foot-contact stability logic instead: penalizes arm/finger motion when not both feet are contacting ground. |
| `LeCAR-Lab/FALCON`                   | force/loco-manip style humanoid RL  |                           Yes | Uses explicit swing/stance phase variables and feet swing height penalty.                                        |

## Direct Finding

For humanoid loco-manipulation RL:

```text
SkillBlender: uses feet swing / gait phase.
FALCON: uses feet swing / gait phase.
GR00T VisualSim2Real: does not use classic feet swing reward; uses contact-stability penalties instead.
```

## Meaning for `wb_grasp`

For the `wb_grasp` task:

```text
walk/locomotion → both hands contact object → lift object
```

Do not add full periodic swing/gait reward first.

A safer first choice is GR00T-style contact stability:

```text
penalize large arm/hand motion when only one foot is contacting ground
```

Reason:

```text
The task is grasp/lift under balance constraints, not pure locomotion tracking.
Classic swing reward may force unnecessary walking behavior before grasping is stable.
```

## Recommended First Reward Direction

Use:

```text
upright reward
feet slip penalty
foot contact/contact force as critic information
hand-object contact reward
lift reward gated by both-hand contact
optional penalty for arm/hand motion during single-foot support
```

Avoid initially:

```text
full gait phase reward
strict swing timing reward
forced feet air-time reward
```


## WB Grasp Debug Guide

This is the clean reading order for the active WB grasp implementation.

## 1. Entrypoints

1. [src/mjlab_husky/scripts/train.py](/home/shuaiyyy/HUMANOIDS/humanoid_skateboarding/src/mjlab_husky/scripts/train.py)
   Read first. This shows how training is launched, how the env config is loaded, how video is wrapped, and which runner is used.
2. [src/mjlab_husky/scripts/play.py](/home/shuaiyyy/HUMANOIDS/humanoid_skateboarding/src/mjlab_husky/scripts/play.py)
   Read second. This shows inference/play behavior, viewer selection, checkpoint loading, and why play is deterministic.

## 2. Task Registration

3. [src/mjlab_husky/tasks/wb_grasp/config/g1/__init__.py](/home/shuaiyyy/HUMANOIDS/humanoid_skateboarding/src/mjlab_husky/tasks/wb_grasp/config/g1/__init__.py)
   This shows how the WB grasp task is registered.
4. [src/mjlab_husky/tasks/wb_grasp/config/g1/env_cfgs.py](/home/shuaiyyy/HUMANOIDS/humanoid_skateboarding/src/mjlab_husky/tasks/wb_grasp/config/g1/env_cfgs.py)
   This is the real task assembly file. Read this carefully. It wires:
   - `num_envs`
   - robot entity
   - toaster entity
   - sensors
   - play vs train overrides

## 3. Main Task Config

5. [src/mjlab_husky/tasks/wb_grasp/wb_grasp_env_cfg.py](/home/shuaiyyy/HUMANOIDS/humanoid_skateboarding/src/mjlab_husky/tasks/wb_grasp/wb_grasp_env_cfg.py)
   This is the most important config file after `env_cfgs.py`. Read top to bottom:
   - observations
   - actions
   - events/reset randomization
   - rewards
   - terminations
   - scene config

## 4. Actual Environment Logic

6. [src/mjlab_husky/envs/g1_wb_grasp_rl_env.py](/home/shuaiyyy/HUMANOIDS/humanoid_skateboarding/src/mjlab_husky/envs/g1_wb_grasp_rl_env.py)
   This is the core runtime implementation. Read fully. Focus on:
   - `__init__`
   - `_init_buffers`
   - `_init_ids_buffers`
   - `step`
   - `_reset_idx`
   - `_compute_contact`
   - `_resample_contact_phases`
   - `_get_hand_toaster_dis`

## 5. Reward and Termination Logic

7. [src/mjlab_husky/tasks/wb_grasp/mdp/rewards.py](/home/shuaiyyy/HUMANOIDS/humanoid_skateboarding/src/mjlab_husky/tasks/wb_grasp/mdp/rewards.py)
   Read every reward function and map each one back to `wb_grasp_env_cfg.py`.
8. [src/mjlab_husky/tasks/wb_grasp/mdp/terminations.py](/home/shuaiyyy/HUMANOIDS/humanoid_skateboarding/src/mjlab_husky/tasks/wb_grasp/mdp/terminations.py)
   Read next. This tells you what ends episodes.
9. [src/mjlab_husky/tasks/wb_grasp/mdp/observations.py](/home/shuaiyyy/HUMANOIDS/humanoid_skateboarding/src/mjlab_husky/tasks/wb_grasp/mdp/observations.py)
   Read if any observation term is unclear.
10. [src/mjlab_husky/tasks/wb_grasp/mdp/velocity_command.py](/home/shuaiyyy/HUMANOIDS/humanoid_skateboarding/src/mjlab_husky/tasks/wb_grasp/mdp/velocity_command.py)
   Read only if command logic matters.

## 6. Asset Definitions

11. [src/mjlab_husky/asset_zoo/robots/skateboard/g1_constants.py](/home/shuaiyyy/HUMANOIDS/humanoid_skateboarding/src/mjlab_husky/asset_zoo/robots/skateboard/g1_constants.py)
   Read fully. This defines:
   - robot XML path
   - toaster XML path
   - actuators
   - init keyframes
   - action scales and constants
12. [src/mjlab_husky/asset_zoo/robots/skateboard/xmls/g1_realsense.xml](/home/shuaiyyy/HUMANOIDS/humanoid_skateboarding/src/mjlab_husky/asset_zoo/robots/skateboard/xmls/g1_realsense.xml)
   Read after `g1_constants.py`. Use it to verify joints, sites, bodies, IMU, and wrists.
13. [src/mjlab_husky/asset_zoo/robots/skateboard/Toaster003/model.xml](/home/shuaiyyy/HUMANOIDS/humanoid_skateboarding/src/mjlab_husky/asset_zoo/robots/skateboard/Toaster003/model.xml)
   Read to verify toaster root, freejoint, marker sites, and collision bodies.

## 7. RL Wrapper Layer

14. [src/mjlab_husky/rl/vecenv_wrapper.py](/home/shuaiyyy/HUMANOIDS/humanoid_skateboarding/src/mjlab_husky/rl/vecenv_wrapper.py)
   Read to see how MJLab env is adapted to RSL-RL.
15. [src/mjlab_husky/rl/vecenv_wrapper_wb.py](/home/shuaiyyy/HUMANOIDS/humanoid_skateboarding/src/mjlab_husky/rl/vecenv_wrapper_wb.py)
   Read only if WB-specific wrapper behavior matters.

## 8. RL Config

16. [src/mjlab_husky/tasks/wb_grasp/config/g1/rl_cfg.py](/home/shuaiyyy/HUMANOIDS/humanoid_skateboarding/src/mjlab_husky/tasks/wb_grasp/config/g1/rl_cfg.py)
   This is the PPO hyperparameter file.
17. [src/mjlab_husky/rl/config.py](/home/shuaiyyy/HUMANOIDS/humanoid_skateboarding/src/mjlab_husky/rl/config.py)
   Shared RL config and AMP-related defaults.

## 9. PPO Implementation

18. [rsl_rl/runners/on_policy_runner.py](/home/shuaiyyy/HUMANOIDS/humanoid_skateboarding/rsl_rl/runners/on_policy_runner.py)
   Read to understand rollout collection, logging, and checkpoints.
19. [rsl_rl/algorithms/ppo.py](/home/shuaiyyy/HUMANOIDS/humanoid_skateboarding/rsl_rl/algorithms/ppo.py)
   Read PPO update logic.
20. [rsl_rl/modules/actor_critic.py](/home/shuaiyyy/HUMANOIDS/humanoid_skateboarding/rsl_rl/modules/actor_critic.py)
   Read how actor/critic outputs are produced and why play uses deterministic mean action.

## 10. Only If You Care About AMP / Discriminator

21. [rsl_rl/runners/amp_on_policy_runner.py](/home/shuaiyyy/HUMANOIDS/humanoid_skateboarding/rsl_rl/runners/amp_on_policy_runner.py)
22. [rsl_rl/algorithms/amp_ppo.py](/home/shuaiyyy/HUMANOIDS/humanoid_skateboarding/rsl_rl/algorithms/amp_ppo.py)
23. [rsl_rl/modules/discriminator_multi.py](/home/shuaiyyy/HUMANOIDS/humanoid_skateboarding/rsl_rl/modules/discriminator_multi.py)
24. [rsl_rl/utils/motion_loader_g1.py](/home/shuaiyyy/HUMANOIDS/humanoid_skateboarding/rsl_rl/utils/motion_loader_g1.py)

## What To Ignore First

- old skater files unless doing historical comparison
- `test_scene/` unless debugging raw MuJoCo
- old logs/checkpoints until the live code path is clear

## Best Reading Strategy

1. `train.py` -> task registration -> `env_cfgs.py`
2. `wb_grasp_env_cfg.py` + `g1_wb_grasp_rl_env.py`
3. `rewards.py` + `terminations.py`
4. `g1_constants.py` + the two XMLs
5. wrapper + PPO runner + actor_critic
