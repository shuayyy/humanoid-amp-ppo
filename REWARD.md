# wb_grasp Reward Fix TODO

## Critical reward logic

* [ ] Check `hands_at_markers`: make sure marker distance threshold is not too loose.

  * Problem: if `hand_dist_thresh` is large, robot gets marker reward without real grasp.
  * Target: use around `0.05` to `0.10`, not `0.75`.

* [ ] Check `lift` reward: lift reward must require real hand-object contact.

  * Problem: object lift reward should not activate only from hands being near markers.
  * Required logic:

    ```text
    lift_reward = lift_height_reward * both_hands_close * both_hands_contact
    ```

* [ ] Check `grasp_success`: success must require both hand contact.

  * Problem: success should not be `hands_close & lifted` only.
  * Required logic:

    ```text
    success = both_hands_close & both_hands_contact & lifted
    ```

## Contact reward checks

* [ ] Verify `hands_contact` reads correct sensors:

  ```text
  left_hand_toaster_contact
  right_hand_toaster_contact
  ```

* [ ] Verify `hands_contact` requires both hands, not only one hand.

  * Required:

    ```text
    left_contact & right_contact
    ```

* [ ] Verify hand contact sensors use correct XML geom names:

  ```text
  left_hand_collision
  left_wrist_collision
  right_hand_collision
  right_wrist_collision
  object
  ```

## Remove broken / misleading rewards

* [ ] Remove or disable `self_collisions` reward for now.

  * Problem: current `robot_collision` sensor is not a real self-collision sensor.
  * Action:

    ```python
    # "self_collisions": RewardTermCfg(...)
    ```

* [ ] Remove or reduce `at_least_one_foot_contact`.

  * Problem: fallen robot can still have one foot touching ground.
  * Better: rely on `upright` + fall termination.
  * If kept, make it very small.

## Locomanip stability reward

* [ ] Add GR00T-style stability penalty:

  * Penalize large arm/hand motion when not both feet are contacting ground.
  * Purpose: prevent manipulation while unstable.

  Logic:

  ```text
  if foot_contact_count != 2:
      penalize arm_joint_velocity / hand_velocity
  ```

* [ ] Do not add full gait/swing reward yet.

  * Reason: task is grasp/lift, not locomotion tracking.
  * Add feet swing only if walking itself fails after reward bugs are fixed.

## Reward staging / gating

* [ ] Check if all rewards are active at all times.

  * Problem: locomotion/contact/lift all active from step 0 can confuse PPO.

* [ ] Gate lift reward:

  ```text
  lift reward active only after both_hands_contact
  ```

* [ ] Gate strong contact reward:

  ```text
  contact reward active only when hands are near markers
  ```

* [ ] Keep locomotion reward active early:

  ```text
  hand_to_toaster / hands_at_markers active from start
  ```

## Regularization sanity

* [ ] Keep `upright` reward.

  * This is important.

* [ ] Keep `feet_slip` penalty only if its contact sensor is correct.

* [ ] Keep `feet_stumble` only if it uses correct foot contact forces.

* [ ] Keep action smoothness penalties:

  ```text
  action_rate_l2
  action_acc_l2
  joint_torques_l2
  ```

* [ ] Do not increase regularization weights before task rewards are logically correct.

## Smoke tests after reward fixes

* [ ] Print reward terms for random policy.

  * Check no reward is always zero unless expected.
  * Check no reward is always max from reset.

* [ ] Manually place hands near markers and verify:

  ```text
  hands_at_markers increases
  hands_contact still zero until contact
  ```

* [ ] Manually create hand-object contact and verify:

  ```text
  hands_contact > 0
  ```

* [ ] Lift object with no hand contact and verify:

  ```text
  lift reward = 0
  success = False
  ```

* [ ] Lift object with both hand contact and verify:

  ```text
  lift reward > 0
  success can become True after hold steps
  ```

* [ ] Fall robot and verify:

  ```text
  upright reward low
  at_least_one_foot_contact does not dominate
  total reward does not stay high
  ```

## REFERENCES FOR FELLOVER

| Repo / task                        | Termination conditions                                                                            |
| ---------------------------------- | ------------------------------------------------------------------------------------------------- |
| `mujocolab/mjlab` G1 velocity      | `time_out`, `fell_over`, `out_of_terrain_bounds`                                                  |
| `mujocolab/mjlab` G1 flat velocity | `time_out`, `fell_over` only; flat config removes `out_of_terrain_bounds`                         |
| `mujocolab/mjlab` G1 tracking      | `time_out`, bad anchor position, bad anchor orientation, bad end-effector/body position           |
| `mujocolab/g1_spinkick_example`    | all G1 tracking terminations + base angular velocity too high                                     |
| `lzyang2000/twist2_mjlab`          | `time_out`, motion end, root height error, roll limit, pitch limit, velocity too large, pose fail |
| `Nagi-ovo/mjlab-homierl`           | `time_out`, `fell_over`                                                                           |
