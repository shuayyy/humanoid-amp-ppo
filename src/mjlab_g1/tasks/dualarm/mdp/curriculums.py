"""Success-adaptive curricula for the dual-arm lift task.

ResMimic-style curriculum design: the virtual object controller's assistance
and the object's reset-height bootstrap decay based on demonstrated task
success (lift-success EMA tracked by the env), not on a blind step schedule.
If performance drops after a difficulty increase, decay pauses until the
policy recovers, so the curriculum self-regulates.
"""

from __future__ import annotations

import torch


def _validate_schedule_start_and_order(
    schedule,
    name: str,
) -> None:
    if len(schedule) == 0:
        raise ValueError(f"{name} must not be empty.")
    if schedule[0][0] != 0:
        raise ValueError(f"{name} first stage must start at 0.")

    previous_step = -1
    for stage in schedule:
        start_step = stage[0]
        if start_step <= previous_step:
            raise ValueError(f"{name} steps must be strictly ascending.")
        previous_step = start_step


def _interpolate_stage_values(schedule, common_step_counter: int) -> tuple[float, ...]:
    if common_step_counter <= schedule[0][0]:
        return tuple(float(value) for value in schedule[0][1:])

    for start_stage, end_stage in zip(schedule, schedule[1:], strict=False):
        start_step = start_stage[0]
        end_step = end_stage[0]
        if common_step_counter <= end_step:
            progress = (common_step_counter - start_step) / max(
                end_step - start_step,
                1,
            )
            return tuple(
                float(start_value) + (float(end_value) - float(start_value)) * progress
                for start_value, end_value in zip(
                    start_stage[1:],
                    end_stage[1:],
                    strict=True,
                )
            )

    return tuple(float(value) for value in schedule[-1][1:])


def virtual_pd_assistance_curriculum(
    env,
    env_ids: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Success-adaptive decay of the virtual-PD assistance scale.

    While the lift-success EMA stays above ``assistance_decay_threshold``, the
    global assistance scale drops by ``assistance_decay_step`` at most once
    every ``assistance_decay_interval`` env steps. Two-sided with hysteresis:
    between the recovery and decay thresholds the scale holds; below
    ``assistance_recovery_threshold`` (difficulty was raised too fast and the
    policy collapsed) the scale steps back UP so the policy is never stranded
    on a rung it cannot climb.
    """
    del env_ids
    cfg = env.cfg

    # Fine-grained rungs at the hard end: smaller steps, longer consolidation.
    if env.virtual_pd_assistance_scale <= cfg.assistance_fine_scale_threshold:
        decay_step = cfg.assistance_fine_decay_step
        decay_interval = cfg.assistance_fine_decay_interval
    else:
        decay_step = cfg.assistance_decay_step
        decay_interval = cfg.assistance_decay_interval

    ready = (
        env.common_step_counter - env.last_assist_decay_step >= decay_interval
    )
    if ready and env.lift_success_ema >= cfg.assistance_decay_threshold:
        env.virtual_pd_assistance_scale = max(
            cfg.assistance_min_scale,
            env.virtual_pd_assistance_scale - decay_step,
        )
        env.last_assist_decay_step = env.common_step_counter
    elif (
        ready
        and env.lift_success_ema < cfg.assistance_recovery_threshold
        and env.virtual_pd_assistance_scale < 1.0
    ):
        env.virtual_pd_assistance_scale = min(
            1.0,
            env.virtual_pd_assistance_scale + decay_step,
        )
        env.last_assist_decay_step = env.common_step_counter

    return {
        "scale": torch.tensor(
            env.virtual_pd_assistance_scale, device=env.device, dtype=torch.float32
        ),
        "lift_success_ema": torch.tensor(
            env.lift_success_ema, device=env.device, dtype=torch.float32
        ),
    }


def object_reset_height_curriculum(
    env,
    env_ids: torch.Tensor,
) -> torch.Tensor:
    """Success-adaptive decay of the object's reset-height bootstrap.

    Sequenced after the assistance curriculum: the spawn height only starts
    lowering once the assistance scale has dropped below
    ``reset_height_start_assistance``, i.e. the robot first learns to carry
    the object at a comfortable height, then learns the deeper reach.
    """
    del env_ids
    cfg = env.cfg

    ready = (
        env.common_step_counter - env.last_height_decay_step
        >= cfg.reset_height_decay_interval
    )
    if (
        ready
        and env.virtual_pd_assistance_scale <= cfg.reset_height_start_assistance
        and env.lift_success_ema >= cfg.reset_height_decay_threshold
    ):
        env.object_reset_height_frac = max(
            0.0,
            env.object_reset_height_frac - cfg.reset_height_decay_step,
        )
        env.last_height_decay_step = env.common_step_counter

    return torch.tensor(
        env.object_reset_height_frac, device=env.device, dtype=torch.float32
    )


def object_spawn_range_curriculum(
    env,
    env_ids: torch.Tensor,
) -> torch.Tensor:
    """Success-adaptive widening of the object spawn-pose randomization.

    Sequenced last in the difficulty ladder: only starts widening once the
    reset-height bootstrap is nearly finished
    (``object_reset_height_frac <= spawn_range_start_height_below``), so the
    policy first masters the nominal grasp, then generalizes it to offset and
    rotated spawns.
    """
    del env_ids
    cfg = env.cfg

    ready = (
        env.common_step_counter - env.last_spawn_widen_step
        >= cfg.spawn_range_widen_interval
    )
    if (
        ready
        and env.object_reset_height_frac <= cfg.spawn_range_start_height_below
        and env.lift_success_ema >= cfg.spawn_range_widen_threshold
    ):
        env.object_spawn_range_frac = min(
            1.0,
            env.object_spawn_range_frac + cfg.spawn_range_widen_step,
        )
        env.last_spawn_widen_step = env.common_step_counter

    return torch.tensor(
        env.object_spawn_range_frac, device=env.device, dtype=torch.float32
    )


def assist_force_penalty_curriculum(
    env,
    env_ids: torch.Tensor,
) -> torch.Tensor:
    """Ramp the virtual-assistance force penalty in as assistance decays.

    weight = assist_force_penalty_max_weight * (1 - assistance_scale)

    At full assistance (bootstrap phase) the penalty is zero, so early
    exploration is not punished for the help it cannot avoid. As the
    assistance scale drops, the policy is increasingly paid for making the
    virtual controller's force unnecessary — i.e. for actually carrying the
    object itself (ResMimic's take-over incentive).
    """
    del env_ids

    weight = env.cfg.assist_force_penalty_max_weight * (
        1.0 - env.virtual_pd_assistance_scale
    )
    term_cfg = env.dualarm_reward_manager.get_term_cfg("assist_force_penalty")
    term_cfg.weight = float(weight)

    return torch.tensor(weight, device=env.device, dtype=torch.float32)


def smoothness_curriculum(
    env,
    env_ids: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Ramp action-rate / joint-velocity penalties in as assistance decays.

    weight = max_weight * (1 - assistance_scale)

    Free fidgeting was the cause of the perpetual stance-shuffling during the
    hold; ramping (instead of penalizing from step 0) keeps early exploration
    unhindered, mirroring the assist-force penalty schedule.
    """
    del env_ids

    ramp = 1.0 - env.virtual_pd_assistance_scale
    action_rate_weight = env.cfg.action_rate_max_weight * ramp
    joint_vel_weight = env.cfg.joint_vel_max_weight * ramp

    term_cfg = env.reg_reward_manager.get_term_cfg("action_rate_l2")
    term_cfg.weight = float(action_rate_weight)
    term_cfg = env.reg_reward_manager.get_term_cfg("joint_vel_l2")
    term_cfg.weight = float(joint_vel_weight)

    return {
        "action_rate_weight": torch.tensor(
            action_rate_weight, device=env.device, dtype=torch.float32
        ),
        "joint_vel_weight": torch.tensor(
            joint_vel_weight, device=env.device, dtype=torch.float32
        ),
    }


def missing_grasp_curriculum(
    env,
    env_ids: torch.Tensor,
) -> torch.Tensor:
    """Ramp the missing-grasp penalty in as assistance decays.

    weight = missing_grasp_max_weight * (1 - assistance_scale)
    """
    del env_ids

    weight = env.cfg.missing_grasp_max_weight * (
        1.0 - env.virtual_pd_assistance_scale
    )
    term_cfg = env.dualarm_reward_manager.get_term_cfg("missing_grasp_during_lift")
    term_cfg.weight = float(weight)

    return torch.tensor(weight, device=env.device, dtype=torch.float32)


def feet_slip_curriculum(
    env,
    env_ids: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """
    Linear schedule for foot-slip penalty strength and deadzone.
    """
    del env_ids

    schedule = env.cfg.feet_slip_curriculum_schedule
    _validate_schedule_start_and_order(schedule, "feet_slip_curriculum_schedule")

    for _, weight, threshold_min in schedule:
        if weight > 0.0:
            raise ValueError("feet_slip_curriculum_schedule weights must be <= 0.0.")
        if threshold_min < 0.0:
            raise ValueError(
                "feet_slip_curriculum_schedule thresholds must be non-negative."
            )

    selected_weight, selected_threshold_min = _interpolate_stage_values(
        schedule,
        env.common_step_counter,
    )

    term_cfg = env.dualarm_reward_manager.get_term_cfg("feet_slip")
    term_cfg.weight = float(selected_weight)
    term_cfg.params["threshold_min"] = float(selected_threshold_min)

    return {
        "weight": torch.tensor(
            selected_weight,
            device=env.device,
            dtype=torch.float32,
        ),
        "threshold_min": torch.tensor(
            selected_threshold_min,
            device=env.device,
            dtype=torch.float32,
        ),
    }
