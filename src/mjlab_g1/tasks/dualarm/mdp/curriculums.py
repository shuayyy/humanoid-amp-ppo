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


def _select_stage(schedule, common_step_counter: int):
    selected_stage = schedule[0]
    for stage in schedule:
        if common_step_counter >= stage[0]:
            selected_stage = stage
        else:
            break
    return selected_stage


def _interpolate_scalar_stage(schedule, common_step_counter: int) -> float:
    return _interpolate_stage_values(schedule, common_step_counter)[0]


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
) -> torch.Tensor:
    """
    Global common-step linear schedule for virtual-PD assistance.

    The same scale is shared by all environments. Contact gating belongs to the
    reward, not the virtual-PD assistance.
    """
    del env_ids

    schedule = env.cfg.virtual_pd_curriculum_schedule
    _validate_schedule_start_and_order(
        schedule,
        "virtual_pd_curriculum_schedule",
    )

    for _, scale in schedule:
        if not 0.0 <= scale <= 1.0:
            raise ValueError(
                "virtual_pd_curriculum_schedule scales must be in [0.0, 1.0]."
            )

    selected_scale = _interpolate_scalar_stage(schedule, env.common_step_counter)

    env.virtual_pd_assistance_scale = float(selected_scale)
    return torch.tensor(
        selected_scale,
        device=env.device,
        dtype=torch.float32,
    )


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


def missing_grasp_curriculum(
    env,
    env_ids: torch.Tensor,
) -> torch.Tensor:
    """
    Linear schedule for the missing-grasp penalty during trajectory lift.
    """
    del env_ids

    schedule = env.cfg.missing_grasp_curriculum_schedule
    _validate_schedule_start_and_order(
        schedule,
        "missing_grasp_curriculum_schedule",
    )

    for _, weight in schedule:
        if weight > 0.0:
            raise ValueError(
                "missing_grasp_curriculum_schedule weights must be <= 0.0."
            )

    selected_weight = _interpolate_scalar_stage(schedule, env.common_step_counter)

    term_cfg = env.dualarm_reward_manager.get_term_cfg("missing_grasp_during_lift")
    term_cfg.weight = float(selected_weight)

    return torch.tensor(
        selected_weight,
        device=env.device,
        dtype=torch.float32,
    )
