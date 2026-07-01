from __future__ import annotations

import torch


def virtual_pd_assistance_curriculum(
    env,
    env_ids: torch.Tensor,
) -> torch.Tensor:
    """
    Global common-step schedule for virtual-PD assistance.

    The same scale is shared by all environments. Per-environment activation is
    handled separately by bilateral grasp-marker contact gating in the env.
    """
    del env_ids

    schedule = env.cfg.virtual_pd_curriculum_schedule
    if len(schedule) == 0:
        raise ValueError("virtual_pd_curriculum_schedule must not be empty.")
    if schedule[0][0] != 0:
        raise ValueError("virtual_pd_curriculum_schedule first stage must start at 0.")

    previous_step = -1
    selected_scale = schedule[0][1]
    for start_step, scale in schedule:
        if start_step <= previous_step:
            raise ValueError(
                "virtual_pd_curriculum_schedule steps must be strictly ascending."
            )
        if not 0.0 <= scale <= 1.0:
            raise ValueError(
                "virtual_pd_curriculum_schedule scales must be in [0.0, 1.0]."
            )

        previous_step = start_step
        if env.common_step_counter >= start_step:
            selected_scale = scale
        else:
            break

    env.virtual_pd_assistance_scale = float(selected_scale)
    return torch.tensor(
        selected_scale,
        device=env.device,
        dtype=torch.float32,
    )
