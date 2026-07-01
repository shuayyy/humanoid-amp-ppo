from __future__ import annotations

from dataclasses import dataclass
import torch
from mjlab.utils.lab_api.math import axis_angle_from_quat, quat_conjugate, quat_mul


@dataclass
class VirtualObjectPdCfg:
    enabled: bool = True

    # Static multiplier. Keep this at 1.0.
    scale: float = 1.0

    # Verified translational PD values.
    kp_pos: float = 800.0
    kd_pos: float = 50.0
    max_force: float = 80.0

    # Verified orientation stabilization values.
    kp_rot: float = 4.0
    kd_rot: float = 0.75
    max_torque: float = 4.0


class VirtualObjectPdController:
    """
    Object-level virtual PD assistance for the toaster lift.

    The controller applies one equivalent world-frame wrench to the toaster
    body. The force is computed as a translational PD term, split into equal
    virtual forces at the two grasp-marker sites, then converted into the
    equivalent off-COM torque. A reset-orientation PD term and angular damping
    stabilize rotation.
    """

    def __init__(
        self,
        toaster,
        object_body_ids,
        grasp_site_ids,
        cfg: VirtualObjectPdCfg,
    ) -> None:
        self.toaster = toaster
        self.object_body_ids = object_body_ids
        self.grasp_site_ids = grasp_site_ids
        self.cfg = cfg
        self.reference_quat_w = toaster.data.root_link_quat_w.clone()

        assert len(object_body_ids) == 1
        assert len(grasp_site_ids) == 2

    @torch.no_grad()
    def clear(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            num_envs = self.toaster.data.root_link_pos_w.shape[0]
        else:
            num_envs = env_ids.shape[0]

        zeros = torch.zeros(
            (num_envs, 1, 3),
            device=self.toaster.data.root_link_pos_w.device,
            dtype=self.toaster.data.root_link_pos_w.dtype,
        )
        self.toaster.write_external_wrench_to_sim(
            forces=zeros,
            torques=zeros,
            body_ids=self.object_body_ids,
            env_ids=env_ids,
        )

    @torch.no_grad()
    def reset(
        self,
        env_ids: torch.Tensor,
        reference_quat_w: torch.Tensor,
    ) -> None:
        assert reference_quat_w.shape == (env_ids.shape[0], 4), reference_quat_w.shape
        self.reference_quat_w[env_ids] = reference_quat_w

    @torch.no_grad()
    def apply(
        self,
        reference_pos_w: torch.Tensor,
        reference_vel_w: torch.Tensor,
        assistance_scale: torch.Tensor,
    ) -> None:
        assert reference_pos_w.shape == reference_vel_w.shape
        assert reference_pos_w.shape[-1] == 3
        assert assistance_scale.shape == (reference_pos_w.shape[0],), assistance_scale.shape

        if not self.cfg.enabled or self.cfg.scale <= 0.0:
            self.clear()
            return

        total_scale = self.cfg.scale * assistance_scale.unsqueeze(-1)

        p_obj_w = self.toaster.data.root_link_pos_w[:, :3]
        q_obj_w = self.toaster.data.root_link_quat_w
        v_obj_w = self.toaster.data.root_link_lin_vel_w[:, :3]
        omega_obj_w = self.toaster.data.root_link_ang_vel_w[:, :3]

        pos_error_w = reference_pos_w - p_obj_w
        vel_error_w = reference_vel_w - v_obj_w

        f_net_w = total_scale * (
            self.cfg.kp_pos * pos_error_w
            + self.cfg.kd_pos * vel_error_w
        )

        f_norm = torch.linalg.vector_norm(f_net_w, dim=-1, keepdim=True)
        force_clip_scale = torch.clamp(
            self.cfg.max_force / f_norm.clamp_min(1.0e-6),
            max=1.0,
        )
        f_net_w = f_net_w * force_clip_scale

        f_left_w = 0.5 * f_net_w
        f_right_w = 0.5 * f_net_w

        marker_pos_w = self.toaster.data.site_pos_w[:, self.grasp_site_ids, :3]
        left_marker_pos_w = marker_pos_w[:, 0, :]
        right_marker_pos_w = marker_pos_w[:, 1, :]

        object_com_pos_w = self.toaster.data.body_com_pos_w[
            :, self.object_body_ids, :3
        ].squeeze(1)

        r_left_w = left_marker_pos_w - object_com_pos_w
        r_right_w = right_marker_pos_w - object_com_pos_w

        q_err_w = quat_mul(self.reference_quat_w, quat_conjugate(q_obj_w))
        orientation_error_w = axis_angle_from_quat(q_err_w)

        tau_marker_w = (
            torch.cross(r_left_w, f_left_w, dim=-1)
            + torch.cross(r_right_w, f_right_w, dim=-1)
        )
        tau_orientation_w = total_scale * self.cfg.kp_rot * orientation_error_w
        tau_damping_w = -total_scale * self.cfg.kd_rot * omega_obj_w
        tau_w = tau_marker_w + tau_orientation_w + tau_damping_w

        tau_norm = torch.linalg.vector_norm(tau_w, dim=-1, keepdim=True)
        torque_clip_scale = torch.clamp(
            self.cfg.max_torque / tau_norm.clamp_min(1.0e-6),
            max=1.0,
        )
        tau_w = tau_w * torque_clip_scale

        self.toaster.write_external_wrench_to_sim(
            forces=f_net_w[:, None, :],
            torques=tau_w[:, None, :],
            body_ids=self.object_body_ids,
        )
