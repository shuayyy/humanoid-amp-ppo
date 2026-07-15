"""Frozen locomotion base policy for ResMimic-style residual learning.

Following ResMimic (Xie et al., 2025), loco-manipulation is learned as a
RESIDUAL on top of a frozen, pre-trained whole-body policy instead of
fine-tuning it: the base policy supplies balance and whole-body coordination,
while the task policy only learns small task-specific corrections. The final
action applied to the robot is::

    a = a_base(proprio) + residual_scale[joint] * a_residual(full task obs)

This module reconstructs the locomotion actor (MLP + observation normalizer)
directly from an RSL-RL checkpoint and replays the locomotion observation
pipeline exactly as the ``ObservationManager`` produced it during training:

* per-frame terms, in order: base_lin_vel(3), base_ang_vel(3)*0.25,
  projected_gravity(3), joint_pos_rel(29), joint_vel_rel(29)*0.05,
  last_base_action(29)  -> 96 dims;
* 5-frame history, term-major flattening (``[termA_t0..t4, termB_t0..t4, …]``,
  oldest -> newest) exactly like mjlab's CircularBuffer + flatten;
* history backfill on reset (first frame after a reset fills all slots);
* EmpiricalNormalization: ``(x - mean) / (std + 0.01)``.

The "actions" term is the base policy's OWN previous output (as during
locomotion training), not the combined residual action.
"""

from __future__ import annotations

from pathlib import Path

import torch
from mjlab.envs import mdp as env_mdp

# Matches EmpiricalNormalization(eps=1e-2) used at locomotion training time.
_NORMALIZER_EPS = 1.0e-2


class FrozenLocomotionPolicy:
    """Deterministic, frozen locomotion actor evaluated inside the dual-arm env."""

    def __init__(
        self,
        checkpoint_path: str,
        num_envs: int,
        device: str,
    ) -> None:
        path = Path(checkpoint_path)
        if not path.exists() and not path.is_absolute():
            # Config paths are repo-root relative; tolerate other CWDs.
            repo_root = Path(__file__).resolve().parents[3]
            if (repo_root / path).exists():
                path = repo_root / path
        if not path.exists():
            raise FileNotFoundError(
                f"Residual base checkpoint not found: {checkpoint_path}"
            )
        ckpt = torch.load(path, map_location=device, weights_only=False)
        actor_sd = ckpt.get("actor_state_dict")
        if not actor_sd:
            raise ValueError(f"No actor_state_dict in checkpoint: {checkpoint_path}")

        self.device = device
        self.num_envs = num_envs

        # Rebuild the MLP from the state dict (Linear layers at mlp.{0,2,4,...},
        # ELU in between, no activation after the output layer).
        layer_ids = sorted(
            int(k.split(".")[1]) for k in actor_sd if k.startswith("mlp.") and k.endswith(".weight")
        )
        self._weights = [actor_sd[f"mlp.{i}.weight"].to(device) for i in layer_ids]
        self._biases = [actor_sd[f"mlp.{i}.bias"].to(device) for i in layer_ids]

        self._norm_mean = actor_sd["obs_normalizer._mean"].to(device)
        self._norm_std = actor_sd["obs_normalizer._std"].to(device)

        self.obs_dim = self._weights[0].shape[1]
        self.num_actions = self._weights[-1].shape[0]
        assert self._norm_mean.shape[-1] == self.obs_dim, (
            f"normalizer dim {self._norm_mean.shape} != obs dim {self.obs_dim}"
        )

        # Locomotion frame layout: lin_vel(3) + ang_vel(3) + gravity(3)
        # + joint_pos(na) + joint_vel(na) + actions(na).
        self.frame_dim = 9 + 3 * self.num_actions
        if self.obs_dim % self.frame_dim != 0:
            raise ValueError(
                f"Locomotion obs dim {self.obs_dim} is not a multiple of the "
                f"per-frame dim {self.frame_dim}; observation layout mismatch."
            )
        self.history_length = self.obs_dim // self.frame_dim

        # Per-term slices of one frame, in policy-group order.
        na = self.num_actions
        bounds = [0, 3, 6, 9, 9 + na, 9 + 2 * na, 9 + 3 * na]
        self._term_slices = list(zip(bounds[:-1], bounds[1:], strict=True))

        # History of raw (already-scaled) frames, oldest -> newest.
        self._history = torch.zeros(
            num_envs, self.history_length, self.frame_dim, device=device
        )
        self._needs_backfill = torch.ones(num_envs, dtype=torch.bool, device=device)
        self.last_action = torch.zeros(num_envs, self.num_actions, device=device)

    @torch.no_grad()
    def reset(self, env_ids: torch.Tensor) -> None:
        """Clear per-env state; history is backfilled on the next update."""
        self.last_action[env_ids] = 0.0
        self._needs_backfill[env_ids] = True

    @torch.no_grad()
    def update(self, env) -> None:
        """Push the current (post-physics, post-reset) frame into the history."""
        frame = self._compute_frame(env)
        backfill = self._needs_backfill
        if backfill.any():
            self._history[backfill] = frame[backfill].unsqueeze(1)
            self._needs_backfill[backfill] = False
        roll = ~backfill
        if roll.any():
            self._history[roll, :-1] = self._history[roll, 1:].clone()
            self._history[roll, -1] = frame[roll]

    @torch.no_grad()
    def act(self, env) -> torch.Tensor:
        """Return the deterministic base action for the current history state."""
        if self._needs_backfill.any():
            # Lazy init: first act() after construction/reset before any update.
            self.update(env)

        obs = self._flatten_term_major()
        obs = (obs - self._norm_mean) / (self._norm_std + _NORMALIZER_EPS)

        x = obs
        for i in range(len(self._weights) - 1):
            x = torch.nn.functional.elu(
                torch.nn.functional.linear(x, self._weights[i], self._biases[i])
            )
        action = torch.nn.functional.linear(x, self._weights[-1], self._biases[-1])

        self.last_action.copy_(action)
        return action

    def _flatten_term_major(self) -> torch.Tensor:
        """[B, H, F] history -> [B, H*F] with term-major block ordering."""
        blocks = [
            self._history[:, :, s:e].reshape(self.num_envs, -1)
            for s, e in self._term_slices
        ]
        return torch.cat(blocks, dim=-1)

    def _compute_frame(self, env) -> torch.Tensor:
        """One 96-dim locomotion obs frame, scaled like the locomotion cfg."""
        base_lin_vel = env_mdp.builtin_sensor(env, sensor_name="robot/imu_lin_vel")
        base_ang_vel = env_mdp.builtin_sensor(env, sensor_name="robot/imu_ang_vel")
        projected_gravity = env_mdp.projected_gravity(env)
        joint_pos = env_mdp.joint_pos_rel(env)
        joint_vel = env_mdp.joint_vel_rel(env)

        return torch.cat(
            (
                base_lin_vel,
                base_ang_vel * 0.25,
                projected_gravity,
                joint_pos,
                joint_vel * 0.05,
                self.last_action,
            ),
            dim=-1,
        )
