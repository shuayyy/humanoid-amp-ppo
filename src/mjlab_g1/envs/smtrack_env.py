"""Shared tracking env for the SuperMimic vs PPO comparison (brief §2).

A stock mjlab `ManagerBasedRlEnv` plus the raw-state contract the SuperMimic learner needs:
exact generalized coordinates (`get_state`), the terminal state of envs that auto-reset
inside `step()` (`pre_reset_state`), the per-episode DR context (`get_xi`) and the
reference frame index (`get_ref_index`). PPO uses it as a plain mjlab env.

One deviation from mjlab's step: mjlab scores terminations and rewards on body poses and
velocities one physics substep stale (no forward() after the last mj_step). Here they are
refreshed first (kinematics + com_pos + com_vel, no collision), so r_{t+1} and d_{t+1} are
exact functions of s_{t+1}, which the learner's analytic reward reproduces (parity check).

Caveat (mjlab 1.4): a partial `reset(env_ids=...)` runs the command manager and the
observation history for ALL envs, so non-reset envs skip one reference frame and get a
duplicated history frame. Training relies on auto-reset inside `step()`, which is exact.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import mujoco_warp as mjwarp
import torch
import warp as wp

from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.envs import types as env_types
from mjlab_g1.tasks.smtrack.mdp.commands import SMMotionCommand
from mjlab_g1.tasks.smtrack.mdp.state import free_body_state, robot_qpos_qvel

if TYPE_CHECKING:
  from mjlab_g1.rl.supermimic.kinematics import TorchFK
  from mjlab_g1.rl.supermimic.reference import MotionLibrary


@dataclass(kw_only=True)
class SMTrackEnvCfg(ManagerBasedRlEnvCfg):
  xi_lo: tuple[float, ...] = ()
  """Lower DR bounds of xi, from the TRAINING DR ranges (kept unchanged in the play cfg)."""
  xi_hi: tuple[float, ...] = ()
  reward_ref_offset: int = 0
  """mjlab pairing: r_{t+1} uses the pre-advance frame, so the goal starts at ref_index."""
  foot_geom_expr: str = r"^(left|right)_foot[1-7]_collision$"
  com_body_name: str = "torso_link"
  box_entity_name: str = "box"
  """Only used when the scene has this entity (lift)."""
  fresh_reward_kinematics: bool = True
  """Refresh body kinematics before terminations/rewards (False = mjlab's stale pairing)."""


class SMTrackEnv(ManagerBasedRlEnv):
  cfg: SMTrackEnvCfg

  def load_managers(self) -> None:
    # Observation terms call get_state/get_xi while the managers are built (shape probe).
    self._init_contract()
    super().load_managers()
    self.pre_reset_state = self.get_state().clone()
    if self.cfg.fresh_reward_kinematics:
      # step() calls termination_manager.compute() first after the physics substeps.
      compute = self.termination_manager.compute

      def compute_on_fresh_kinematics() -> torch.Tensor:
        self.refresh_kinematics()
        return compute()

      self.termination_manager.compute = compute_on_fresh_kinematics  # type: ignore[method-assign]

  def _init_contract(self) -> None:
    robot = self.scene["robot"]
    ix = robot.indexing
    self.has_box = self.cfg.box_entity_name in self.scene.entities
    self.state_dim = 71 + (13 if self.has_box else 0)
    foot_local = robot.find_geoms(self.cfg.foot_geom_expr)[0]
    # foot_friction uses shared_random: every foot geom carries the same value.
    self._foot_gid = int(ix.geom_ids[foot_local[0]])
    self._com_bid = int(ix.body_ids[robot.find_bodies(self.cfg.com_body_name)[0][0]])
    if self.has_box:
      box = self.scene[self.cfg.box_entity_name]
      self._box_bid = int(box.indexing.body_ids[0])
      self._box_gid = int(box.indexing.geom_ids[0])
    xi_dim = 33 + (2 if self.has_box else 0)
    if len(self.cfg.xi_lo) != xi_dim or len(self.cfg.xi_hi) != xi_dim:
      raise ValueError(f"cfg.xi_lo/xi_hi must have {xi_dim} entries")
    self.xi_lo = torch.tensor(self.cfg.xi_lo, dtype=torch.float32, device=self.device)
    self.xi_hi = torch.tensor(self.cfg.xi_hi, dtype=torch.float32, device=self.device)
    n = self.num_envs
    self._captured = torch.zeros(n, self.state_dim, device=self.device)
    self._reset_in_step = torch.zeros(n, dtype=torch.bool, device=self.device)
    self._fk: TorchFK | None = None
    self._motion_lib: MotionLibrary | None = None

  # Contract.

  @property
  def env_origins(self) -> torch.Tensor:
    return self.scene.env_origins

  @property
  def motion_command(self) -> SMMotionCommand:
    return cast(SMMotionCommand, self.command_manager.get_term("motion"))

  @property
  def motion_file(self) -> str:
    return self.cfg.commands["motion"].motion_file  # type: ignore[attr-defined]

  @property
  def motion_lib(self) -> MotionLibrary:
    """The learner's MotionLibrary on the same npz (built on first use)."""
    if self._motion_lib is None:
      from mjlab_g1.rl.supermimic.reference import MotionLibrary

      self._motion_lib = MotionLibrary(self.motion_file, self.device)
    return self._motion_lib

  @property
  def torch_fk(self) -> TorchFK:
    if self._fk is None:
      from mjlab_g1.rl.supermimic.kinematics import TorchFK

      self._fk = TorchFK.g1(device=self.device)
    return self._fk

  def get_state(self, env_ids: torch.Tensor | None = None) -> torch.Tensor:
    """[N,71|84] = cat(qpos, qvel[, box_qpos, box_qvel]) from raw sim.data (brief §1)."""
    qpos, qvel = robot_qpos_qvel(self, self.scene["robot"], env_ids)
    parts = [qpos, qvel]
    if self.has_box:
      parts.append(free_body_state(self, self.scene[self.cfg.box_entity_name], env_ids))
    return torch.cat(parts, dim=-1)

  def get_xi(self) -> torch.Tensor:
    """[N,33|35] raw DR context: [foot friction, encoder bias 29, torso CoM offset 3
    (, box mass scale, box friction scale)]."""
    m = self.sim.model
    n = self.num_envs

    def per_env(x: torch.Tensor) -> torch.Tensor:
      return x.expand(n, *x.shape[1:]) if x.shape[0] != n else x

    com0 = self.sim.get_default_field("body_ipos")[self._com_bid]
    parts = [
      per_env(m.geom_friction[:, self._foot_gid, 0:1]),
      self.scene["robot"].data.encoder_bias,
      per_env(m.body_ipos[:, self._com_bid]) - com0,
    ]
    if self.has_box:
      mass0 = self.sim.get_default_field("body_mass")[self._box_bid]
      fric0 = self.sim.get_default_field("geom_friction")[self._box_gid, 0]
      parts.append(per_env(m.body_mass[:, self._box_bid : self._box_bid + 1]) / mass0)
      parts.append(per_env(m.geom_friction[:, self._box_gid, 0:1]) / fric0)
    return torch.cat(parts, dim=-1).float()

  def get_ref_index(self) -> torch.Tensor:
    """[N] frame that the reward of the NEXT step is scored against (pre-advance)."""
    return self.motion_command.time_steps.clone()

  @property
  def reward_term_names(self) -> list[str]:
    return list(self.reward_manager.active_terms)

  def get_reward_terms(self) -> torch.Tensor:
    """[N,T] weighted per-term rewards of the last step, dt-scaled (sum = reward_buf)."""
    # The manager stores the unscaled rate (value / dt) in _step_reward.
    scale = self.step_dt if self.cfg.scale_rewards_by_dt else 1.0
    return self.reward_manager._step_reward * scale

  def refresh_kinematics(self) -> None:
    """Recompute xpos/xquat/subtree_com/cvel from the current qpos/qvel."""
    sim = self.sim
    with wp.ScopedDevice(sim.wp_device):
      mjwarp.kinematics(sim.wp_model, sim.wp_data)
      mjwarp.com_pos(sim.wp_model, sim.wp_data)
      mjwarp.com_vel(sim.wp_model, sim.wp_data)

  # Overrides.

  def _reset_idx(self, env_ids: torch.Tensor | None = None) -> None:
    if env_ids is None:
      env_ids = torch.arange(self.num_envs, device=self.device)
    # Snapshot s_{t+1} before the reset overwrites it.
    self._captured[env_ids] = self.get_state(env_ids)
    self._reset_in_step[env_ids] = True
    super()._reset_idx(env_ids)

  def step(self, action: torch.Tensor) -> env_types.VecEnvStepReturn:
    self._reset_in_step.zero_()
    obs, rew, terminated, truncated, extras = super().step(action)
    # Non-reset envs still hold s_{t+1} (forward() does not touch qpos/qvel).
    self.pre_reset_state = torch.where(
      self._reset_in_step[:, None], self._captured, self.get_state()
    )
    # A failure on the last clip frame is a failure, not a truncation.
    truncated = truncated & ~terminated
    self.reset_time_outs = truncated
    return obs, rew, terminated, truncated, extras
