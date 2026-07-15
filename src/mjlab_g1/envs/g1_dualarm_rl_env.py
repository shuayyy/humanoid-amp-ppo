"""G1 dual-arm manipulation task environment."""
from dataclasses import dataclass, field

import mujoco
import torch

import warp as wp
from prettytable import PrettyTable
import math
from mjlab.envs import types
from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnvCfg
from mjlab.managers.reward_manager import RewardManager, RewardTermCfg
from mjlab.scene import Scene
from mjlab.sim.sim import Simulation
from mjlab.tasks.manipulation.mdp.commands import LiftingCommand
from mjlab.utils.logging import print_info
from mjlab.viewer.offscreen_renderer import OffscreenRenderer
from mjlab.utils.lab_api.math import quat_mul
from mjlab_g1.envs.amp_observations import g1_rich_amp_observations
from mjlab_g1.tasks.dualarm.virtual_object_pd import (
    VirtualObjectPdCfg,
    VirtualObjectPdController,
)
_DESIRED_FRAME_COLORS = ((1.0, 0.5, 0.5), (0.5, 1.0, 0.5), (0.5, 0.5, 1.0))

# dataclass auto-generates init/printing for config classes.
# kw_only=True forces fields to be passed by name, avoiding argument-order mistakes.

@dataclass(kw_only = True)
class G1DualarmManagerBasedRlEnvCfg(ManagerBasedRlEnvCfg):

    # ManagerBasedRlEnvCfg is the base configuration class for an RL environment.
    # It stores common settings like scene, actions, observations, rewards, and resets.
    # This task config inherits from it to reuse the standard MJLab environment setup.
    # Extra fields here are task-specific settings for the dual-arm manipulation environment.
    dualarm_rewards: dict[str, RewardTermCfg] = field (default_factory = dict)
    regularization_rewards: dict[str, RewardTermCfg] = field (default_factory = dict)

    # These fields store task-specific reward groups.
    # Each dictionary maps a reward name to its RewardTermCfg.
    # field(default_factory=dict/list) creates a fresh empty dict/list for every config object.
    # This avoids different config objects accidentally sharing the same mutable default.

    # Lift trajectory. The lift is CONTACT-TRIGGERED: the reference starts
    # rising only after both grasp-marker contacts have been held for
    # `lift_trigger_contact_steps` consecutive steps, then takes
    # (trajectory_end_s - trajectory_start_s) seconds. A clock-triggered lift
    # left un-grasped episodes unrecoverable: the reference departed at 1.5 s
    # whether or not the robot had grasped, with no path to re-grasp.
    trajectory_start_s: float = 1.5
    trajectory_end_s: float = 3.5
    # Set to 0 to restore the legacy CLOCK-triggered lift (reference rises at
    # trajectory_start_s unconditionally) — required to faithfully evaluate
    # checkpoints trained before the contact-triggered change.
    lift_trigger_contact_steps: int = 10
    trajectory_lift_delta_z: float = 0.625
    trajectory_position_tolerance: float = 0.115
    hold_steps: int = 20
    fall_angle_thresh: float = math.radians(70.0)
    virtual_pd_cfg: VirtualObjectPdCfg = field(default_factory=VirtualObjectPdCfg)

    # ResMimic-style residual learning: when set, a frozen locomotion actor is
    # loaded from this checkpoint and the RL policy's output becomes a residual
    # added on top of the base action:
    #   applied_action = base_action + residual_scale[joint] * policy_action
    # The base policy supplies balance/whole-body coordination; the task policy
    # only learns task-specific corrections. Residual authority is per joint
    # group: the legs stay mostly under base-policy control (protecting
    # balance) while the arms/waist get the freedom manipulation needs.
    residual_base_checkpoint: str | None = None
    residual_scale_legs: float = 0.1
    residual_scale_waist: float = 0.25
    residual_scale_arms: float = 0.5

    # Depth perception. Disabled by default: the policy already receives
    # privileged object state (pose, grasp markers, trajectory reference), so
    # depth features are redundant for the fixed-object lift while being by
    # far the largest GPU cost (per-env 224x224 renders + DeFM encoding; the
    # 16k-env OOM in job 2088161 was the encoder). Re-enable for vision-based
    # fine-tuning; the camera sensor and obs term are added back with this
    # flag in env_cfgs.py.
    use_depth: bool = False
    # Depth features are recomputed every N env steps and held in between
    # (~25 Hz effective at N=2, matching a real depth camera). Rendering +
    # DeFM encoding every step dominates wall-clock at large env counts.
    depth_update_interval: int = 2
    # DeFM forward is chunked to bound activation memory: a single 16k-image
    # batch needs ~25 GiB of activations and OOMs next to the sim. The
    # encoder is frozen, so chunking is exact (job 2088161 failure).
    depth_encode_chunk_size: int = 2048

    # Success-adaptive curricula (ResMimic-style): instead of blind step
    # schedules, assistance/bootstrap difficulty decays only when the policy
    # demonstrably succeeds (object at goal height AND both grasp contacts
    # held at episode end). Success is tracked as an EMA over finished
    # episodes with the horizon below (in episodes).
    success_ema_horizon: int = 200

    # Virtual-PD assistance decay: when the success EMA exceeds the decay
    # threshold, the assistance scale drops by `decay_step` at most once per
    # `decay_interval` env steps. Two-sided: if success collapses below the
    # recovery threshold (difficulty was raised too fast), the scale steps
    # back up instead of leaving the policy stranded.
    assistance_decay_threshold: float = 0.6
    assistance_recovery_threshold: float = 0.2
    assistance_decay_step: float = 0.05
    assistance_decay_interval: int = 250
    assistance_min_scale: float = 0.0
    # Below this scale each rung is disproportionately harder (the PD can no
    # longer hold the object alone near ~0.15), so decay slows down: smaller
    # steps, longer consolidation between them.
    assistance_fine_scale_threshold: float = 0.3
    assistance_fine_decay_step: float = 0.025
    assistance_fine_decay_interval: int = 500

    # Contact-bootstrap: fraction of trajectory_lift_delta_z that the object is
    # raised at reset (1.0 = spawns at goal height for a comfortable reach,
    # 0.0 = spawns on the ground). Decayed adaptively, but only after the
    # assistance scale has dropped below `reset_height_start_assistance`, so
    # the robot first learns to carry the object, then to reach lower for it.
    reset_height_decay_threshold: float = 0.6
    reset_height_decay_step: float = 0.05
    reset_height_decay_interval: int = 250
    reset_height_start_assistance: float = 0.5

    # Object spawn-pose randomization (robustness): at full difficulty the
    # toaster spawns with XY offset up to +/- object_spawn_xy_range and yaw
    # up to +/- object_spawn_yaw_range. Widened adaptively 0 -> 1 by the
    # spawn-range curriculum, and only after the reset-height bootstrap is
    # nearly done (frac <= spawn_range_start_height_below), so the policy
    # first learns the nominal grasp, then generalizes it.
    object_spawn_xy_range: float = 0.08
    object_spawn_yaw_range: float = math.radians(20.0)
    spawn_range_widen_threshold: float = 0.6
    spawn_range_widen_step: float = 0.1
    spawn_range_widen_interval: int = 250
    spawn_range_start_height_below: float = 0.25

    # Penalty on the virtual-PD force actually applied, normalized by
    # max_force. Ramped in as assistance decays (weight scales with
    # 1 - assistance_scale) so the bootstrap phase is not punished, but the
    # policy is increasingly paid to make the assistance unnecessary.
    assist_force_penalty_max_weight: float = -1.0

    # Missing-grasp penalty during the lift window, ramped in the same way.
    missing_grasp_max_weight: float = -0.5

    # Smoothness penalties, ramped in with (1 - assistance_scale) like the
    # force penalty. Both were previously absent, making the perpetual
    # stance-fidgeting during the hold completely free.
    action_rate_max_weight: float = -0.01
    joint_vel_max_weight: float = -1.0e-3

    feet_slip_curriculum_schedule: tuple[tuple[int, float, float], ...] = (
        (0, -0.5, 0.05),
        (12_000, -0.5, 0.05),
        (36_000, -2.0, 0.03),
    )


    """Whether in evaluation mode. If True, will save metrics to JSON and exit after all episodes complete."""
    eval_output_dir: str | None = None
    """Directory to save eval metrics JSON files. If None, saves to current directory."""

class G1DualarmManagerBasedRlEnv(ManagerBasedRlEnv):
    """Manager-based RL environment."""

    # Class-level metadata for the environment.
    # is_vector_env tells the RL code this env runs many parallel simulations.
    # metadata stores render/version information for logging and viewer support.
    # cfg is a type hint saying this env uses G1DualarmManagerBasedRlEnvCfg.

    is_vector_env = True
    metadata = {
        "render_modes": [None, "rgb_array"],
        "mujoco_version": mujoco.__version__,
        "warp_version": wp.config.version,
    }
    cfg: G1DualarmManagerBasedRlEnvCfg

    def __init__(
        self,
        cfg: G1DualarmManagerBasedRlEnvCfg,
        device: str,
        render_mode: str | None = None,
        **kwargs,
    ) -> None:
        
        # Initialize base environment state.
        self.cfg = cfg  # type: ignore[assignment]
        if self.cfg.seed is not None:
            self.cfg.seed = self.seed(self.cfg.seed)
        self._sim_step_counter = 0
        self.extras = {}
        self.obs_buf = {}

        """
        Scene/entity access.

        Build the MJLab scene and MuJoCo simulation, then expose task entities
        such as `robot` and `toaster` for observations, rewards, and resets.
        """
        self.scene = Scene(self.cfg.scene, device=device)
        self.sim = Simulation(
            num_envs=self.scene.num_envs,
            cfg=self.cfg.sim,
            model=self.scene.compile(),
            device=device,
        ) 
        

        self.scene.initialize(
            mj_model=self.sim.mj_model,
            model = self.sim.model,
            data = self.sim.data,
        )

        if self.cfg.use_depth and self.scene.sensor_context is None:
            raise RuntimeError(
                "No sensor context. Check that head_depth is in cfg.scene.sensors."
            )

        if self.scene.sensor_context is not None:
            self.sim.set_sensor_context(self.scene.sensor_context)
        

        # Print environment info.
        print_info("")
        table = PrettyTable()
        table.title = "Base Environment"
        table.field_names = ["Property", "Value"]
        table.align["Property"] = "l"
        table.align["Value"] = "l"
        table.add_row(["Number of environments", self.num_envs])
        table.add_row(["Environment device", self.device])
        table.add_row(["Environment seed", self.cfg.seed])
        table.add_row(["Physics step-size", self.physics_dt])
        table.add_row(["Environment step-size", self.step_dt])
        print_info(table.get_string())
        print_info("")


        self.robot = self.scene["robot"]
        self.object = self.scene["toaster"]
        self.toaster = self.object
        self._init_buffers()

        # Both curricula start at full assistance / full bootstrap and decay
        # adaptively with the lift-success EMA (see curriculums.py). In eval/
        # play mode no curriculum runs, so start at the FINAL difficulty:
        # zero assistance, ground spawn — otherwise play would show the PD
        # controller carrying a floating object.
        eval_mode = bool(getattr(self.cfg, "eval_mode", False))
        self.virtual_pd_assistance_scale = 0.0 if eval_mode else 1.0
        self.object_reset_height_frac = 0.0 if eval_mode else 1.0
        self.object_spawn_range_frac = 0.0
        self.lift_success_ema = 0.0
        self.last_assist_decay_step = 0
        self.last_height_decay_step = 0
        self.last_spawn_widen_step = 0

        self.virtual_pd_controller = VirtualObjectPdController(
            toaster=self.toaster,
            object_body_ids=self.object_body_ids,
            grasp_site_ids=self.grasp_site_ids,
            cfg=self.cfg.virtual_pd_cfg,
        )

        # ResMimic-style residual learning: frozen locomotion base policy.
        self.base_policy = None
        self.residual_scale = None
        if self.cfg.residual_base_checkpoint is not None:
            from mjlab_g1.rl.residual_base_policy import FrozenLocomotionPolicy

            self.base_policy = FrozenLocomotionPolicy(
                checkpoint_path=self.cfg.residual_base_checkpoint,
                num_envs=self.num_envs,
                device=self.device,
            )
            self.residual_scale = self._build_residual_scale(
                self.base_policy.num_actions
            )
            print_info(
                "[INFO]: Residual mode: frozen locomotion base policy loaded "
                f"from {self.cfg.residual_base_checkpoint} (residual scale "
                f"legs {self.cfg.residual_scale_legs}, "
                f"waist {self.cfg.residual_scale_waist}, "
                f"arms {self.cfg.residual_scale_arms})."
            )

        self.common_step_counter = 0
        self.episode_length_buf = torch.zeros(
            cfg.scene.num_envs, device=self.device, dtype=torch.long
        )
        self._manual_reset_pending = torch.zeros(
            cfg.scene.num_envs, device=self.device, dtype=torch.bool
        )
        self.render_mode = render_mode
        self._offline_renderer: OffscreenRenderer | None = None
        if self.render_mode == "rgb_array":
            renderer = OffscreenRenderer(
                model=self.sim.mj_model, cfg=self.cfg.viewer, scene=self.scene
            )
            renderer.initialize()
            self._offline_renderer = renderer
        self.metadata["render_fps"] = 1.0 / self.step_dt  # type: ignore

        #Load all managers
        self.load_managers()
        self._sync_place_pos_command_xy()
        self.setup_manager_visualizers()
   
  
    def _build_residual_scale(self, num_actions: int) -> torch.Tensor:
        """Per-joint residual authority: legs low, waist medium, arms high.

        Actions map to actuators in joint order for this robot (29 matching
        actuators/joints), so joint indices index the action vector directly.
        """
        joint_names = self.robot.joint_names
        assert len(joint_names) == num_actions, (
            f"Expected identical joint/action ordering "
            f"({len(joint_names)} joints vs {num_actions} actions)."
        )
        scale = torch.full(
            (num_actions,), self.cfg.residual_scale_arms, device=self.device
        )
        leg_ids, _ = self.robot.find_joints(
            (r".*hip.*", r".*knee.*", r".*ankle.*"), preserve_order=True
        )
        waist_ids, _ = self.robot.find_joints((r".*waist.*",), preserve_order=True)
        assert len(leg_ids) > 0 and len(waist_ids) > 0
        scale[leg_ids] = self.cfg.residual_scale_legs
        scale[waist_ids] = self.cfg.residual_scale_waist
        return scale

    def _init_buffers(self):
        """
        Buffers.

        Allocate per-environment tensors used across steps, rewards,
        terminations, and success holding.
        """
        self._init_ids_buffers()
        self.success_hold_buf = torch.zeros(
            self.num_envs,
            dtype=torch.long,
            device=self.device,
            requires_grad=False,
        )
        # Mean fraction of the virtual-PD force cap used during the last
        # control step (averaged over decimation substeps). Consumed by the
        # `virtual_assistance_force` reward term.
        self.assist_force_frac = torch.zeros(self.num_envs, device=self.device)
        self._assist_force_accum = torch.zeros(self.num_envs, device=self.device)
        # Contact-triggered lift state: episode step at which the lift began
        # (-1 = not started) and consecutive bilateral-contact step counter.
        self.lift_start_step = torch.full(
            (self.num_envs,), -1, dtype=torch.long, device=self.device
        )
        self.contact_settle_buf = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.trajectory_start_pos_w = (
            self.toaster.data.default_root_state[:, :3].clone()
            + self.scene.env_origins
        )

        self.trajectory_goal_pos_w = self.trajectory_start_pos_w.clone()
        self.trajectory_goal_pos_w[:, 2] += self.cfg.trajectory_lift_delta_z

        self.depth_encoder = None
        self.depth_feature_buf = None
        if self.cfg.use_depth:
            from mjlab_g1.perception.encoders import DeFMEncoder

            self.depth_encoder = DeFMEncoder(device=self.device)

            self.depth_feature_buf = torch.zeros(
                self.num_envs,
                self.depth_encoder.output_dim,
                device=self.device,
                dtype=torch.float32,
            )

    def update_depth_features(self) -> None:
        """Render batched depth and encode one frozen 192-D feature per environment."""
        if not self.cfg.use_depth:
            return
        assert self.depth_encoder is not None
        assert self.depth_feature_buf is not None
        self.sim.sense()

        depth = self.scene["head_depth"].data.depth
        assert depth is not None
        assert depth.shape == (
            self.num_envs,
            224,
            224,
            1,
        ), depth.shape
        assert torch.isfinite(depth).all()

        # Chunked forward: bounds encoder activation memory at large env
        # counts (frozen encoder => chunking is exact).
        chunk = max(int(self.cfg.depth_encode_chunk_size), 1)
        for start in range(0, self.num_envs, chunk):
            end = min(start + chunk, self.num_envs)
            self.depth_feature_buf[start:end] = self.depth_encoder(
                depth[start:end]
            )

        assert torch.isfinite(self.depth_feature_buf).all()

    def _init_ids_buffers(self):
        """
        Site/body ID lookup.

        Cache MuJoCo body and site ids once so observation and reward functions
        can gather hand, foot, and grasp-marker state efficiently every step.
        """
        self.left_hand_body_id, _ = self.robot.find_bodies(
            name_keys=["left_wrist_yaw_link"],
            preserve_order=True,
        )
        self.right_hand_body_id, _ = self.robot.find_bodies(
            name_keys=["right_wrist_yaw_link"],
            preserve_order=True,
        )
        self.feet_body_ids, _ = self.robot.find_bodies(
            name_keys=["left_ankle_roll_link", "right_ankle_roll_link"],
            preserve_order=True,
        )
        self.torso_body_id, _ = self.robot.find_bodies(
            name_keys=["torso_link"],
            preserve_order=True,
        )
        self.waist_joint_ids, _ = self.robot.find_joints(
            (r".*waist.*",), preserve_order=True
        )
        self.left_leg_sym_joint_ids, _ = self.robot.find_joints(
            ("left_hip_pitch_joint", "left_knee_joint"), preserve_order=True
        )
        self.right_leg_sym_joint_ids, _ = self.robot.find_joints(
            ("right_hip_pitch_joint", "right_knee_joint"), preserve_order=True
        )
        self.hand_site_ids, _ = self.robot.find_sites(
            name_keys=["left_palm", "right_palm"],
            preserve_order=True,
        )
        self.grasp_site_ids, _ = self.toaster.find_sites(
            name_keys=["left_grasp_marker", "right_grasp_marker"],
            preserve_order=True,
        )
        self.object_body_ids, _ = self.toaster.find_bodies(
            name_keys=["object"],
            preserve_order=True,
        )
        assert len(self.object_body_ids) == 1
        assert len(self.grasp_site_ids) == 2
        self.left_foot_site_ids, _ = self.robot.find_sites(name_keys=["left_foot_1", "left_foot_2", "left_foot_3", "left_foot_4"], preserve_order=True)
        self.right_foot_site_ids, _ = self.robot.find_sites(name_keys=["right_foot_1", "right_foot_2", "right_foot_3", "right_foot_4"], preserve_order=True)

    def load_managers(self) -> None:
        super().load_managers()

        self.dualarm_reward_manager = RewardManager(
            self.cfg.dualarm_rewards, self, scale_by_dt = self.cfg.scale_rewards_by_dt
        )
        print_info(f"[INFO]: {self.dualarm_reward_manager}")

        self.reg_reward_manager = RewardManager(
            self.cfg.regularization_rewards, self, scale_by_dt = self.cfg.scale_rewards_by_dt
        )
        print_info(f"[INFO]: {self.reg_reward_manager}")
    
    def step(self, action: torch.Tensor) -> types.VecEnvStepReturn:
        """Apply action, step the simulation, and return observations, rewards, dones, and infos."""
        action = action.to(self.device)
        if self.base_policy is not None:
            # ResMimic residual composition: the frozen locomotion actor
            # provides the base action; the RL policy contributes a per-joint
            # scaled task-specific correction (legs low, arms high).
            base_action = self.base_policy.act(self)
            action = base_action + self.residual_scale * action
        self.action_manager.process_action(action)
        q_before = self.robot.data.joint_pos.clone()

        for _ in range(self.cfg.decimation):
            """DECIMATION is a common technique in RL environments where the simulation runs at a higher frequency than the agent's action frequency. For example, if the simulation runs at 1000 Hz and the agent acts at 20 Hz, then decimation would be 50. This means that for each action taken by the agent, the simulation will step forward 50 times before the next action is applied. This allows for more realistic physics and smoother control while keeping the agent's decision-making at a manageable frequency."""
            # Number of physics/simulation substeps executed for each single RL/control step.
            # Higher decimation means actions are held constant for more simulator steps.

            self._sim_step_counter += 1
            self.action_manager.apply_action()
            self.scene.write_data_to_sim()
            self._apply_virtual_pd_assistance()
            self._assist_force_accum += self.virtual_pd_controller.last_force_norm

            self.sim.step()
            self.scene.update(dt=self.physics_dt)

        self.assist_force_frac = self._assist_force_accum / (
            self.cfg.decimation * max(self.cfg.virtual_pd_cfg.max_force, 1.0e-6)
        )
        self._assist_force_accum.zero_()

        self.episode_length_buf += 1
        self.common_step_counter += 1

        # Contact-triggered lift: latch the lift start once both marker
        # contacts have been held for the required settle time. (Skipped in
        # legacy clock-trigger mode, lift_trigger_contact_steps <= 0.)
        if self.cfg.lift_trigger_contact_steps > 0:
            contacts = self._both_marker_contacts()
            self.contact_settle_buf = torch.where(
                contacts,
                self.contact_settle_buf + 1,
                torch.zeros_like(self.contact_settle_buf),
            )
            lift_triggered = (self.lift_start_step < 0) & (
                self.contact_settle_buf >= self.cfg.lift_trigger_contact_steps
            )
            self.lift_start_step[lift_triggered] = self.episode_length_buf[
                lift_triggered
            ]

        self.curriculum_manager.compute()

        self.reset_buf = self.termination_manager.compute()
        self.reset_terminated = self.termination_manager.terminated
        self.reset_time_outs = self.termination_manager.time_outs

        self._sync_place_pos_command_xy()
        dualarm_reward_buf = self.dualarm_reward_manager.compute(self.step_dt)
        reg_reward_buf = self.reg_reward_manager.compute(self.step_dt)

        self.reward_buf = dualarm_reward_buf + reg_reward_buf

        # Sustained-hold success tracking: count consecutive steps with the
        # object at goal height AND both grasp contacts active. The adaptive
        # curricula only count an episode as a success if this hold lasted
        # `hold_steps` — a lucky terminal frame no longer moves the EMA.
        holding = self._object_lifted() & self._both_marker_contacts()
        self.success_hold_buf = torch.where(
            holding,
            self.success_hold_buf + 1,
            torch.zeros_like(self.success_hold_buf),
        )

        self.reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(self.reset_env_ids) > 0:
            self._reset_idx(self.reset_env_ids)
            self.scene.write_data_to_sim()
            self.sim.forward()

        self.command_manager.compute(dt = self.step_dt)
        self._sync_place_pos_command_xy()

        if "interval" in self.event_manager.available_modes:
            self.event_manager.apply(mode="interval", dt=self.step_dt)

        if self.base_policy is not None:
            # Push the post-physics (and post-reset) frame so the next act()
            # sees the same state the RL policy observes below.
            self.base_policy.update(self)

        # Amortize depth rendering + DeFM encoding: refresh every N steps
        # (counter was incremented above, so `- 1` makes the first step of
        # training/resume a refresh step), hold the feature buffer otherwise.
        if (
            self.cfg.use_depth
            and (self.common_step_counter - 1)
            % max(self.cfg.depth_update_interval, 1)
            == 0
        ):
            self.update_depth_features()
        self.obs_buf = self.observation_manager.compute(update_history=True)
        return(
            self.obs_buf,
            self.reward_buf,
            self.reset_terminated,
            self.reset_time_outs,
            self.extras,
        )
    
    def _reset_idx(self, env_ids: torch.Tensor | None = None) -> None:
        """
        Reset logic.

        Restore robot/object root states, reset task reward managers, and clear
        per-episode buffers for the environments that just terminated.
        """
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)

        # Update the lift-success EMA from the episodes that are about to
        # reset. Success = object held at goal (with both grasp contacts) for
        # at least `hold_steps` consecutive steps; drives the adaptive
        # assistance/bootstrap curricula.
        success = self.success_hold_buf[env_ids] >= self.cfg.hold_steps
        n = int(env_ids.numel())
        if n > 0:
            alpha = n / (n + max(self.cfg.success_ema_horizon, 1))
            self.lift_success_ema += alpha * (
                float(success.float().mean()) - self.lift_success_ema
            )

        super()._reset_idx(env_ids)

        env_origins = self.scene.env_origins[env_ids]

        robot_root_state = self.robot.data.default_root_state[env_ids].clone()
        robot_root_state[:, :3] += env_origins
        self.robot.write_root_state_to_sim(robot_root_state, env_ids)

        toaster_root_state = self.toaster.data.default_root_state[env_ids].clone()
        toaster_root_state[:, :3] += env_origins

        # Spawn-pose randomization, widened 0 -> 1 by the adaptive curriculum:
        # random XY offset and world-frame yaw so the policy generalizes the
        # grasp instead of memorizing one reach trajectory.
        frac = self.object_spawn_range_frac
        if frac > 0.0:
            n = env_ids.numel()
            xy_offset = (
                torch.rand(n, 2, device=self.device) * 2.0 - 1.0
            ) * (frac * self.cfg.object_spawn_xy_range)
            toaster_root_state[:, :2] += xy_offset

            half_yaw = 0.5 * (
                torch.rand(n, device=self.device) * 2.0 - 1.0
            ) * (frac * self.cfg.object_spawn_yaw_range)
            zeros = torch.zeros_like(half_yaw)
            yaw_quat = torch.stack(
                (torch.cos(half_yaw), zeros, zeros, torch.sin(half_yaw)), dim=-1
            )
            toaster_root_state[:, 3:7] = quat_mul(
                yaw_quat, toaster_root_state[:, 3:7]
            )

        # Fixed goal height (ground spawn + full lift), independent of the
        # bootstrap. The object is raised toward this goal at reset by the
        # contact-bootstrap curriculum so early episodes need no deep reach.
        goal_z = toaster_root_state[:, 2] + self.cfg.trajectory_lift_delta_z
        bootstrap_dz = self.object_reset_height_frac * self.cfg.trajectory_lift_delta_z
        toaster_root_state[:, 2] += bootstrap_dz

        self.toaster.write_root_state_to_sim(toaster_root_state, env_ids)
        assert self.virtual_pd_controller is not None
        self.virtual_pd_controller.reset(
            env_ids=env_ids,
            reference_quat_w=toaster_root_state[:, 3:7],
        )
        self.virtual_pd_controller.clear(env_ids)
        self.trajectory_start_pos_w[env_ids] = toaster_root_state[:, :3]

        self.trajectory_goal_pos_w[env_ids] = toaster_root_state[:, :3]
        self.trajectory_goal_pos_w[env_ids, 2] = goal_z
        self._sync_place_pos_command_xy(env_ids)

        info = self.dualarm_reward_manager.reset(env_ids)
        self.extras["log"].update(info)
        info = self.reg_reward_manager.reset(env_ids)
        self.extras["log"].update(info)

        self.success_hold_buf[env_ids] = 0
        self.lift_start_step[env_ids] = -1
        self.contact_settle_buf[env_ids] = 0
        if self.base_policy is not None:
            self.base_policy.reset(env_ids)

    def _get_hand_toaster_dis(self):
        """
        Helper methods.

        Return vector differences from each hand palm site to its corresponding
        toaster grasp-marker site.
        """
        hand_pos = self.robot.data.site_pos_w[:, self.hand_site_ids, :3]
        grasp_pos = self.toaster.data.site_pos_w[:, self.grasp_site_ids, :3]

        return grasp_pos - hand_pos

    def get_object_trajectory_reference(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Analytic reference for:
          hold at initial toaster position
          -> smooth vertical lift (contact-triggered)
          -> hold at final position.

        The lift phase starts per-env when ``lift_start_step`` is latched
        (bilateral grasp held for the settle time), not at a fixed clock time,
        so an episode with a late grasp still gets a full lift attempt.

        Returns:
          reference_position_w: [num_envs, 3]
          reference_velocity_w: [num_envs, 3]
        """
        duration_s = self.cfg.trajectory_end_s - self.cfg.trajectory_start_s

        if duration_s <= 0.0:
            raise ValueError(
                f"trajectory_end_s ({self.cfg.trajectory_end_s}) must be larger "
                f"than trajectory_start_s ({self.cfg.trajectory_start_s})."
            )

        started, steps_since_lift = self._lift_progress()
        time_since_lift_s = steps_since_lift.float() * self.step_dt

        # u = 0 before the lift is triggered; u = 1 after it finishes.
        u = (time_since_lift_s / duration_s).clamp(0.0, 1.0) * started.float()

        # Smoothstep: zero velocity at trajectory start and end.
        smooth_u = 3.0 * u.square() - 2.0 * u.pow(3)
        smooth_du_dt = (6.0 * u - 6.0 * u.square()) / duration_s

        delta_pos_w = self.trajectory_goal_pos_w - self.trajectory_start_pos_w

        reference_position_w = (
            self.trajectory_start_pos_w
            + smooth_u.unsqueeze(-1) * delta_pos_w
        )

        reference_velocity_w = (
            smooth_du_dt.unsqueeze(-1) * delta_pos_w
        )

        return reference_position_w, reference_velocity_w

    def _apply_virtual_pd_assistance(self) -> None:
        """
        Physics-substep virtual-PD hook.
        """
        assert self.virtual_pd_controller is not None

        reference_pos_w, reference_vel_w = self.get_object_trajectory_reference()

        assistance_scale = torch.full(
            (self.num_envs,),
            self.virtual_pd_assistance_scale,
            device=self.device,
            dtype=reference_pos_w.dtype,
        )

        self.virtual_pd_controller.apply(
            reference_pos_w,
            reference_vel_w,
            assistance_scale,
        )

    def _sync_place_pos_command_xy(
        self, env_ids: torch.Tensor | None = None
    ) -> None:
        """Use the toaster's current world XY and keep the command's configured Z."""
        command = self.command_manager.get_term("place_pos")
        if not isinstance(command, LiftingCommand):
            return

        if env_ids is None:
            command.target_pos[:, :2] = self.toaster.data.root_link_pos_w[:, :2]
        else:
            command.target_pos[env_ids, :2] = self.toaster.data.root_link_pos_w[
                env_ids, :2
            ]

    def _lift_progress(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (lift_started, steps_since_lift_start) per env.

        Contact-triggered by default; falls back to the legacy clock trigger
        when ``lift_trigger_contact_steps <= 0`` (for evaluating checkpoints
        trained before the contact-trigger change).
        """
        if self.cfg.lift_trigger_contact_steps > 0:
            started = self.lift_start_step >= 0
            steps_since = (self.episode_length_buf - self.lift_start_step).clamp(
                min=0
            )
        else:
            start_steps = int(round(self.cfg.trajectory_start_s / self.step_dt))
            started = self.episode_length_buf >= start_steps
            steps_since = (self.episode_length_buf - start_steps).clamp(min=0)
        return started, steps_since

    def lift_phase_active(self) -> torch.Tensor:
        """Whether the lift is currently in motion."""
        duration_steps = int(
            (self.cfg.trajectory_end_s - self.cfg.trajectory_start_s)
            / self.step_dt
        )
        started, steps_since_lift = self._lift_progress()
        return started & (steps_since_lift <= duration_steps)

    def _both_marker_contacts(self) -> torch.Tensor:
        """Return whether both hand/grasp-marker contact sensors are active."""
        left = self.scene["left_hand_toaster_contact"].data.found
        right = self.scene["right_hand_toaster_contact"].data.found
        if left is None or right is None:
            # Sensors not populated yet (initial reset before first sim step).
            return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        return torch.any(left > 0, dim=-1) & torch.any(right > 0, dim=-1)

    def _object_lifted(self) -> torch.Tensor:
        """Return whether toaster reached the final trajectory lift height."""
        toaster_z = self.toaster.data.root_link_pos_w[:, 2]
        required_z = (
            self.trajectory_goal_pos_w[:, 2]
            - self.cfg.trajectory_position_tolerance
        )
        return toaster_z >= required_z

    def get_amp_observations(self):
        return g1_rich_amp_observations(self.robot, self.num_envs, self.device)
