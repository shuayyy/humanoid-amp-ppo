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

    trajectory_start_s: float = 1.5
    trajectory_end_s: float = 3.5
    trajectory_lift_delta_z: float = 0.625
    trajectory_position_tolerance: float = 0.115
    hold_steps: int = 20
    fall_angle_thresh: float = math.radians(70.0)
    virtual_pd_cfg: VirtualObjectPdCfg = field(default_factory=VirtualObjectPdCfg)
    virtual_pd_curriculum_schedule: tuple[tuple[int, float], ...] = (
        (0, 1.0),
        (350_000, 0.75),
        (550_000, 0.50),
        (750_000, 0.25),
        (950_000, 0.0),
    )
    feet_slip_curriculum_schedule: tuple[tuple[int, float, float], ...] = (
        (0, -0.5, 0.05),
        (300_000, -1.0, 0.04),
        (600_000, -2.0, 0.03),
    )
    missing_grasp_curriculum_schedule: tuple[tuple[int, float], ...] = (
        (0, 0.0),
        (250_000, -0.25),
        (500_000, -0.5),
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

        if self.scene.sensor_context is None:
            raise RuntimeError(
                "No sensor context. Check that head_depth is in cfg.scene.sensors."
            )

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
        self.virtual_pd_assistance_scale = float(
            self.cfg.virtual_pd_curriculum_schedule[0][1]
        )

        self.virtual_pd_controller = VirtualObjectPdController(
            toaster=self.toaster,
            object_body_ids=self.object_body_ids,
            grasp_site_ids=self.grasp_site_ids,
            cfg=self.cfg.virtual_pd_cfg,
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
        self.trajectory_start_pos_w = (
            self.toaster.data.default_root_state[:, :3].clone()
            + self.scene.env_origins
        )

        self.trajectory_goal_pos_w = self.trajectory_start_pos_w.clone()
        self.trajectory_goal_pos_w[:, 2] += self.cfg.trajectory_lift_delta_z

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

        features = self.depth_encoder(depth)

        assert features.shape == (
            self.num_envs,
            self.depth_encoder.output_dim,
        ), features.shape
        assert torch.isfinite(features).all()

        self.depth_feature_buf.copy_(features)

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
        self.action_manager.process_action(action.to(self.device))
        q_before = self.robot.data.joint_pos.clone()

        for _ in range(self.cfg.decimation):
            """DECIMATION is a common technique in RL environments where the simulation runs at a higher frequency than the agent's action frequency. For example, if the simulation runs at 1000 Hz and the agent acts at 20 Hz, then decimation would be 50. This means that for each action taken by the agent, the simulation will step forward 50 times before the next action is applied. This allows for more realistic physics and smoother control while keeping the agent's decision-making at a manageable frequency."""
            # Number of physics/simulation substeps executed for each single RL/control step.
            # Higher decimation means actions are held constant for more simulator steps.

            self._sim_step_counter += 1
            self.action_manager.apply_action()
            self.scene.write_data_to_sim()
            self._apply_virtual_pd_assistance()
        
            self.sim.step()
            self.scene.update(dt=self.physics_dt)

        self.episode_length_buf += 1
        self.common_step_counter += 1
        self.curriculum_manager.compute()

        self.reset_buf = self.termination_manager.compute()
        self.reset_terminated = self.termination_manager.terminated
        self.reset_time_outs = self.termination_manager.time_outs

        self._sync_place_pos_command_xy()
        dualarm_reward_buf = self.dualarm_reward_manager.compute(self.step_dt)
        reg_reward_buf = self.reg_reward_manager.compute(self.step_dt)

        self.reward_buf = dualarm_reward_buf + reg_reward_buf
        self.reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(self.reset_env_ids) > 0:
            self._reset_idx(self.reset_env_ids)
            self.scene.write_data_to_sim()
            self.sim.forward()

        self.command_manager.compute(dt = self.step_dt)
        self._sync_place_pos_command_xy()

        if "interval" in self.event_manager.available_modes:
            self.event_manager.apply(mode="interval", dt=self.step_dt)

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

        super()._reset_idx(env_ids)

        env_origins = self.scene.env_origins[env_ids]

        robot_root_state = self.robot.data.default_root_state[env_ids].clone()
        robot_root_state[:, :3] += env_origins
        self.robot.write_root_state_to_sim(robot_root_state, env_ids)

        toaster_root_state = self.toaster.data.default_root_state[env_ids].clone()
        toaster_root_state[:, :3] += env_origins
        self.toaster.write_root_state_to_sim(toaster_root_state, env_ids)
        assert self.virtual_pd_controller is not None
        self.virtual_pd_controller.reset(
            env_ids=env_ids,
            reference_quat_w=toaster_root_state[:, 3:7],
        )
        self.virtual_pd_controller.clear(env_ids)
        self.trajectory_start_pos_w[env_ids] = toaster_root_state[:, :3]

        self.trajectory_goal_pos_w[env_ids] = toaster_root_state[:, :3]
        self.trajectory_goal_pos_w[env_ids, 2] += self.cfg.trajectory_lift_delta_z
        self._sync_place_pos_command_xy(env_ids)

        info = self.dualarm_reward_manager.reset(env_ids)
        self.extras["log"].update(info)
        info = self.reg_reward_manager.reset(env_ids)
        self.extras["log"].update(info)

        self.success_hold_buf[env_ids] = 0

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
        time_s: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Analytic reference for:
          hold at initial toaster position
          -> smooth vertical lift
          -> hold at final position.

        Returns:
          reference_position_w: [num_envs, 3]
          reference_velocity_w: [num_envs, 3]
        """
        if time_s is None:
            time_s = self.episode_length_buf.float() * self.step_dt

        if time_s.ndim == 0:
            time_s = time_s.expand(self.num_envs)

        assert time_s.shape == (self.num_envs,), time_s.shape

        start_s = self.cfg.trajectory_start_s
        end_s = self.cfg.trajectory_end_s
        duration_s = end_s - start_s

        if duration_s <= 0.0:
            raise ValueError(
                f"trajectory_end_s ({end_s}) must be larger than "
                f"trajectory_start_s ({start_s})."
            )

        # u = 0 before lift begins; u = 1 after lift finishes.
        u = ((time_s - start_s) / duration_s).clamp(0.0, 1.0)

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
