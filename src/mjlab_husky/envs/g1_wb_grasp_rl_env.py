"""G1 skateboarding task environment.

This file defines the real simulation environment for the Unitree G1
skateboarding task. It contains task-specific logic for the robot,
skateboard, reset behavior, contact phases, AMP observations, and helper
functions.

Pipeline:
    PPO/AMP runner
        -> vecenv_wrapper.py
        -> g1_skate_rl_env.py
        -> actual simulation task

File roles:
    vec_env.py:
        Abstract interface / rulebook for RSL-RL environments.

    vecenv_wrapper.py:
        Bridge between the MJLab environment and RSL-RL.

    g1_skate_rl_env.py:
        Actual task environment logic for skateboarding.

For adapting this file to fridge / whole-body manipulation:
    - Replace skateboard logic with fridge/object logic.
    - Replace foot-marker logic with hand-handle logic.
    - Replace skate phases with locomotion/grasp/lift/place phases.
    
    """
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
from mjlab.utils.logging import print_info
from mjlab.viewer.offscreen_renderer import OffscreenRenderer
from mjlab.utils.lab_api.math import(
    subtract_frame_transforms,
    quat_apply,
    quat_mul,
    matrix_from_quat,
)

from mjlab.viewer.debug_visualizer import DebugVisualizer
_DESIRED_FRAME_COLORS = ((1.0, 0.5, 0.5), (0.5, 1.0, 0.5), (0.5, 0.5, 1.0))

# dataclass auto-generates init/printing for config classes.
# kw_only=True forces fields to be passed by name, avoiding argument-order mistakes.

@dataclass(kw_only = True)
class G1GraspManagerBasedRlEnvCfg(ManagerBasedRlEnvCfg):

    # ManagerBasedRlEnvCfg is the base configuration class for an RL environment.
    # It stores common settings like scene, actions, observations, rewards, and resets.
    # This task config inherits from it to reuse the standard MJLab environment setup.
    # Extra fields here are task-specific settings for the G1 skateboarding environment.
    locomotion_rewards: dict[str, RewardTermCfg] = field (default_factory= dict)
    grasp_rewards: dict[str, RewardTermCfg] = field (default_factory = dict)
    regularization_rewards: dict[str, RewardTermCfg] = field (default_factory = dict)

    # These fields store groups of reward terms for different task phases.
    # Each dictionary maps a reward name to its RewardTermCfg.
    # field(default_factory=dict/list) creates a fresh empty dict/list for every config object.
    # This avoids different config objects accidentally sharing the same mutable default.


    # Extra fields here are task-specific settings for the G1 whole-body grasp environment.
    lift_height_thresh: float = 0.8
    hold_steps: int = 20
    fall_angle_thresh: float = math.radians(70.0)

    cycle_time: float = 6.0
    phase_ratios: list[float] = field(default_factory=list)

    """Whether in evaluation mode. If True, will save metrics to JSON and exit after all episodes complete."""
    eval_output_dir: str | None = None
    """Directory to save eval metrics JSON files. If None, saves to current directory."""

class G1GraspManagerBasedRlEnv(ManagerBasedRlEnv):
    """Manager-based RL environment."""

    # Class-level metadata for the environment.
    # is_vector_env tells the RL code this env runs many parallel simulations.
    # metadata stores render/version information for logging and viewer support.
    # cfg is a type hint saying this env uses G1GraspManagerBasedRlEnvCfg.

    is_vector_env = True
    metadata = {
        "render_modes": [None, "rgb_array"],
        "mujoco_version": mujoco.__version__,
        "warp_version": wp.config.version,
    }
    cfg: G1GraspManagerBasedRlEnvCfg

    def __init__(
        self,
        cfg: G1GraspManagerBasedRlEnvCfg,
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


        self.cycle_time = self.cfg.cycle_time
        self.robot = self.scene["robot"]
        # self.object = self.scene["toaster"]
        # self.toaster = self.object
        self._init_buffers()

        self.common_step_counter = 0
        self.episode_length_buf = torch.zeros(
            cfg.scene.num_envs, device=self.device, dtype=torch.long
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
        self.setup_manager_visualizers()
   
  
    def _init_buffers(self):
        """
        Buffers.

        Allocate per-environment tensors used across steps, rewards,
        terminations, phase tracking, contact filtering, and success holding.
        """
        self._init_ids_buffers()
        self.phase_ratios = torch.tensor(self.cfg.phase_ratios, device=self.device).repeat(self.num_envs, 1)
        self.last_contacts = torch.zeros(self.num_envs, 2, dtype=torch.bool, device=self.device, requires_grad=False)
        self.contact_filt = torch.zeros_like(self.last_contacts)
        self.contact_phase = torch.zeros(self.num_envs, 2, dtype=torch.float, device=self.device, requires_grad=False)

        self.phase_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long, requires_grad=False)
        self.success_hold_buf = torch.zeros(
            self.num_envs,
            dtype=torch.long,
            device=self.device,
            requires_grad=False,
        )
        # Toaster disabled for locomotion-only runs.
        self.object_lift_target_pos_w = torch.zeros(
            self.num_envs, 3, device=self.device, dtype=torch.float
        )
        # self.object_lift_target_pos_w = self.toaster.data.default_root_state[:, :3].clone()
        # self.object_lift_target_pos_w += self.scene.env_origins
        # self.object_lift_target_pos_w[:, 2] += self.cfg.lift_height_thresh
        self.still = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False)
        
        # push_init_body_pose = torch.from_numpy(np.load("dataset/ref_pose/push_start_pose_b.npy")).to(self.device).repeat(self.num_envs, 1 , 1)
        # steer_init_body_pose = torch.from_numpy(np.load("dataset/ref_pose/steer_start_pose_b.npy")).to(self.device).repeat(self.num_envs, 1 , 1)


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
        # Feet body ids used by feet_slip reward.
        # Same body names used in Twist2 MJLab.
        self.feet_body_ids, _ = self.robot.find_bodies(
            name_keys=["left_ankle_roll_link", "right_ankle_roll_link"],
            preserve_order=True,
        )
        # Toaster disabled for locomotion-only runs.
        self.marker_body_ids = []
        self.hand_site_ids, _ = self.robot.find_sites(
            name_keys=["left_palm", "right_palm"],
            preserve_order=True,
        )
        self.grasp_site_ids = []
        # self.marker_body_ids, _ = self.toaster.find_sites(
        #     name_keys=[".*_marker"], preserve_order=True
        # )
        # self.grasp_site_ids, _ = self.toaster.find_sites(
        #     name_keys=["left_grasp_marker", "right_grasp_marker"],
        #     preserve_order=True,
        # )
        self.left_foot_site_ids, _ = self.robot.find_sites(name_keys=["left_foot_1", "left_foot_2", "left_foot_3", "left_foot_4"], preserve_order=True)
        self.right_foot_site_ids, _ = self.robot.find_sites(name_keys=["right_foot_1", "right_foot_2", "right_foot_3", "right_foot_4"], preserve_order=True)

    def load_managers(self) -> None:
        # First load the default MJLab managers:
        # action_manager, observation_manager, termination_manager, etc.
        super().load_managers()

        self.locomotion_reward_manager = RewardManager(
            self.cfg.locomotion_rewards, self, scale_by_dt = self.cfg.scale_rewards_by_dt
        )
        print_info(f"[INFO]: {self.locomotion_reward_manager}")

        self.grasp_reward_manager = RewardManager(
            self.cfg.grasp_rewards, self, scale_by_dt = self.cfg.scale_rewards_by_dt
        )
        print_info(f"[INFO]: {self.grasp_reward_manager}")

        self.reg_reward_manager = RewardManager(
            self.cfg.regularization_rewards, self, scale_by_dt = self.cfg.scale_rewards_by_dt
        )
        print_info(f"[INFO]: {self.reg_reward_manager}")
    
    def get_heading_target_w(self) -> torch.Tensor:
        return self.toaster_target_pos_w
    
    def step(self, action: torch.Tensor) -> types.VecEnvStepReturn:
        """Apply action, step the simulation, and return observations, rewards, dones, and infos."""
        self.action_manager.process_action(action.to(self.device))
        # self.still = self.command_manager.get_command("locomotion")[:, 0] < 0.1
        self.still[:] = False
        q_before = self.robot.data.joint_pos.clone()

        for _ in range(self.cfg.decimation):
            """DECIMATION is a common technique in RL environments where the simulation runs at a higher frequency than the agent's action frequency. For example, if the simulation runs at 1000 Hz and the agent acts at 20 Hz, then decimation would be 50. This means that for each action taken by the agent, the simulation will step forward 50 times before the next action is applied. This allows for more realistic physics and smoother control while keeping the agent's decision-making at a manageable frequency."""
            # Number of physics/simulation substeps executed for each single RL/control step.
            # Higher decimation means actions are held constant for more simulator steps.

            self._sim_step_counter += 1
            self.action_manager.apply_action()
            self.scene.write_data_to_sim()
        
            self.sim.step()
            self.scene.update(dt=self.physics_dt)

        if self.common_step_counter % 50 == 0:
            q_after = self.robot.data.joint_pos
            print(
                "[JOINT DEBUG]",
                "joint_delta_max=",
                (q_after - q_before).abs().max().item(),
            )
        
        self.episode_length_buf += 1
        self.phase_length_buf += 1
        self.common_step_counter += 1
        self._compute_contact()

        self.reset_buf = self.termination_manager.compute()
        self.reset_terminated = self.termination_manager.terminated
        self.reset_time_outs = self.termination_manager.time_outs

        # Phase gating disabled for WB grasp MVP.
        # All task rewards are active for the full episode.
        # Keep contact_phase / phase observation code for later curriculum use.
        # contact_coef = self.contact_phase.clone()
        # push_reward_buf = self.push_reward_manager.compute(self.step_dt) * contact_coef[:, 0]
        # steer_reward_buf = self.steer_reward_manager.compute(self.step_dt) * contact_coef[:, 1]
        # transition_reward_buf = self.transition_reward_manager.compute(self.step_dt) * torch.logical_or(contact_coef[:, 2], contact_coef[:, 3])
        # self.reward_buf = steer_reward_buf + push_reward_buf + reg_reward_buf + transition_reward_buf
        locomotion_reward_buf = self.locomotion_reward_manager.compute(self.step_dt)
        grasp_reward_buf = self.grasp_reward_manager.compute(self.step_dt)
        reg_reward_buf = self.reg_reward_manager.compute(self.step_dt)

        self.reward_buf = locomotion_reward_buf + grasp_reward_buf + reg_reward_buf
        self.reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(self.reset_env_ids) > 0:
            self._reset_idx(self.reset_env_ids)
            self.scene.write_data_to_sim()
            self.sim.forward()

        self.command_manager.compute(dt = self.step_dt)

        if "interval" in self.event_manager.available_modes:
            self.event_manager.apply(mode="interval", dt=self.step_dt)

        self.obs_buf = self.observation_manager.compute(update_history=True)
        return(
            self.obs_buf,
            self.reward_buf,
            self.reset_terminated,
            self.reset_time_outs,
            self.extras,
        )
    
    def update_visualizers(self, visualizer: DebugVisualizer) -> None:
        super().update_visualizers(visualizer)
        # self._visualize_transition_target(visualizer)
        self._visualize_contact_phase(visualizer)
    
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

        # toaster_root_state = self.toaster.data.default_root_state[env_ids].clone()
        # toaster_root_state[:, :3] += env_origins
        # self.toaster.write_root_state_to_sim(toaster_root_state, env_ids)
        # self.object_lift_target_pos_w[env_ids] = toaster_root_state[:, :3]
        # self.object_lift_target_pos_w[env_ids, 2] += self.cfg.lift_height_thresh

        info = self.locomotion_reward_manager.reset(env_ids)
        self.extras["log"].update(info)
        info = self.grasp_reward_manager.reset(env_ids)
        self.extras["log"].update(info)
        info = self.reg_reward_manager.reset(env_ids)
        self.extras["log"].update(info)

        self.phase_length_buf[env_ids] = 0
        self.success_hold_buf[env_ids] = 0

    def _compute_contact(self):
        """
        Contact-phase state.

        Toaster contact is disabled for locomotion-only runs, so keep the buffers
        at zero and just update the phase values.
        """
        # left_sensor = self.scene.sensors["left_hand_toaster_contact"]
        # right_sensor = self.scene.sensors["right_hand_toaster_contact"]
        # assert left_sensor.data.force is not None
        # assert right_sensor.data.force is not None
        # left_contact = torch.any(torch.norm(left_sensor.data.force, dim=-1) > 2.0, dim=-1)
        # right_contact = torch.any(torch.norm(right_sensor.data.force, dim=-1) > 2.0, dim=-1)
        # contact = torch.stack([left_contact, right_contact], dim=-1)
        contact = torch.zeros_like(self.last_contacts)
        self.contact_filt = contact
        self.last_contacts = contact
        self._resample_contact_phases()

    def _resample_contact_phases(self):
        self.last_contact_phase = self.contact_phase.clone()
        phase = self._get_phase()

        locomotion_phase = (phase >= self.phase_ratios[:, 0]) & (phase < self.phase_ratios[:, 1])
        grasp_phase = (phase >= self.phase_ratios[:, 1]) & (phase <= self.phase_ratios[:, 2])

        self.contact_phase[:, 0] = locomotion_phase.float()
        self.contact_phase[:, 1] = grasp_phase.float()

    # Convert per-environment phase step counters into a normalized cycle phase in [0, 1].
    # For envs marked as `still`, reset the phase counter at each half-cycle boundary.
    def _get_phase(self):
        # Convert phase step count into normalized phase [0, 1]
        phase = (self.phase_length_buf * self.step_dt / self.cycle_time)
        return torch.clip(phase, 0.0, 1.0)

    def _get_hand_toaster_dis(self):
        """
        Helper methods.

        Return vector differences from each hand palm site to its corresponding
        toaster grasp-marker site.
        """
        hand_pos = self.robot.data.site_pos_w[:, self.hand_site_ids, :3]
        grasp_pos = self.toaster.data.site_pos_w[:, self.grasp_site_ids, :3]

        return grasp_pos - hand_pos

    def _object_lifted(self) -> torch.Tensor:
        """Return whether the toaster has been lifted past the configured height threshold."""
        toaster_z = self.toaster.data.root_link_pos_w[:, 2]
        return toaster_z > self.object_lift_target_pos_w[:, 2]
    
    def stand_still(self, sensor_name: str, sensor2_name: str) -> torch.Tensor:
        """Return whether both named contact sensors are active in each environment."""
        sensor = self.scene.sensors[sensor_name]
        sensor2 = self.scene.sensors[sensor2_name]
        assert sensor.data.found is not None
        assert sensor2.data.found is not None

        contact = torch.any(sensor.data.found > 0, dim=-1)
        contact2 = torch.any(sensor2.data.found > 0, dim=-1)

        return (contact & contact2).float()


    def get_amp_observations(self):
        return self.robot.data.joint_pos
    
    def _visualize_contact_phase(self, visualizer: DebugVisualizer):
        contact_phase = self.contact_phase.clone()

        locomotion_phase = contact_phase[:, 0]
        grasp_phase = contact_phase[:, 1]

        target_pos_w = self.robot.data.root_link_pos_w.clone()
        target_pos_w[..., 2] += 0.75

        env_idx = visualizer.env_idx

        if locomotion_phase[env_idx].item():
            visualizer.add_sphere(
                center=target_pos_w[env_idx],
                radius=0.05,
                color=(0.0, 0.0, 1.0, 1.0),
                label="locomotion_phase",
            )

        if grasp_phase[env_idx].item():
            visualizer.add_sphere(
                center=target_pos_w[env_idx],
                radius=0.05,
                color=(1.0, 0.0, 0.0, 1.0),
                label="grasp_phase",
            )
