"""G1 locomotion task environment."""
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
from mjlab.viewer.debug_visualizer import DebugVisualizer
_DESIRED_FRAME_COLORS = ((1.0, 0.5, 0.5), (0.5, 1.0, 0.5), (0.5, 0.5, 1.0))

# dataclass auto-generates init/printing for config classes.
# kw_only=True forces fields to be passed by name, avoiding argument-order mistakes.

@dataclass(kw_only = True)
class G1LocomotionManagerBasedRlEnvCfg(ManagerBasedRlEnvCfg):

    # ManagerBasedRlEnvCfg is the base configuration class for an RL environment.
    # It stores common settings like scene, actions, observations, rewards, and resets.
    # This task config inherits from it to reuse the standard MJLab environment setup.
    # Extra fields here are task-specific settings for the G1 locomotion environment.
    locomotion_rewards: dict[str, RewardTermCfg] = field (default_factory= dict)
    regularization_rewards: dict[str, RewardTermCfg] = field (default_factory = dict)

    # These fields store task-specific reward groups.
    # Each dictionary maps a reward name to its RewardTermCfg.
    # field(default_factory=dict/list) creates a fresh empty dict/list for every config object.
    # This avoids different config objects accidentally sharing the same mutable default.

    fall_angle_thresh: float = math.radians(70.0)

    """Whether in evaluation mode. If True, will save metrics to JSON and exit after all episodes complete."""
    eval_output_dir: str | None = None
    """Directory to save eval metrics JSON files. If None, saves to current directory."""

class G1LocomotionManagerBasedRlEnv(ManagerBasedRlEnv):
    """Manager-based RL environment."""

    # Class-level metadata for the environment.
    # is_vector_env tells the RL code this env runs many parallel simulations.
    # metadata stores render/version information for logging and viewer support.
    # cfg is a type hint saying this env uses G1LocomotionManagerBasedRlEnvCfg.

    is_vector_env = True
    metadata = {
        "render_modes": [None, "rgb_array"],
        "mujoco_version": mujoco.__version__,
        "warp_version": wp.config.version,
    }
    cfg: G1LocomotionManagerBasedRlEnvCfg

    def __init__(
        self,
        cfg: G1LocomotionManagerBasedRlEnvCfg,
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
        such as `robot` for observations, rewards, and resets.
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
        self.robot = self.scene["robot"]
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
        and terminations.
        """
        self._init_ids_buffers()


    def _init_ids_buffers(self):
        """
        Site/body ID lookup.

        Cache MuJoCo body ids once so reward functions can gather foot state
        efficiently every step.
        """
        self.feet_body_ids, _ = self.robot.find_bodies(
            name_keys=["left_ankle_roll_link", "right_ankle_roll_link"],
            preserve_order=True,
        )

    def load_managers(self) -> None:
        super().load_managers()

        self.locomotion_reward_manager = RewardManager(
            self.cfg.locomotion_rewards, self, scale_by_dt = self.cfg.scale_rewards_by_dt
        )
        print_info(f"[INFO]: {self.locomotion_reward_manager}")

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
        
            self.sim.step()
            self.scene.update(dt=self.physics_dt)

        self.episode_length_buf += 1
        self.common_step_counter += 1

        self.reset_buf = self.termination_manager.compute()
        self.reset_terminated = self.termination_manager.terminated
        self.reset_time_outs = self.termination_manager.time_outs

        locomotion_reward_buf = self.locomotion_reward_manager.compute(self.step_dt)
        reg_reward_buf = self.reg_reward_manager.compute(self.step_dt)

        self.reward_buf = locomotion_reward_buf + reg_reward_buf
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
    
    def _reset_idx(self, env_ids: torch.Tensor | None = None) -> None:
        """
        Reset logic.

        Restore robot root states, reset task reward managers, and clear
        per-episode buffers for the environments that just terminated.
        """
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)

        super()._reset_idx(env_ids)

        env_origins = self.scene.env_origins[env_ids]

        robot_root_state = self.robot.data.default_root_state[env_ids].clone()
        robot_root_state[:, :3] += env_origins
        self.robot.write_root_state_to_sim(robot_root_state, env_ids)

        info = self.locomotion_reward_manager.reset(env_ids)
        self.extras["log"].update(info)
        info = self.reg_reward_manager.reset(env_ids)
        self.extras["log"].update(info)

    def get_amp_observations(self):
        return self.robot.data.joint_pos
