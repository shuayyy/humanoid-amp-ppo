"""Task-level config helpers for the G1 whole-body grasp task.

This file defines grasp-specific config fields such as reach rewards,
grasp rewards, phase timing, and evaluation options. The final environment
configuration is built later in config/g1/env_cfgs.py.
"""


import math
from mjlab.envs import mdp as env_mdp
from mjlab.tasks.velocity import mdp as velocity_mdp
from mjlab_husky.envs.g1_wb_grasp_rl_env import G1GraspManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.command_manager import CommandTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab_husky.tasks.wb_grasp import mdp
from mjlab.terrains import TerrainImporterCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig


def make_g1_wb_grasp_env_cfg() -> G1GraspManagerBasedRlEnvCfg:
    ##
    # Observations
    ##
    # policy term = one observation input given to the actor/policy network.
    policy_terms = {
        "base_lin_vel": ObservationTermCfg(
            func=mdp.builtin_sensor,
            params={"sensor_name": "robot/imu_lin_vel"},
        ),
        "base_ang_vel": ObservationTermCfg(
            func=mdp.builtin_sensor,
            params={"sensor_name": "robot/imu_ang_vel"},
            noise=Unoise(n_min=-0.2, n_max=0.2),
            scale=0.25,
        ),
        # "root_pos": ObservationTermCfg(func=mdp.root_pos),
        # "root_ori": ObservationTermCfg(func=mdp.root_ori),

        "projected_gravity": ObservationTermCfg(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        ),
        "joint_pos": ObservationTermCfg(
            func=mdp.joint_pos_rel,
            noise=Unoise(n_min=-0.01, n_max=0.01),
        ),
        "joint_vel": ObservationTermCfg(
            func=mdp.joint_vel_rel,
            noise=Unoise(n_min=-1.5, n_max=1.5),
            scale=0.05,
        ),
        "actions": ObservationTermCfg(func=mdp.last_action),
        "object_pose": ObservationTermCfg(func=mdp.object_pose),
        "left_grasp_marker": ObservationTermCfg(
            func=mdp.left_grasp_marker_pos,
        ),
        "right_grasp_marker": ObservationTermCfg(
            func=mdp.right_grasp_marker_pos,
        ),
        # "place_pos": ObservationTermCfg(func=mdp.place_pos),
        # "vision": ObservationTermCfg(func=mdp.vision),
    }

    critic_terms = {
        **policy_terms,
            "foot_air_time": ObservationTermCfg(
            func=mdp.foot_air_time,
            params={"sensor_name": "feet_ground_contact"},
            ),
            "foot_contact": ObservationTermCfg(
            func=mdp.foot_contact,
            params={"sensor_name": "feet_ground_contact"},
            ),
            "foot_contact_forces": ObservationTermCfg(
            func=mdp.foot_contact_forces,
            params={"sensor_name": "feet_ground_contact"},
            ),
    }

    observations = {
        "policy": ObservationGroupCfg(
            terms=policy_terms,
            concatenate_terms=True,
            enable_corruption=True,
            history_length=5,
            flatten_history_dim=True,
        ),
        "critic": ObservationGroupCfg(
            terms=critic_terms,
            concatenate_terms=True,
            enable_corruption=False,
            history_length=5,
            
        ),
    }

    ##
    # Actions
    ##

    actions: dict[str, ActionTermCfg] = {
        "joint_pos": JointPositionActionCfg(
            entity_name="robot",  # Apply this action term to the robot entity.
            actuator_names=(".*",),  # Control all actuators/joints that match this regex.
            scale=0.5,  # Scale normalized policy outputs before converting to joint targets.
            use_default_offset=True,  # Interpret commands around the robot's default joint pose.
        )
    }

    ##
    # Commands
    ##
    # No task command for fixed toaster grasp yet.
    commands: dict[str, CommandTermCfg] = {}

    ##
    # Events
    ##
    """
    Events handle reset-time/startup operations, including domain randomization like friction, mass, COM, and joint reset.
    """

    events = {
        "reset_robot_joints": EventTermCfg(
            func=mdp.reset_joints_by_offset,
            mode="reset",
            params={
                "position_range": (-0.01, 0.01),
                "velocity_range": (0.0, 0.0),
                "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
            },
        ),
        "base_com": EventTermCfg(
            mode="startup",
            func=mdp.randomize_field,
            domain_randomization=True,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=("torso_link",)),
                "operation": "add",
                "field": "body_ipos",
                "ranges": {
                    0: (-0.025, 0.025),
                    1: (-0.025, 0.025),
                    2: (-0.03, 0.03),
                },
            },
        ),
        "object_com": EventTermCfg(
            mode="startup",
            func=mdp.randomize_field,
            domain_randomization=True,
            params={
                "asset_cfg": SceneEntityCfg("toaster", body_names=("object",)),
                "operation": "add",
                "field": "body_ipos",
                "ranges": {
                    0: (-0.01, 0.01),
                    1: (-0.01, 0.01),
                    2: (-0.01, 0.01),
                },
            },
        ),
        "robot_friction": EventTermCfg(
            mode="startup",
            func=mdp.randomize_field,
            domain_randomization=True,
            params={
                "asset_cfg": SceneEntityCfg("robot", geom_names=(".*",)),
                "operation": "scale",
                "field": "geom_friction",
                "ranges": (0.3, 1.6),
            },
        ),
        "object_friction": EventTermCfg(
            mode="startup",
            func=mdp.randomize_field,
            domain_randomization=True,
            params={
                "asset_cfg": SceneEntityCfg("toaster", geom_names=(".*",)),
                "operation": "scale",
                "field": "geom_friction",
                "ranges": (0.8, 1.2),
            },
        ),
        "foot_friction": EventTermCfg(
            mode="startup",
            func=mdp.randomize_field,
            domain_randomization=True,
            params={
                "asset_cfg": SceneEntityCfg("robot", geom_names=(r"^(left|right)_foot[1-7]_collision$",)),  # Set per-robot.
                "operation": "abs",
                "field": "geom_friction",
                "ranges": (0.3, 1.8),
            },
        ),
    }

    ##
    # Rewards
    ##

    ### reach rewards
    reach_rewards = {
        "hand_to_toaster": RewardTermCfg(
            func=mdp.hand_to_toaster,
            weight=50.0,
            params={"d_scale": 1.5},
        ),
        # "dist_to_toaster": RewardTermCfg(
        #     func=mdp.dist_to_toaster,
        #     weight=50.0,
        #     params={"d_scale": 1.5},
        # ),
    }

    grasp_rewards = {
      "hands_at_markers": RewardTermCfg(
        func=mdp.hands_at_markers,
        weight=50.0,
        params={"left_sensor": "left_hand_toaster_contact", "right_sensor": "right_hand_toaster_contact"},
      ),
      "hands_contact": RewardTermCfg(
        func=mdp.hands_contact,
        weight=10.0,
        params={"sensor_name": "toaster_contact"},
      ),
      "lift": RewardTermCfg(
        func=mdp.lift,
        weight=50.0,
        params={
          "left_sensor": "left_hand_toaster_contact",
          "right_sensor": "right_hand_toaster_contact",
        },
      ),
    }

    ### regularization rewards
    regularization_rewards = {
        "alive": RewardTermCfg(
            func=env_mdp.is_alive,
            weight=0.5,
        ),
        "dof_pos_limits": RewardTermCfg(func=mdp.joint_pos_limits, weight=-5.0),
        "action_rate_l2": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.01),
        "action_acc_l2": RewardTermCfg(func=mdp.action_acc_l2, weight=-0.01),
        # "joint_vel_l2": RewardTermCfg(func=mdp.joint_vel_l2, weight=-1e-3),
        # "joint_acc_l2": RewardTermCfg(func=mdp.joint_acc_l2, weight=-5.0e-7),
        "joint_torques_l2": RewardTermCfg(func=mdp.joint_torques_l2, weight=-1e-6),
        "self_collisions": RewardTermCfg(func=mdp.self_collision_cost, weight=-0.1, params={"sensor_name": "self_collision"}),
        # Disabled for now because env.step() sets `self.still[:] = False`,
        # so this reward is always zero.
        # "stand_still": RewardTermCfg(
        #     func=mdp.stand_still,
        #     weight=1.0,
        #     params={"d_scale": 1.5},
        # ),
        "upright": RewardTermCfg(
            func=velocity_mdp.flat_orientation,
            weight=5.0,
            params={
                "std": 0.5,
                "asset_cfg": SceneEntityCfg("robot", body_names=("torso_link",)),
            },
        ),
        "feet_stumble": RewardTermCfg(
            func=mdp.feet_stumble,
            weight=-0.1,
        ),
        "feet_slip": RewardTermCfg(
            func=mdp.feet_slip,
            weight=-0.05,
            params={"contact_force_threshold": 5.0},
        ),
        "at_least_one_foot_contact": RewardTermCfg(
            func=mdp.at_least_one_foot_contact,
            weight=0.5,
            params={
                "contact_force_threshold": 5.0,
                "illegal_sensor_names": (
                    "illegal_ground_contact",
                    "illegal_toaster_contact",
                ),
            },
        ),
        "illegal_contact": RewardTermCfg(
            func=mdp.illegal_contact_penalty,
            weight=-2.5,
            params={
                "sensor_names": (
                    "illegal_ground_contact",
                    "illegal_toaster_contact",
                ),
            },
        ),
    }
    ##
    # Terminations TODO: change termnation for the whole body grasp
    ##

    terminations = {
        "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
        # Disabled for now to avoid immediately training a fall-termination policy.
        "fell_over": TerminationTermCfg(
            func=mdp.bad_orientation,
            params={"limit_angle": math.radians(70.0)},
        ),
        "illegal_contact": TerminationTermCfg(
            func=mdp.illegal_contact,
            params={
                "sensor_names": (
                    "illegal_ground_contact",
                    "illegal_toaster_contact",
                ),
            },
        ),
        "success": TerminationTermCfg(
            func=mdp.grasp_success_held,
            params={
                "left_sensor": "left_hand_toaster_contact",
                "right_sensor": "right_hand_toaster_contact",
            },
        ),
    }

    ##
    # Curriculum
    ##

    curriculum = {}

    ##
    # Assemble and return
    ##

    return G1GraspManagerBasedRlEnvCfg(
        scene=SceneCfg(
            terrain=TerrainImporterCfg(
                terrain_type="plane",
                terrain_generator=None,
            ),
            num_envs=1024,
            extent=2.0,
        ),
        observations=observations,
        actions=actions,
        commands=commands,
        events=events,
        terminations=terminations,
        curriculum=curriculum,
        reach_rewards=reach_rewards,
        grasp_rewards=grasp_rewards,
        regularization_rewards=regularization_rewards,
        viewer=ViewerConfig(
            origin_type=ViewerConfig.OriginType.ASSET_BODY,
            entity_name="robot",
            body_name="",  # Set per-robot.
            distance=4.0,
            elevation=-10.0,
            azimuth=210.0,
        ),
        sim=SimulationCfg(
            nconmax=35,
            njmax=1500,
            mujoco=MujocoCfg(
                timestep=0.005,
                iterations=10,
                ls_iterations=20,
            ),
        ),
        cycle_time= 10.0,
        phase_ratios=[0.0, 0.5, 1.0],
        decimation=4,
        episode_length_s=10.0,
    )
