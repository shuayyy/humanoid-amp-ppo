"""Task-level config helpers for the G1 locomotion task."""


import math
from mjlab.envs import mdp as env_mdp
from mjlab_g1.envs.g1_locomotion_rl_env import G1LocomotionManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.command_manager import CommandTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab_g1.tasks.locomotion import mdp
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig


def make_g1_locomotion_env_cfg() -> G1LocomotionManagerBasedRlEnvCfg:
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
    commands: dict[str, CommandTermCfg] = {
    "gostraight": UniformVelocityCommandCfg(
        entity_name="robot",
        resampling_time_range=(3.0, 8.0),
        rel_standing_envs=0.0,
        rel_heading_envs=0.0,
        # rel_forward_envs=1.0,
        heading_command=False,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=UniformVelocityCommandCfg.Ranges(
        lin_vel_x=(1.0, 1.3),
        lin_vel_y=(0.0, 0.0),
        ang_vel_z=(0.0, 0.0),
        heading=None,
        ),
    )
    }


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

        "push_robot": EventTermCfg(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(1.0, 3.0),
        params={
            "velocity_range": {
            "x": (-0.1, 0.1),
            "y": (-0.1, 0.1),
            "z": (-0.1, 0.1),
            "roll": (-0.32, 0.32),
            "pitch": (-0.32, 0.32),
            "yaw": (-0.48, 0.48),
            },
        },
        ),
        
        "foot_friction": EventTermCfg(
            mode="startup",
            func=mdp.randomize_field,
            params={
                "asset_cfg": SceneEntityCfg("robot", geom_names=(r"^(left|right)_foot[1-7]_collision$",)),  # Set per-robot.
                "operation": "abs",
                "field": "geom_friction",
                "ranges": (0.3, 1.2),
            },
        ),
    }

    ##
    # Rewards
    ##

    ### locomotion rewards
    locomotion_rewards = {
        "track_linear_velocity": RewardTermCfg(
            func=mdp.track_lin_vel,
            weight=5.0,
            params={
                "command_name": "gostraight",
                "std": math.sqrt(0.25),
                "y_deadzone": (-0.0075, 0.075),
                "z_deadzone": (-0.0075, 0.075),
            },
        ),
        "yaw_rate_penalty": RewardTermCfg(
            func=mdp.yaw_rate_penalty,
            weight=-0.05,
            params={
                "command_name": "gostraight",
                "threshold": 0.1,
            },
        ),
        "feet_stumble": RewardTermCfg(
            func=mdp.feet_stumble,
            weight=-0.1,
        ),
        "feet_slip": RewardTermCfg(
            func=mdp.feet_slip,
            weight=-0.05,
            params={"threshold_min": 0.05},
        ),
        "air_time": RewardTermCfg(
            func=mdp.feet_air_time,
            weight=0.05,  # Override per-robot.
            params={
                "sensor_name": "feet_ground_contact",
                "threshold_min": 0.05,
                "threshold_max": 0.5,
                "command_name": "gostraight",
                "command_threshold": 0.5,
            },
        ),
        "upright": RewardTermCfg(
            func=mdp.torso_upright,
            weight=1.0,
            params={
                "std": math.sqrt(0.2),
                "tilt_threshold": 0.25,
            },
        ),
        "soft_landing": RewardTermCfg(
            func=mdp.soft_landing,
            weight=-1e-5,
            params={
                "sensor_name": "feet_ground_contact",
                "threshold_min": 0.05,
            },
            ),
        }
    
    ### regularization rewards
    regularization_rewards = {
        "alive": RewardTermCfg(
            func=env_mdp.is_alive,
            weight=0.1,
        ),
        "dof_pos_limits": RewardTermCfg(func=mdp.joint_pos_limits, weight=-0.1),
        "action_rate_l2": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.05),
        # "action_acc_l2": RewardTermCfg(func=mdp.action_acc_l2, weight=-0.01),
        # "joint_vel_l2": RewardTermCfg(func=mdp.joint_vel_l2, weight=-1e-3),
        # "joint_acc_l2": RewardTermCfg(func=mdp.joint_acc_l2, weight=-5.0e-7),
        # "joint_torques_l2": RewardTermCfg(func=mdp.joint_torques_l2, weight=-1e-6),
        "self_collisions": RewardTermCfg(func=mdp.self_collision_cost, weight=-0.1, params={"sensor_name": "self_collision"}),
    }

    terminations = {
        "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
        # Disabled for now to avoid immediately training a fall-termination policy.
        "fell_over": TerminationTermCfg(
            func=mdp.bad_orientation,
            params={"limit_angle": math.radians(70.0)},
        ),
    }

    ##
    # Curriculum
    ##

    curriculum = {}

    ##
    # Assemble and return
    ##

    return G1LocomotionManagerBasedRlEnvCfg(
        scene=SceneCfg(
            terrain=TerrainEntityCfg(
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
        locomotion_rewards=locomotion_rewards,
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
        decimation=4,
        episode_length_s=10.0,
    )
