"""Task-level config helpers for the G1 dual-arm manipulation task."""


import math
from mjlab.envs import mdp as env_mdp
from mjlab_g1.envs.g1_dualarm_rl_env import G1DualarmManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.command_manager import CommandTermCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.tasks.manipulation.mdp.commands import LiftingCommandCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab_g1.tasks.dualarm import mdp
from mjlab_g1.tasks.dualarm.virtual_object_pd import VirtualObjectPdCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig


def make_g1_dualarm_env_cfg() -> G1DualarmManagerBasedRlEnvCfg:
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

        "object_pose": ObservationTermCfg(func=mdp.object_pose),
        "left_grasp_marker": ObservationTermCfg(
            func=mdp.left_grasp_marker_pos,
        ),
        "right_grasp_marker": ObservationTermCfg(
            func=mdp.right_grasp_marker_pos,
        ),
        "trajectory_reference_pos": ObservationTermCfg(
            func=mdp.trajectory_reference_pos,
        ),
        "depth_features": ObservationTermCfg(
            func=mdp.get_depth_features,
        ),
        # "vision": ObservationTermCfg(func=mdp.vision),
    }

    critic_terms = {
        **policy_terms,
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
        "place_pos": LiftingCommandCfg(
            entity_name="toaster",
            resampling_time_range=(1.0e9, 1.0e9),
            success_threshold=0.075,
            difficulty="fixed",
            target_position_range=LiftingCommandCfg.TargetPositionRangeCfg(
                x=(0.0, 0.0),
                y=(0.0, 0.0),
                z=(0.750, 0.750),
            ),
            object_pose_range=None,
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

        "foot_friction": EventTermCfg(
            mode="startup",
            func=mdp.randomize_field,
            params={
                "asset_cfg": SceneEntityCfg("robot", geom_names=(r"^(left|right)_foot[1-7]_collision$",)),  # Set per-robot.
                "operation": "abs",
                "field": "geom_friction",
                "ranges": (0.6, 1.0),
            },
        ),


        "object_com": EventTermCfg(
            mode="startup",
            func=mdp.randomize_field,
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
            params={
                "asset_cfg": SceneEntityCfg("robot", geom_names=(".*",)),
                "operation": "scale",
                "field": "geom_friction",
                "ranges": (0.7, 1.3),
            },
        ),
        "object_friction": EventTermCfg(
            mode="startup",
            func=mdp.randomize_field,
            params={
                "asset_cfg": SceneEntityCfg("toaster", geom_names=(".*",)),
                "operation": "scale",
                "field": "geom_friction",
                "ranges": (0.8, 1.2),
            },
        ),

    }

    ##
    # Rewards
    ##

    dualarm_rewards = {
        "yaw_rate_penalty": RewardTermCfg(
            func=mdp.yaw_rate_penalty,
            weight=-0.5,
            params={"threshold": 0.005},
        ),
        "angular_vel_penalty": RewardTermCfg(
            func=mdp.angular_vel_penalty,
            weight=-0.5,
            params={"threshold": 0.01},
        ),
        "feet_slip": RewardTermCfg(
            func=mdp.feet_slip,
            weight=-0.5,
            params={"threshold_min": 0.05},
        ),
        # "linear_vel_penalty": RewardTermCfg(
        #     func=mdp.linear_vel_penalty,
        #     weight=-0.5,
        #     params={"threshold": 0.075},
        # ),
        "hands_contact": RewardTermCfg(
            func=mdp.hands_contact,
            weight=0.01,
            params={"sensor_name": "toaster_contact", "min_reward_time_s": 1.0},
        ),
        "hands_at_markers": RewardTermCfg(
            func=mdp.hands_at_markers,
            weight=5.0,
            params={
                "left_sensor": "left_hand_toaster_contact",
                "right_sensor": "right_hand_toaster_contact",
                "min_reward_time_s": 1.0,
            },
        ),
        "marker_force": RewardTermCfg(
            func=mdp.marker_force,
            weight=2.0,
            params={
                "left_sensor": "left_hand_toaster_contact",
                "right_sensor": "right_hand_toaster_contact",
                "min_reward_time_s": 0.5,
                "target_force": 10.0,
            },
        ),
        "hand_to_toaster": RewardTermCfg(
            func=mdp.hand_to_toaster,
            weight=5.0,
            params={"d_scale": 0.75},
        ),
        # Sharp bilateral bonus: only pays off when BOTH palms are on their
        # markers, guiding the robot into the pre-contact grasp configuration.
        "hands_near_markers": RewardTermCfg(
            func=mdp.hands_near_markers,
            weight=5.0,
            params={"d_scale": 0.15},
        ),
        # Even sharper: supplies gradient across the last few cm into contact,
        # where hands_near_markers has already saturated.
        "grasp_approach": RewardTermCfg(
            func=mdp.grasp_approach,
            weight=5.0,
            params={"d_scale": 0.06},
        ),
        "object_trajectory_tracking": RewardTermCfg(
            func=mdp.object_trajectory_tracking,
            weight=10.0,
            params={
                "left_sensor": "left_hand_toaster_contact",
                "right_sensor": "right_hand_toaster_contact",
                "position_tolerance": 0.075,
            },
        ),
        "missing_grasp_during_lift": RewardTermCfg(
            func=mdp.missing_grasp_during_lift,
            weight=0.0,
            params={
                "left_sensor": "left_hand_toaster_contact",
                "right_sensor": "right_hand_toaster_contact",
            },
        ),
        "feet_contact": RewardTermCfg(
            func=mdp.feet_contact,
            weight=2.5,
        ),
        # Stand-first: give the robot a reason to stay upright and alive, since
        # the contact-gated task rewards are ~0 until it learns to grasp.
        "upright": RewardTermCfg(
            func=mdp.upright,
            weight=1.0,
        ),
        "alive": RewardTermCfg(
            func=env_mdp.is_alive,
            weight=1.0,
        ),
    }
    
    ### regularization rewards
    regularization_rewards = {
        # "alive": RewardTermCfg(
        #     func=env_mdp.is_alive,
        #     weight=0.1,
        # ),
        "dof_pos_limits": RewardTermCfg(func=mdp.joint_pos_limits, weight=-0.1),
        # "action_rate_l2": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.05),
        # "action_acc_l2": RewardTermCfg(func=mdp.action_acc_l2, weight=-0.01),
        # "joint_vel_l2": RewardTermCfg(func=mdp.joint_vel_l2, weight=-1e-3),
        # "joint_acc_l2": RewardTermCfg(func=mdp.joint_acc_l2, weight=-5.0e-7),
        # "joint_torques_l2": RewardTermCfg(func=mdp.joint_torques_l2, weight=-1e-6),
        "self_collisions": RewardTermCfg(func=mdp.self_collision_cost, weight=-0.1, params={"sensor_name": "self_collision"}),
        "illegal_contact": RewardTermCfg(
            func=mdp.illegal_contact_penalty,
            weight=-1.0e-3,
            params={
                "sensor_names": (
                    "illegal_ground_contact",
                    # "illegal_toaster_contact",
                ),
            },
        ),
    }
    terminations = {
        "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
        # Ends the episode when the torso tilts past 70deg, so episode length is a
        # real survival signal and falling costs future upright/alive reward.
        "fell_over": TerminationTermCfg(
            func=mdp.bad_orientation,
            params={"limit_angle": math.radians(70.0)},
        ),
        # "illegal_contact": TerminationTermCfg(
        #     func=mdp.illegal_contact,
        #     params={
        #         "sensor_names": (
        #             "illegal_ground_contact",
        #             "illegal_toaster_contact",
        #         ),
        #     },
        # ),
        # "success": TerminationTermCfg(
        #     func=mdp.grasp_success_held,
        #     params={
        #         "left_sensor": "left_hand_toaster_contact",
        #         "right_sensor": "right_hand_toaster_contact",
        #     },
        # ),
    }

    ##
    # Curriculum
    ##

    curriculum = {
        "virtual_pd_assistance": CurriculumTermCfg(
            func=mdp.virtual_pd_assistance_curriculum,
        ),
        "object_reset_height": CurriculumTermCfg(
            func=mdp.object_reset_height_curriculum,
        ),
        "feet_slip": CurriculumTermCfg(
            func=mdp.feet_slip_curriculum,
        ),
        "missing_grasp": CurriculumTermCfg(
            func=mdp.missing_grasp_curriculum,
        ),
    }

    ##
    # Assemble and return
    ##

    return G1DualarmManagerBasedRlEnvCfg(
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
        dualarm_rewards=dualarm_rewards,
        regularization_rewards=regularization_rewards,
        virtual_pd_cfg=VirtualObjectPdCfg(
            enabled=True,
            scale=1.0,
            kp_pos=800.0,
            kd_pos=50.0,
            max_force=80.0,
            kp_rot=4.0,
            kd_rot=0.75,
            max_torque=4.0,
        ),
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
        episode_length_s=7.0,
    )
