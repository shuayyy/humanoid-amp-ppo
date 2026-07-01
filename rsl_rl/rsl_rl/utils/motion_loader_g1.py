import os
from os.path import join as pjoin
import glob
import json
import logging

import torch
import numpy as np
from pybullet_utils import transformations

from rsl_rl.utils import motion_util

_EPS = np.finfo(float).eps * 4.0


def normalize_quat_wxyz(q: torch.Tensor) -> torch.Tensor:
    return q / torch.linalg.norm(q, dim=-1, keepdim=True).clamp_min(1.0e-8)


def quaternion_slerp(q0, q1, fraction, spin=0, shortestpath=True):
    """Batch quaternion spherical linear interpolation."""
    q0 = normalize_quat_wxyz(q0)
    q1 = normalize_quat_wxyz(q1)

    dot = torch.sum(q0 * q1, dim=-1, keepdim=True)
    if shortestpath:
        q1 = torch.where(dot < 0.0, -q1, q1)
        dot = torch.abs(dot)
    dot = torch.clamp(dot, -1.0, 1.0)

    lerp = normalize_quat_wxyz((1.0 - fraction) * q0 + fraction * q1)
    angle = torch.acos(dot) + spin * torch.pi
    sin_angle = torch.sin(angle)
    slerp = (
        torch.sin((1.0 - fraction) * angle) / sin_angle.clamp_min(1.0e-8) * q0
        + torch.sin(fraction * angle) / sin_angle.clamp_min(1.0e-8) * q1
    )
    use_lerp = torch.abs(sin_angle) < 1.0e-6
    return normalize_quat_wxyz(torch.where(use_lerp, lerp, slerp))


def quat_conjugate_wxyz(q: torch.Tensor) -> torch.Tensor:
    return torch.cat((q[..., :1], -q[..., 1:]), dim=-1)


def quat_mul_wxyz(q: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    w1, x1, y1, z1 = q.unbind(dim=-1)
    w2, x2, y2, z2 = r.unbind(dim=-1)
    return torch.stack(
        (
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ),
        dim=-1,
    )


def quat_apply_inverse_wxyz(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    q_vec = -q[..., 1:]
    q_w = q[..., :1]
    t = 2.0 * torch.cross(q_vec, v, dim=-1)
    return v + q_w * t + torch.cross(q_vec, t, dim=-1)


def angular_velocity_b_from_quat_delta(
    previous_quat_wxyz: torch.Tensor,
    current_quat_wxyz: torch.Tensor,
    dt: float,
) -> torch.Tensor:
    q_delta = quat_mul_wxyz(
        quat_conjugate_wxyz(previous_quat_wxyz),
        current_quat_wxyz,
    )
    q_delta = torch.where(q_delta[..., :1] < 0.0, -q_delta, q_delta)
    return 2.0 * q_delta[..., 1:] / dt


class G1_AMPLoader:

    def __init__(
            self,
            device,
            time_between_frames,
            motion_files,
            preload_transitions=False,
            num_preload_transitions=1000000,
            num_frames=5,
            amp_observation_mode="joint_pos",
        ):
        """Expert dataset provides AMP observations from Dog mocap dataset.

        time_between_frames: Amount of time in seconds between transition.
        """
        self.device = device
        self.time_between_frames = time_between_frames
        self.num_frames = num_frames
        self.amp_observation_mode = amp_observation_mode
        self._amp_observation_dim = self._resolve_amp_observation_dim(
            amp_observation_mode
        )
        
        # Values to store for each trajectory.
        self.trajectories = []
        self.trajectories_full = []
        self.trajectory_names = []
        self.trajectory_idxs = []
        self.trajectory_lens = []  # Traj length in seconds.
        self.trajectory_weights = []
        self.trajectory_frame_durations = []
        self.trajectory_num_frames = []
        self.motion_dir = motion_files
        for i, motion_file in enumerate(os.listdir(motion_files)):
            self.trajectory_names.append(motion_file)
            motion_path = pjoin(motion_files, motion_file)
            motion_data = np.load(motion_path, allow_pickle=True)
            motion_data_processed = np.zeros((motion_data.shape[0],36))
            
            for f_i in range(motion_data.shape[0]):
                motion_data_processed[f_i, :3] = motion_data[f_i, :3]   # base pos
                motion_data_processed[f_i, 3:7] = motion_data[f_i, 3:7]  # base quat   (wxyz)
                motion_data_processed[f_i, 7:36] = motion_data[f_i, 7:36]  # base vel
                '''
                NOTE The order of motion_data_processed is
                base pos 0:3,
                base quat 3:7,  wxyz
                dof pos  7:36,  (mujoco joint order)
                '''
            self.trajectories.append(torch.tensor(
                motion_data_processed[:, 7:],
                dtype=torch.float32,
                device=self.device
            ))
            
            self.trajectories_full.append(torch.tensor(
                motion_data_processed,
                dtype=torch.float32,
                device=self.device
            ))
            
            self.trajectory_idxs.append(i)
            self.trajectory_weights.append(1 / len(os.listdir(motion_files)))
            frame_duration = 1 / 30
            
            self.trajectory_frame_durations.append(frame_duration)
            traj_len = (motion_data_processed.shape[0] - 1) * frame_duration # seconds
            self.trajectory_lens.append(traj_len)
            self.trajectory_num_frames.append(float(motion_data_processed.shape[0]))
            print(f"Loaded {traj_len}s. motion from {motion_file}.")
            
        # Trajectory weights are used to sample some trajectories more than others.
        self.trajectory_weights = np.array(self.trajectory_weights) / np.sum(self.trajectory_weights)
        self.trajectory_frame_durations = np.array(self.trajectory_frame_durations)
        self.trajectory_lens = np.array(self.trajectory_lens)
        self.trajectory_num_frames = np.array(self.trajectory_num_frames)

        # Preload transitions.
        self.preload_transitions = preload_transitions
        if self.preload_transitions:
            print(f'Preloading {num_preload_transitions} transitions')
    
            traj_idxs = self.weighted_traj_idx_sample_batch(num_preload_transitions)
            times = self.traj_time_sample_batch(traj_idxs)
            self.preloaded_s_prior = self.get_full_frame_at_time_batch(traj_idxs, times - self.time_between_frames)
            self.preloaded_s = self.get_full_frame_at_time_batch(traj_idxs, times)
            self.preloaded_s_next = self.get_full_frame_at_time_batch(traj_idxs, times + self.time_between_frames)
            print(f'Finished preloading')

            # 预加载多帧数据
            self.preloaded_frames = []
            for i in range(self.num_frames):
                frame_time = times + (i - (self.num_frames - 2)) * self.time_between_frames
                full_frame = self.get_full_frame_at_time_batch(traj_idxs, frame_time)
                previous_full_frame = self.get_full_frame_at_time_batch(
                    traj_idxs,
                    frame_time - self.time_between_frames,
                )
                processed_frame = self.build_amp_observation(
                    full_frame,
                    previous_full_frame,
                )

                self.preloaded_frames.append(processed_frame)
            print(f'Finished preloading multiple frames')

        self.all_trajectories_full = torch.vstack(self.trajectories_full)

    def weighted_traj_idx_sample(self):
        """Get traj idx via weighted sampling."""
        return np.random.choice(
            self.trajectory_idxs, p=self.trajectory_weights)

    def weighted_traj_idx_sample_batch(self, size):
        """Batch sample traj idxs."""
        return np.random.choice(
            self.trajectory_idxs, size=size, p=self.trajectory_weights,
            replace=True)

    def traj_time_sample(self, traj_idx):
        """Sample random time for traj."""
        history_s = max(0, self.num_frames - 1) * self.time_between_frames
        latest_s = max(0.0, self.trajectory_lens[traj_idx] - self.time_between_frames)
        earliest_s = min(history_s, latest_s)
        return earliest_s + (latest_s - earliest_s) * np.random.uniform()

    def traj_time_sample_batch(self, traj_idxs):
        """Sample random time for multiple trajectories."""
        history_s = max(0, self.num_frames - 1) * self.time_between_frames
        latest_s = np.maximum(
            0.0,
            self.trajectory_lens[traj_idxs] - self.time_between_frames,
        )
        earliest_s = np.minimum(history_s, latest_s)
        return earliest_s + (latest_s - earliest_s) * np.random.uniform(
            size=len(traj_idxs)
        )

    def slerp(self, val0, val1, blend):
        return (1.0 - blend) * val0 + blend * val1

    def get_trajectory(self, traj_idx):
        """Returns trajectory of AMP observations."""
        return self.trajectories_full[traj_idx]

    def get_frame_at_time(self, traj_idx, time):
        """Returns frame for the given trajectory at the specified time."""
        traj_idxs = np.array([traj_idx])
        times = np.array([time])
        return self.get_frame_at_time_batch(traj_idxs, times).squeeze(0)

    def get_frame_at_time_batch(self, traj_idxs, times):
        """Returns frame for the given trajectory at the specified time."""
        times = np.asarray(times)
        full_frame = self.get_full_frame_at_time_batch(traj_idxs, times)
        previous_full_frame = self.get_full_frame_at_time_batch(
            traj_idxs,
            times - self.time_between_frames,
        )
        return self.build_amp_observation(full_frame, previous_full_frame)

    def get_full_frame_at_time(self, traj_idx, time):
        """Returns full frame for the given trajectory at the specified time."""
        traj_idxs = np.array([traj_idx])
        times = np.array([time])
        return self.get_full_frame_at_time_batch(traj_idxs, times).squeeze(0)

    def get_full_frame_at_time_batch(self, traj_idxs, times):
        times = np.asarray(times)
        traj_lens = self.trajectory_lens[traj_idxs]
        frame_durations = self.trajectory_frame_durations[traj_idxs]
        max_times = np.maximum(traj_lens - 1.0e-6, 0.0)
        times = np.clip(times, 0.0, max_times)
        frame_pos = times / frame_durations
        n = self.trajectory_num_frames[traj_idxs].astype(np.int32)
        idx_low = np.floor(frame_pos).astype(np.int32)
        idx_low = np.clip(idx_low, 0, n - 1)
        idx_high = np.minimum(idx_low + 1, n - 1)
        all_frame_pos_starts = torch.zeros(len(traj_idxs), 3, device=self.device)
        all_frame_pos_ends = torch.zeros(len(traj_idxs), 3, device=self.device)
        all_frame_rot_starts = torch.zeros(len(traj_idxs), 4, device=self.device)
        all_frame_rot_ends = torch.zeros(len(traj_idxs), 4, device=self.device)
        all_frame_amp_starts = torch.zeros(len(traj_idxs), 29, device=self.device)
        all_frame_amp_ends = torch.zeros(len(traj_idxs),  29, device=self.device)
        for traj_idx in set(traj_idxs):
            trajectory = self.trajectories_full[traj_idx]
            traj_mask = traj_idxs == traj_idx
            all_frame_pos_starts[traj_mask] = G1_AMPLoader.get_root_pos_batch(trajectory[idx_low[traj_mask]])
            all_frame_pos_ends[traj_mask] = G1_AMPLoader.get_root_pos_batch(trajectory[idx_high[traj_mask]])
            all_frame_rot_starts[traj_mask] = G1_AMPLoader.get_root_rot_batch(trajectory[idx_low[traj_mask]])
            all_frame_rot_ends[traj_mask] = G1_AMPLoader.get_root_rot_batch(trajectory[idx_high[traj_mask]])
            all_frame_amp_starts[traj_mask] = trajectory[idx_low[traj_mask]][:, 7:36] # base vel3+ang3, dof vel23+ang23
            all_frame_amp_ends[traj_mask] = trajectory[idx_high[traj_mask]][:, 7:36]  # base vel3+ang3, dof vel23+ang23
        blend = torch.tensor(frame_pos - idx_low, device=self.device, dtype=torch.float32).unsqueeze(-1)
        pos_blend = self.slerp(all_frame_pos_starts, all_frame_pos_ends, blend)
        rot_blend = quaternion_slerp(all_frame_rot_starts, all_frame_rot_ends, blend)
        amp_blend = self.slerp(all_frame_amp_starts, all_frame_amp_ends, blend)
        return torch.cat([pos_blend, rot_blend, amp_blend], dim=-1)

    def build_amp_observation(
        self,
        full_frame: torch.Tensor,
        previous_full_frame: torch.Tensor,
    ) -> torch.Tensor:
        joint_pos = full_frame[:, 7:36]
        if self.amp_observation_mode == "joint_pos":
            return joint_pos

        joint_vel = (joint_pos - previous_full_frame[:, 7:36]) / self.time_between_frames

        root_quat = full_frame[:, 3:7]
        previous_root_quat = previous_full_frame[:, 3:7]
        root_lin_vel_w = (
            full_frame[:, :3] - previous_full_frame[:, :3]
        ) / self.time_between_frames
        root_lin_vel_b = quat_apply_inverse_wxyz(root_quat, root_lin_vel_w)
        root_ang_vel_b = angular_velocity_b_from_quat_delta(
            previous_root_quat,
            root_quat,
            self.time_between_frames,
        )
        gravity_w = torch.zeros(full_frame.shape[0], 3, device=self.device)
        gravity_w[:, 2] = -1.0
        projected_gravity_b = quat_apply_inverse_wxyz(root_quat, gravity_w)

        return torch.cat(
            (
                joint_pos,
                joint_vel,
                root_lin_vel_b,
                root_ang_vel_b,
                projected_gravity_b,
            ),
            dim=-1,
        )

    def get_frame(self):
        """Returns random frame."""
        traj_idx = self.weighted_traj_idx_sample()
        sampled_time = self.traj_time_sample(traj_idx)
        return self.get_frame_at_time(traj_idx, sampled_time)

    def get_full_frame(self):
        """Returns random full frame."""
        traj_idx = self.weighted_traj_idx_sample()
        sampled_time = self.traj_time_sample(traj_idx)
        return self.get_full_frame_at_time(traj_idx, sampled_time)

    def get_full_frame_batch(self, num_frames):
        if self.preload_transitions:
            idxs = np.random.choice(
                self.preloaded_s.shape[0], size=num_frames)
            return self.preloaded_s[idxs]
        else:
            traj_idxs = self.weighted_traj_idx_sample_batch(num_frames)
            times = self.traj_time_sample_batch(traj_idxs)
            return self.get_full_frame_at_time_batch(traj_idxs, times)

    def blend_frame_pose(self, frame0, frame1, blend):
        """Linearly interpolate between two frames, including orientation.

        Args:
            frame0: First frame to be blended corresponds to (blend = 0).
            frame1: Second frame to be blended corresponds to (blend = 1).
            blend: Float between [0, 1], specifying the interpolation between
            the two frames.
        Returns:
            An interpolation of the two frames.
        """
        root_pos0, root_pos1 = G1_AMPLoader.get_root_pos(frame0), G1_AMPLoader.get_root_pos(frame1)
        root_rot0, root_rot1 = G1_AMPLoader.get_root_rot(frame0), G1_AMPLoader.get_root_rot(frame1)
        joints0, joints1 = G1_AMPLoader.get_joint_pose(frame0), G1_AMPLoader.get_joint_pose(frame1)
        # tar_toe_pos_0, tar_toe_pos_1 = G1_AMPLoader.get_tar_toe_pos_local(frame0), G1_AMPLoader.get_tar_toe_pos_local(frame1)
        linear_vel_0, linear_vel_1 = G1_AMPLoader.get_linear_vel(frame0), G1_AMPLoader.get_linear_vel(frame1)
        angular_vel_0, angular_vel_1 = G1_AMPLoader.get_angular_vel(frame0), G1_AMPLoader.get_angular_vel(frame1)

        blend_root_pos = self.slerp(root_pos0, root_pos1, blend)
        blend_root_rot = transformations.quaternion_slerp(root_rot0.cpu().numpy(), root_rot1.cpu().numpy(), blend)
        blend_root_rot = torch.tensor(motion_util.standardize_quaternion(blend_root_rot),dtype=torch.float32, device=self.device)
        blend_joints = self.slerp(joints0, joints1, blend)
        # blend_tar_toe_pos = self.slerp(tar_toe_pos_0, tar_toe_pos_1, blend)
        blend_linear_vel = self.slerp(linear_vel_0, linear_vel_1, blend)
        blend_angular_vel = self.slerp(angular_vel_0, angular_vel_1, blend)

        return torch.cat([blend_root_pos, blend_root_rot, blend_linear_vel, blend_angular_vel, blend_joints])
    
    def feed_forward_generator_29dof_multi(self, num_mini_batch, mini_batch_size): 
        """Generates a batch of AMP transitions."""
        for _ in range(num_mini_batch):
            if self.preload_transitions:
                idxs = np.random.choice(self.preloaded_s.shape[0], size=mini_batch_size)

                frames = []
                for i in range(self.num_frames):
                    # 数据已在预加载时预处理，直接索引即可
                    s = self.preloaded_frames[i][idxs]
                    frames.append(s)
            else:
                raise NotImplementedError("AMP mini-batches require preloaded transitions.")
            yield torch.stack(frames, dim=1)

    @property
    def observation_dim(self):
        """Size of AMP observations."""
        return self._amp_observation_dim

    @staticmethod
    def _resolve_amp_observation_dim(amp_observation_mode: str) -> int:
        if amp_observation_mode == "joint_pos":
            return 29
        if amp_observation_mode == "rich":
            return 67
        raise ValueError(
            "Unsupported AMP observation mode: "
            f"{amp_observation_mode!r}. Expected 'joint_pos' or 'rich'."
        )

    @property
    def num_motions(self):
        return len(self.trajectory_names)
    @staticmethod
    def get_root_pos(pose):
        return pose[0:3]
    
    @staticmethod
    def get_root_pos_batch(poses):
        return poses[:, 0:3]

    @staticmethod
    def get_root_rot(pose):
        return pose[3:7]

    @staticmethod
    def get_root_rot_batch(poses):
        return poses[:, 3:7]

    @staticmethod
    def get_joint_pose_batch_12dof(poses):
        return poses[:, 13:25]

    @staticmethod
    def get_tar_toe_pos_local(pose):
        return pose[G1_AMPLoader.TAR_TOE_POS_LOCAL_START_IDX:G1_AMPLoader.TAR_TOE_POS_LOCAL_END_IDX]

    @staticmethod
    def get_tar_toe_pos_local_batch(poses):
        return poses[:, G1_AMPLoader.TAR_TOE_POS_LOCAL_START_IDX:G1_AMPLoader.TAR_TOE_POS_LOCAL_END_IDX]
