# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os
import time

import torch

from rsl_rl.env import VecEnv
from rsl_rl.runners.on_policy_runner import OnPolicyRunner
from rsl_rl.utils import check_nan


class AMPOnPolicyRunner(OnPolicyRunner):
    """RSL-RL 5.2 on-policy runner with G1 AMP rollout handling."""

    def __init__(self, env: VecEnv, train_cfg: dict, log_dir: str | None = None, device: str = "cpu") -> None:
        train_cfg["algorithm"].setdefault("rnd_cfg", None)
        train_cfg["algorithm"].setdefault("symmetry_cfg", None)
        train_cfg["algorithm"].setdefault("share_cnn_encoders", None)
        train_cfg.setdefault("torch_compile_mode", None)
        super().__init__(env, train_cfg, log_dir, device)
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]
        self.logger_type = self.cfg.get("logger", "tensorboard").lower()
        self.disable_logs = self.logger.disable_logs

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False) -> None:  # noqa: C901
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf,
                high=int(self.env.max_episode_length),
            )

        obs = self.env.get_observations().to(self.device)
        amp_obs = self.env.get_amp_observations().to(self.device)
        self.alg.train_mode()

        if self.is_distributed:
            print(f"Synchronizing parameters for rank {self.gpu_global_rank}...")
            self.alg.broadcast_parameters()

        self.logger.init_logging_writer()
        if hasattr(self.logger, "logger_type"):
            self.logger_type = self.logger.logger_type
        self.writer = self.logger.writer

        amp_obs_frames = torch.zeros(
            self.env.num_envs,
            self.alg.amp_num_frames,
            self.alg.amp_observation_dim,
            device=self.device,
        )
        amp_obs_frames = torch.cat((amp_obs_frames[:, 1:], amp_obs.unsqueeze(1)), dim=1)

        start_it = self.current_learning_iteration
        total_it = start_it + num_learning_iterations
        for it in range(start_it, total_it):
            start = time.time()
            with torch.inference_mode():
                for _ in range(self.cfg["num_steps_per_env"]):
                    actions = self.alg.act(obs, amp_obs)
                    obs, rewards, dones, extras = self.env.step(actions.to(self.env.device))

                    if self.cfg.get("check_for_nan", True):
                        check_nan(obs, rewards, dones)

                    obs, rewards, dones = (
                        obs.to(self.device),
                        rewards.to(self.device),
                        dones.to(self.device),
                    )

                    next_amp_obs = self.env.get_amp_observations().to(self.device)
                    next_amp_obs_with_term = next_amp_obs.clone()
                    reset_env_ids = self.env.reset_env_ids
                    if reset_env_ids is not None and len(reset_env_ids) > 0:
                        terminal_amp_states = self.env.get_amp_observations().to(self.device)[reset_env_ids]
                        next_amp_obs_with_term[reset_env_ids] = terminal_amp_states

                    amp_obs_frames = torch.cat(
                        (amp_obs_frames[:, 1:], next_amp_obs_with_term.unsqueeze(1)),
                        dim=1,
                    )
                    rewards, _, _ = self.alg.discriminator.predict_amp_reward(
                        amp_obs_frames,
                        rewards,
                        normalizer=self.alg.amp_normalizer,
                    )

                    self.alg.process_env_step(
                        obs,
                        rewards,
                        dones,
                        extras,
                        next_amp_obs_with_term,
                        amp_obs_frames,
                    )

                    if reset_env_ids is not None and len(reset_env_ids) > 0:
                        amp_obs_frames[reset_env_ids] = 0

                    amp_obs = next_amp_obs.clone()
                    intrinsic_rewards = self.alg.intrinsic_rewards if self.cfg["algorithm"]["rnd_cfg"] else None
                    self.logger.process_env_step(rewards, dones, extras, intrinsic_rewards)

                stop = time.time()
                collect_time = stop - start
                start = stop

                self.alg.compute_returns(obs)

            loss_dict = self.alg.update()

            stop = time.time()
            learn_time = stop - start
            self.current_learning_iteration = it

            self.logger.log(
                it=it,
                start_it=start_it,
                total_it=total_it,
                collect_time=collect_time,
                learn_time=learn_time,
                loss_dict=loss_dict,
                learning_rate=self.alg.learning_rate,
                action_std=self.alg.get_policy().output_std,
                rnd_weight=self.alg.rnd.weight if self.cfg["algorithm"]["rnd_cfg"] else None,
            )

            if self.logger.writer is not None and it % self.cfg["save_interval"] == 0:
                self.save(os.path.join(self.logger.log_dir, f"model_{it}.pt"))  # type: ignore

        if self.logger.writer is not None:
            self.save(os.path.join(self.logger.log_dir, f"model_{self.current_learning_iteration}.pt"))  # type: ignore
            self.logger.stop_logging_writer()
