# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from itertools import chain

import numpy as np
import torch
import torch.nn as nn
from tensordict import TensorDict

from rsl_rl.algorithms.ppo import PPO
from rsl_rl.env import VecEnv
from rsl_rl.extensions import resolve_rnd_config, resolve_symmetry_config
from rsl_rl.models import MLPModel
from rsl_rl.modules.discriminator_multi import DiscriminatorMulti
from rsl_rl.storage import RolloutStorage
from rsl_rl.storage.replay_buffer_multi import ReplayBufferMulti
from rsl_rl.utils import compile_model, resolve_callable, resolve_obs_groups, resolve_optimizer
from rsl_rl.utils.motion_loader_g1 import G1_AMPLoader


class RunningMeanStd:
    """Running mean/std tracker used by the AMP discriminator normalizer."""

    def __init__(self, epsilon: float = 1e-4, shape: int | tuple[int, ...] = ()) -> None:
        self.mean = np.zeros(shape, np.float64)
        self.var = np.ones(shape, np.float64)
        self.count = epsilon

    def update(self, arr: np.ndarray) -> None:
        batch_mean = np.mean(arr, axis=0)
        batch_var = np.var(arr, axis=0)
        batch_count = arr.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean: np.ndarray, batch_var: np.ndarray, batch_count: int) -> None:
        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + np.square(delta) * self.count * batch_count / total_count
        new_var = m_2 / total_count

        self.mean = new_mean
        self.var = new_var
        self.count = total_count


class Normalizer(RunningMeanStd):
    """Numpy/Torch normalizer kept compatible with the old AMP implementation."""

    def __init__(self, input_dim: int, epsilon: float = 1e-4, clip_obs: float = 10.0) -> None:
        super().__init__(shape=input_dim)
        self.epsilon = epsilon
        self.clip_obs = clip_obs

    def normalize(self, input_array: np.ndarray) -> np.ndarray:
        return np.clip(
            (input_array - self.mean) / np.sqrt(self.var + self.epsilon),
            -self.clip_obs,
            self.clip_obs,
        )

    def normalize_torch(self, input_tensor: torch.Tensor, device: str | torch.device) -> torch.Tensor:
        mean = torch.tensor(self.mean, device=device, dtype=torch.float32)
        std = torch.sqrt(torch.tensor(self.var + self.epsilon, device=device, dtype=torch.float32))
        return torch.clamp((input_tensor - mean) / std, -self.clip_obs, self.clip_obs)


class AMP_PPO(PPO):
    """PPO with the G1 AMP discriminator reward ported to the RSL-RL 5.2 API."""

    def __init__(
        self,
        actor: MLPModel,
        critic: MLPModel,
        storage: RolloutStorage,
        discriminator: DiscriminatorMulti,
        amp_data: G1_AMPLoader,
        amp_normalizer: Normalizer,
        amp_num_frames: int = 1,
        amp_replay_buffer_size: int = 100000,
        num_learning_epochs: int = 5,
        num_mini_batches: int = 4,
        clip_param: float = 0.2,
        gamma: float = 0.99,
        lam: float = 0.95,
        value_loss_coef: float = 1.0,
        entropy_coef: float = 0.01,
        learning_rate: float = 0.001,
        max_grad_norm: float = 1.0,
        optimizer: str = "adam",
        use_clipped_value_loss: bool = True,
        schedule: str = "adaptive",
        desired_kl: float = 0.01,
        normalize_advantage_per_mini_batch: bool = False,
        device: str = "cpu",
        rnd_cfg: dict | None = None,
        symmetry_cfg: dict | None = None,
        multi_gpu_cfg: dict | None = None,
    ) -> None:
        super().__init__(
            actor=actor,
            critic=critic,
            storage=storage,
            num_learning_epochs=num_learning_epochs,
            num_mini_batches=num_mini_batches,
            clip_param=clip_param,
            gamma=gamma,
            lam=lam,
            value_loss_coef=value_loss_coef,
            entropy_coef=entropy_coef,
            learning_rate=learning_rate,
            max_grad_norm=max_grad_norm,
            optimizer=optimizer,
            use_clipped_value_loss=use_clipped_value_loss,
            schedule=schedule,
            desired_kl=desired_kl,
            normalize_advantage_per_mini_batch=normalize_advantage_per_mini_batch,
            device=device,
            rnd_cfg=rnd_cfg,
            symmetry_cfg=symmetry_cfg,
            multi_gpu_cfg=multi_gpu_cfg,
        )

        self.discriminator = discriminator.to(self.device)
        self.amp_num_frames = amp_num_frames
        self.amp_observation_dim = discriminator.state_dim
        self.amp_storage = ReplayBufferMulti(
            discriminator.state_dim,
            amp_replay_buffer_size,
            amp_num_frames,
            self.device,
        )
        self.amp_data = amp_data
        self.amp_normalizer = amp_normalizer

        optimizer_class = resolve_optimizer(optimizer)
        self.optimizer = optimizer_class(
            [
                {
                    "params": list(chain(self.actor.parameters(), self.critic.parameters())),
                    "name": "policy",
                },
                {
                    "params": self.discriminator.trunk.parameters(),
                    "weight_decay": 10e-4,
                    "name": "amp_trunk",
                },
                {
                    "params": self.discriminator.amp_linear.parameters(),
                    "weight_decay": 10e-2,
                    "name": "amp_head",
                },
            ],
            lr=learning_rate,
        )

    def act(self, obs: TensorDict, amp_obs: torch.Tensor | None = None) -> torch.Tensor:
        del amp_obs
        return super().act(obs)

    def process_env_step(
        self,
        obs: TensorDict,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        extras: dict[str, torch.Tensor],
        amp_obs: torch.Tensor | None = None,
        amp_obs_frames: torch.Tensor | None = None,
    ) -> None:
        self.actor.update_normalization(obs)
        self.critic.update_normalization(obs)
        if self.rnd:
            self.rnd.update_normalization(obs)

        self.transition.rewards = rewards.clone()
        self.transition.dones = dones

        if self.rnd:
            self.intrinsic_rewards = self.rnd.get_intrinsic_reward(obs)
            self.transition.rewards += self.intrinsic_rewards

        if "time_outs" in extras:
            self.transition.rewards += self.gamma * torch.squeeze(
                self.transition.values * extras["time_outs"].unsqueeze(1).to(self.device),  # type: ignore
                1,
            )

        if amp_obs_frames is not None:
            self.amp_storage.insert(amp_obs_frames.detach())
        elif amp_obs is not None:
            amp_obs_frames = amp_obs.unsqueeze(1) if amp_obs.dim() == 2 else amp_obs
            if amp_obs_frames.shape[1] != self.amp_num_frames:
                raise RuntimeError(
                    "Expected AMP observations with "
                    f"{self.amp_num_frames} frames, got {amp_obs_frames.shape[1]}."
                )
            self.amp_storage.insert(amp_obs_frames.detach())

        self.storage.add_transition(self.transition)
        self.transition.clear()
        self.actor.reset(dones)
        self.critic.reset(dones)

    def update(self) -> dict[str, float]:  # noqa: C901
        mean_value_loss = 0.0
        mean_surrogate_loss = 0.0
        mean_entropy = 0.0
        mean_amp_loss = 0.0
        mean_grad_pen_loss = 0.0
        mean_policy_pred = 0.0
        mean_expert_pred = 0.0
        mean_rnd_loss = 0.0 if self.rnd else None
        mean_symmetry_loss = 0.0 if self.symmetry else None

        if self.actor.is_recurrent or self.critic.is_recurrent:
            generator = self.storage.recurrent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        mini_batch_size = self.storage.num_envs * self.storage.num_transitions_per_env // self.num_mini_batches
        num_updates = self.num_learning_epochs * self.num_mini_batches
        amp_policy_generator = self.amp_storage.feed_forward_generator(num_updates, mini_batch_size)
        amp_expert_generator = self.amp_data.feed_forward_generator_29dof_multi(num_updates, mini_batch_size)

        for batch, sample_amp_policy, sample_amp_expert in zip(generator, amp_policy_generator, amp_expert_generator):
            original_batch_size = batch.observations.batch_size[0]

            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    batch.advantages = (batch.advantages - batch.advantages.mean()) / (batch.advantages.std() + 1e-8)  # type: ignore

            if self.symmetry:
                self.symmetry.augment_batch(batch, original_batch_size)

            self.actor(
                batch.observations,
                masks=batch.masks,
                hidden_state=batch.hidden_states[0],
                stochastic_output=True,
            )
            actions_log_prob = self.actor.get_output_log_prob(batch.actions)  # type: ignore
            values = self.critic(batch.observations, masks=batch.masks, hidden_state=batch.hidden_states[1])
            distribution_params = tuple(p[:original_batch_size] for p in self.actor.output_distribution_params)
            entropy = self.actor.output_entropy[:original_batch_size]

            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = self.actor.get_kl_divergence(batch.old_distribution_params, distribution_params)  # type: ignore
                    kl_mean = torch.mean(kl)

                    if self.is_multi_gpu:
                        torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
                        kl_mean /= self.gpu_world_size

                    if self.gpu_global_rank == 0:
                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                    if self.is_multi_gpu:
                        lr_tensor = torch.tensor(self.learning_rate, device=self.device)
                        torch.distributed.broadcast(lr_tensor, src=0)
                        self.learning_rate = lr_tensor.item()

                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            ratio = torch.exp(actions_log_prob - torch.squeeze(batch.old_actions_log_prob))  # type: ignore
            surrogate = -torch.squeeze(batch.advantages) * ratio  # type: ignore
            surrogate_clipped = -torch.squeeze(batch.advantages) * torch.clamp(  # type: ignore
                ratio,
                1.0 - self.clip_param,
                1.0 + self.clip_param,
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            if self.use_clipped_value_loss:
                value_clipped = batch.values + (values - batch.values).clamp(-self.clip_param, self.clip_param)
                value_losses = (values - batch.returns).pow(2)
                value_losses_clipped = (value_clipped - batch.returns).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (batch.returns - values).pow(2).mean()

            loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy.mean()

            rnd_loss = self.rnd.compute_loss(batch.observations[:original_batch_size]) if self.rnd else None  # type: ignore

            if self.symmetry:
                symmetry_loss = self.symmetry.compute_loss(self.actor, batch, original_batch_size)
                if self.symmetry.use_mirror_loss:
                    loss = loss + self.symmetry.mirror_loss_coeff * symmetry_loss

            expert_states = sample_amp_expert.to(self.device)
            policy_states = sample_amp_policy.to(self.device)

            with torch.no_grad():
                expert_states = self.amp_normalizer.normalize_torch(expert_states, self.device)
                policy_states = self.amp_normalizer.normalize_torch(policy_states, self.device)

            policy_d = self.discriminator(policy_states.flatten(1))
            expert_d = self.discriminator(expert_states.flatten(1))

            expert_loss = nn.functional.mse_loss(expert_d, torch.ones_like(expert_d))
            policy_loss = nn.functional.mse_loss(policy_d, -torch.ones_like(policy_d))
            amp_loss = 0.5 * (expert_loss + policy_loss)
            grad_pen_loss = self.discriminator.compute_grad_pen(expert_states, lambda_=5)

            loss = loss + amp_loss + grad_pen_loss
            self.amp_normalizer.update(policy_states.detach().cpu().numpy())
            self.amp_normalizer.update(expert_states.detach().cpu().numpy())

            self.optimizer.zero_grad()
            loss.backward()
            if self.rnd:
                self.rnd.optimizer.zero_grad()
                rnd_loss.backward()

            if self.is_multi_gpu:
                self.reduce_parameters()

            nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.optimizer.step()
            if self.rnd:
                self.rnd.optimizer.step()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy.mean().item()
            mean_amp_loss += amp_loss.item()
            mean_grad_pen_loss += grad_pen_loss.item()
            mean_policy_pred += policy_loss.mean().item()
            mean_expert_pred += expert_loss.mean().item()
            if mean_rnd_loss is not None:
                mean_rnd_loss += rnd_loss.item()
            if mean_symmetry_loss is not None:
                mean_symmetry_loss += symmetry_loss.item()

        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy /= num_updates
        mean_amp_loss /= num_updates
        mean_grad_pen_loss /= num_updates
        mean_policy_pred /= num_updates
        mean_expert_pred /= num_updates
        if mean_rnd_loss is not None:
            mean_rnd_loss /= num_updates
        if mean_symmetry_loss is not None:
            mean_symmetry_loss /= num_updates

        loss_dict = {
            "value_function": mean_value_loss,
            "surrogate": mean_surrogate_loss,
            "entropy": mean_entropy,
            "amp": mean_amp_loss,
            "amp_grad_pen": mean_grad_pen_loss,
            "amp_policy_pred": mean_policy_pred,
            "amp_expert_pred": mean_expert_pred,
        }
        if self.rnd:
            loss_dict["rnd"] = mean_rnd_loss
        if self.symmetry:
            loss_dict["symmetry"] = mean_symmetry_loss

        self.storage.clear()
        return loss_dict

    def train_mode(self) -> None:
        super().train_mode()
        self.discriminator.train()

    def eval_mode(self) -> None:
        super().eval_mode()
        self.discriminator.eval()

    def compile(self, mode: str | None = None) -> None:
        self.actor = compile_model(self._raw_actor, mode)  # type: ignore
        self.critic = compile_model(self._raw_critic, mode)  # type: ignore
        self.discriminator = compile_model(self.discriminator, mode)  # type: ignore

    def save(self) -> dict:
        saved_dict = super().save()
        saved_dict["discriminator_state_dict"] = self.discriminator.state_dict()
        saved_dict["amp_normalizer_mean"] = self.amp_normalizer.mean
        saved_dict["amp_normalizer_var"] = self.amp_normalizer.var
        saved_dict["amp_normalizer_count"] = self.amp_normalizer.count
        return saved_dict

    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
        should_load_iteration = super().load(loaded_dict, load_cfg, strict)
        if load_cfg is None or load_cfg.get("discriminator", True):
            if "discriminator_state_dict" in loaded_dict:
                self.discriminator.load_state_dict(loaded_dict["discriminator_state_dict"], strict=strict)
        if "amp_normalizer_mean" in loaded_dict:
            self.amp_normalizer.mean = loaded_dict["amp_normalizer_mean"]
            self.amp_normalizer.var = loaded_dict["amp_normalizer_var"]
            self.amp_normalizer.count = loaded_dict["amp_normalizer_count"]
        return should_load_iteration

    @staticmethod
    def construct_algorithm(obs: TensorDict, env: VecEnv, cfg: dict, device: str) -> "AMP_PPO":
        """Construct AMP-PPO from native RSL-RL 5.2 actor/critic config."""
        cfg["algorithm"].setdefault("rnd_cfg", None)
        cfg["algorithm"].setdefault("symmetry_cfg", None)
        cfg["algorithm"].setdefault("share_cnn_encoders", None)

        alg_cfg = dict(cfg["algorithm"])
        actor_cfg = dict(cfg["actor"])
        critic_cfg = dict(cfg["critic"])

        alg_class: type[AMP_PPO] = resolve_callable(alg_cfg.pop("class_name"))  # type: ignore
        actor_class: type[MLPModel] = resolve_callable(actor_cfg.pop("class_name"))  # type: ignore
        critic_class: type[MLPModel] = resolve_callable(critic_cfg.pop("class_name"))  # type: ignore

        default_sets = ["actor", "critic"]
        if alg_cfg["rnd_cfg"] is not None:
            default_sets.append("rnd_state")
        cfg["obs_groups"] = resolve_obs_groups(obs, cfg["obs_groups"], default_sets)

        alg_cfg = resolve_rnd_config(alg_cfg, obs, cfg["obs_groups"], env)
        alg_cfg = resolve_symmetry_config(alg_cfg, env)

        actor: MLPModel = actor_class(obs, cfg["obs_groups"], "actor", env.num_actions, **actor_cfg).to(device)
        print(f"Actor Model: {actor}")
        if alg_cfg.pop("share_cnn_encoders", None):
            critic_cfg["cnns"] = actor.cnns  # type: ignore
        critic: MLPModel = critic_class(obs, cfg["obs_groups"], "critic", 1, **critic_cfg).to(device)
        print(f"Critic Model: {critic}")

        amp_data = G1_AMPLoader(
            device,
            time_between_frames=1 / 50.0,
            preload_transitions=True,
            num_preload_transitions=cfg["amp_num_preload_transitions"],
            motion_files=cfg["amp_motion_files"],
            num_frames=cfg["amp_num_frames"],
            amp_observation_mode=cfg.get("amp_observation_mode", "joint_pos"),
        )
        amp_observation_dim = amp_data.observation_dim if cfg["amp_num_obs"] == 0 else cfg["amp_num_obs"]
        if amp_observation_dim != amp_data.observation_dim:
            raise ValueError(
                "AMP observation dimension mismatch: "
                f"config requested {amp_observation_dim}, "
                f"but motion loader produces {amp_data.observation_dim} "
                f"for mode {cfg.get('amp_observation_mode', 'joint_pos')!r}."
            )
        amp_num_frames = 1 if cfg["amp_num_frames"] == 0 else cfg["amp_num_frames"]
        amp_normalizer = Normalizer(amp_observation_dim)
        discriminator = DiscriminatorMulti(
            amp_observation_dim,
            cfg["amp_reward_coef"],
            cfg["amp_discr_hidden_dims"],
            device,
            amp_num_frames,
            cfg["amp_task_reward_lerp"],
            cfg["use_lerp"],
            cfg.get("amp_reward_additive_scale", 0.02),
        ).to(device)

        storage = RolloutStorage("rl", env.num_envs, cfg["num_steps_per_env"], obs, [env.num_actions], device)
        alg = alg_class(
            actor,
            critic,
            storage,
            discriminator,
            amp_data,
            amp_normalizer,
            amp_num_frames,
            device=device,
            **alg_cfg,
            multi_gpu_cfg=cfg["multi_gpu"],
        )
        alg.compile(cfg.get("torch_compile_mode"))
        return alg

    def broadcast_parameters(self) -> None:
        model_params = [
            self._raw_actor.state_dict(),
            self._raw_critic.state_dict(),
            self.discriminator.state_dict(),
        ]
        if self.rnd:
            model_params.append(self.rnd.predictor.state_dict())
        torch.distributed.broadcast_object_list(model_params, src=0)
        self._raw_actor.load_state_dict(model_params[0])
        self._raw_critic.load_state_dict(model_params[1])
        self.discriminator.load_state_dict(model_params[2])
        if self.rnd:
            self.rnd.predictor.load_state_dict(model_params[3])

    def reduce_parameters(self) -> None:
        all_params = chain(self.actor.parameters(), self.critic.parameters(), self.discriminator.parameters())
        if self.rnd:
            all_params = chain(all_params, self.rnd.parameters())
        all_params = list(all_params)
        grads = [param.grad.view(-1) for param in all_params if param.grad is not None]
        all_grads = torch.cat(grads)
        torch.distributed.all_reduce(all_grads, op=torch.distributed.ReduceOp.SUM)
        all_grads /= self.gpu_world_size

        offset = 0
        for param in all_params:
            if param.grad is not None:
                numel = param.numel()
                param.grad.data.copy_(all_grads[offset : offset + numel].view_as(param.grad.data))
                offset += numel
