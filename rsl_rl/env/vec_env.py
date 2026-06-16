from __future__ import annotations

import torch
from abc import ABC, abstractmethod
from tensordict import TensorDict


class VecEnv(ABC):
    num_envs: int
    num_actions: int
    max_episode_length: int | torch.Tensor
    max_episode_length_s: float

    episode_length_buf: torch.Tensor
    device: torch.device | str
    cfg: dict | object

    reset_env_ids: torch.Tensor | None = None

    @abstractmethod
    def get_observations(self) -> TensorDict:
        raise NotImplementedError

    @abstractmethod
    def get_amp_observations(self) -> TensorDict:
        raise NotImplementedError

    @abstractmethod
    def step(self, actions: torch.Tensor) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
        raise NotImplementedError
