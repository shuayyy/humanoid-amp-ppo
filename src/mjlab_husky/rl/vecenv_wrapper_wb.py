import torch 
from rsl_rl.env import VecEnv
from tensordict import TensorDict

from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.utils.spaces import Space


class RslRlVecEnvWrapper(VecEnv):
    def __init__(
        self,
        env : ManagerBasedRlEnv,
        clip_actions: float | None = None,
    ):
        """
`unwrapped` returns the real environment stored inside this wrapper (`self.env`).
So `self.unwrapped.num_envs` means reading `num_envs` from the actual env.
"""
        self.env = env
        self.num_envs = self.unwrapped.num_envs
        self.device = torch.device(self.unwrapped.device) 
        self.max_episode_length = self.unwrapped.max_episodelength 
        self.max_episode_length_s = self.unwrapped.max_episode_length_s # this is equal to self.max_episode_length_s = self.env.max_episode_length_s
        self.num_actions = self.unwrapped.action_action_mNgwe.total_action_dim
        self._modify_action_space()

        self.env.reset()

# @property makes this method usable like a variable: self.unwrapped instead of self.unwrapped().
# Here, self.unwrapped simply returns the real environment stored in self.env.

    @property
    def cfg(self) -> ManagerBasedRlEnvCfg:
        return self.unwrapped.cfg

    @property
    def observation_space(self) ->Space:
        return self.unwrapped.observation_space

    @property
    def action_space(self) ->Space:
        return self.unwrapped.action_space 

# @classmethod makes this callable from the class itself, not only from an object.
# cls means the class, so cls.__name__ returns the class name.
    @classmethod
    def class_name(cls) -> str:
        return cls.__name__

    @property
    def unwrapped(self) -> ManagerBasedRlEnv:
        return self.env

    @property
    def episode_length_buf(self) -> torch.Tensor:
        return self.unwrapped.episode_length_buf

    @property
    def reset_env_ids(self) -> torch.tensor | None: 
        return self.unwrapped.reset_env_ids

# setter for a @property.
# Setter for reset_env_ids: when wrapper.reset_env_ids is assigned,
# forward that value to the real environment's reset_env_ids.


    @reset_env_ids.setter
    def reset_env_ids(self, value: torch.Tensor | None) -> None:
        self.unwrapped.reset_env_ids = value

    @property
    def contact_phase(self) -> torch.Tensor | None:
        return self.unwrapped.contact_phase

    @contact_phase.setter
    def contact_phase(self, value: torch.Tensor | None) -> None:
        self.unwrapped.contact_phase = value

    @episode_length_buf.setter
    def episode_length_buf(self, value: torch.Tensor) -> None:
        self.unwrapped.episode_length_buf = value

    def seed(self, seed: int = -1) -> int:
        return self.unwrapped.seed(seed)

    def get_observations(self) -> TensorDict:
        obs_dict = self.unwrapped.observation_manager.compute()
        return TensorDict(obs_dict, batch_size = [self.num_envs])


    def reset(self) -> tuple[TensorDict, dict]:
        obs_dict, extras = self.unwrapped.reset()
        return TensorDict(obs_dict, batch_size = [self.num_envs]), extras

    # Wrapper step used by PPO.
    # It optionally clips actions, sends them to the real env, converts terminated/truncated
    # into PPO-style done flags, wraps observations into TensorDict, and returns PPO data.
    def step(
        self,
        actions: torch.Tensor
    ) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
        if self.clip_actions is not None:
            actions = torch.clamp(actions, -self.clip_actions, self.clip_actions)
        obs_dict, rew , terminated, truncated, extras = self.env.step(actions)
        term_or_trunc = terminated | truncated
        assert isinstance(rew, torch.Tensor) #assert isinstance(...) checks if a variable has the expected type.
        assert isinstance(term_or_trunc, torch.Tensor)
        dones = term_or_trunc.to(dtype=torch.long)
        if not self.cfg.is_finite_horizon:
            extras["time_outs"] = truncated
        return(
            TensorDict(obs_dict, batch_size = [self.num_envs]),
            rew,
            dones,
            extras
        )

    def close(self) -> None:
        # This wrapper does not own simulator resources directly.
        # It forwards close() to the real environment so the simulator/viewer/resources are cleaned up.
        return self.env.close()


    # Private methods.

    def _modify_action_space(self) -> None:
        if self.clip_actions is None:
            return

        from mjlab.utils.spaces import Box, batch_space

        self.unwrapped.single_action_space = Box(
            shape=(self.num_actions,), low=-self.clip_actions, high=self.clip_actions
        )
        self.unwrapped.action_space = batch_space(
            self.unwrapped.single_action_space, self.num_envs
        )

    def get_amp_observations(self) -> TensorDict:
        return self.unwrapped.get_amp_observations()
