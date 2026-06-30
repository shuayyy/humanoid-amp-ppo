from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv as ManagerBasedRlEnv
from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnvCfg as ManagerBasedRlEnvCfg
from mjlab.envs.types import VecEnvObs as VecEnvObs
from mjlab.envs.types import VecEnvStepReturn as VecEnvStepReturn

__all__ = [
  "ManagerBasedRlEnv",
  "ManagerBasedRlEnvCfg",
  "VecEnvObs",
  "VecEnvStepReturn",
  "G1LocomotionManagerBasedRlEnvCfg",
  "G1LocomotionManagerBasedRlEnv",
  "G1DualarmManagerBasedRlEnvCfg",
  "G1DualarmManagerBasedRlEnv",
]


def __getattr__(name: str):
  if name in {
    "G1LocomotionManagerBasedRlEnvCfg",
    "G1LocomotionManagerBasedRlEnv",
  }:
    from mjlab_g1.envs.g1_locomotion_rl_env import (
      G1LocomotionManagerBasedRlEnv,
      G1LocomotionManagerBasedRlEnvCfg,
    )

    values = {
      "G1LocomotionManagerBasedRlEnvCfg": G1LocomotionManagerBasedRlEnvCfg,
      "G1LocomotionManagerBasedRlEnv": G1LocomotionManagerBasedRlEnv,
    }
    return values[name]

  if name in {
    "G1DualarmManagerBasedRlEnvCfg",
    "G1DualarmManagerBasedRlEnv",
  }:
    from mjlab_g1.envs.g1_dualarm_rl_env import (
      G1DualarmManagerBasedRlEnv,
      G1DualarmManagerBasedRlEnvCfg,
    )

    values = {
      "G1DualarmManagerBasedRlEnvCfg": G1DualarmManagerBasedRlEnvCfg,
      "G1DualarmManagerBasedRlEnv": G1DualarmManagerBasedRlEnv,
    }
    return values[name]

  raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
