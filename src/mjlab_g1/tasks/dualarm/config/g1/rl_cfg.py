"""RL configuration for Unitree G1 dualarm task."""

from mjlab_g1.rl import (
  RslRlOnPolicyRunnerCfg,
  RslRlPpoActorCriticCfg,
  RslRlPpoAlgorithmCfg,
)

# Keep AMP imports commented for later use.
from mjlab_g1.rl import RslRlAMPOnPolicyRunnerCfg


def unitree_g1_dualarm_ppo_runner_cfg() -> RslRlAMPOnPolicyRunnerCfg:
# def unitree_g1_dualarm_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Create RL runner configuration for Unitree G1 dualarm task."""
  return RslRlAMPOnPolicyRunnerCfg(
  # return RslRlOnPolicyRunnerCfg(
    policy=RslRlPpoActorCriticCfg(
      init_noise_std=1.0,
      actor_obs_normalization=True,
      critic_obs_normalization=True,
      actor_hidden_dims=(512, 256, 128),
      critic_hidden_dims=(512, 256, 128, 64),
      activation="elu",
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.005,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
      class_name="AMP_PPO",
      # class_name="PPO",
    ),
    experiment_name="g1_dualarm",
    amp_reward_coef=15.0,
    amp_motion_files="dataset/dualarm",
    save_interval=50,
    num_steps_per_env=24,
    max_iterations=50_000,
  )
