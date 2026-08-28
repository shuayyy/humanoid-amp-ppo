"""RL configuration for Unitree G1 dualarm task."""

from mjlab_g1.rl import (
  RslRlGaussianDistributionCfg,
  RslRlPpoActorCfg,
  RslRlPpoAlgorithmCfg,
  RslRlPpoCriticCfg,
  RslRlAMPOnPolicyRunnerCfg,
)


def unitree_g1_dualarm_ppo_runner_cfg() -> RslRlAMPOnPolicyRunnerCfg:
  """Create RL runner configuration for Unitree G1 dualarm task."""
  return RslRlAMPOnPolicyRunnerCfg(
    actor=RslRlPpoActorCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
      distribution_cfg=RslRlGaussianDistributionCfg(
        init_std=1.0,
        std_type="scalar",
        learn_std=True,
        std_range=(1e-6, 1e6),
      ),
    ),
    critic=RslRlPpoCriticCfg(
      hidden_dims=(512, 256, 128, 64),
      activation="elu",
      obs_normalization=True,
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
      class_name="rsl_rl.algorithms:AMP_PPO",
      # class_name="PPO",
    ),
    experiment_name="g1_dualarm",
    clip_actions=1.5,
    # Must stay in sync with envs/amp_observations.py.
    amp_num_obs=29,
    amp_observation_mode="joint_pos",
    amp_dataset_fps=30.0,
    amp_discr_hidden_dims=(128, 128),
    # 40.0 reproduces the released dual-arm policy (v16 R3, model_4999): the
    # coef-40 style pressure is what produces its mocap-depth squat. 25.0 was
    # the v12-era value and bootstraps contact ~2.3x sooner, but the shipped
    # checkpoint was trained at 40 — changing this invalidates reproduction.
    amp_reward_coef=40.0,
    amp_reward_schedule="constant",
    amp_prelift_reward_coef=None,
    amp_motion_files="dataset/dualarm",
    save_interval=50,
    num_steps_per_env=24,
    max_iterations=50_000,
  )
