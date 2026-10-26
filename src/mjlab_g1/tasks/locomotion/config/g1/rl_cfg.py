"""RL configuration for Unitree G1 locomotion task."""

from mjlab_g1.rl import (
  RslRlGaussianDistributionCfg,
  RslRlPpoActorCfg,
  RslRlPpoAlgorithmCfg,
  RslRlPpoCriticCfg,
)

# Keep AMP imports commented for later use.
from mjlab_g1.rl import RslRlAMPOnPolicyRunnerCfg


def unitree_g1_locomotion_ppo_runner_cfg() -> RslRlAMPOnPolicyRunnerCfg:
# def unitree_g1_locomotion_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Create RL runner configuration for Unitree G1 locomotion task."""
  return RslRlAMPOnPolicyRunnerCfg(
  # return RslRlOnPolicyRunnerCfg(
    actor=RslRlPpoActorCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
      distribution_cfg=RslRlGaussianDistributionCfg(
        init_std=1.0,
        std_type="scalar",
        learn_std=True,
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
    experiment_name="g1_locomotion",
    # Follows envs/amp_observations.py, which is shared with the dual-arm task
    # and now emits joint angles only. Not a locomotion-motivated change: the
    # velocity/gravity channels were dropped because they carried no
    # discriminative signal for the dual-arm reach (see the note there), and
    # the shared helper makes this side track it or fail the dim assert.
    amp_num_obs=29,
    amp_observation_mode="joint_pos",
    # Ramp the style coefficient over the first 5k iterations.
    amp_reward_schedule="piecewise_linear",
    amp_reward_schedule_points=((0, 0.0), (5_000, 20.0)),
    save_interval=50,
    num_steps_per_env=24,
    max_iterations=50_000,
  )
