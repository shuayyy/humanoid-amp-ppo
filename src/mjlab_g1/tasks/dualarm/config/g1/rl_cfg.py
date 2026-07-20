"""RL configuration for Unitree G1 dualarm task."""

from mjlab_g1.rl import (
  RslRlOnPolicyRunnerCfg,
  RslRlGaussianDistributionCfg,
  RslRlPpoActorCfg,
  RslRlPpoAlgorithmCfg,
  RslRlPpoCriticCfg,
)

# Keep AMP imports commented for later use.
from mjlab_g1.rl import RslRlAMPOnPolicyRunnerCfg


def unitree_g1_dualarm_ppo_runner_cfg() -> RslRlAMPOnPolicyRunnerCfg:
# def unitree_g1_dualarm_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Create RL runner configuration for Unitree G1 dualarm task."""
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
    experiment_name="g1_dualarm",
    amp_num_obs=67,
    amp_observation_mode="rich",
    # AMP is a style regularizer, not the paycheck. Task rewards are
    # dt-scaled (x0.02/step): full task success earns ~0.7/step and standing
    # ~0.1/step. The additive AMP reward is coef * 0.02 * [0..1] per step, so
    # coef 5 caps it at the standing baseline (0.1/step) and keeps the grasp
    # discovery signal dominant. (The old (40 -> 10) schedule paid more for
    # imitating the mocap than for completing the task.)
    amp_reward_coef=2.0,
    amp_reward_schedule="piecewise_linear",
    amp_reward_schedule_points=(
      (0, 5.0),
      (500, 5.0),
      (2_500, 2.0),
    ),
    # Style matters most during the reach, where task shaping is sparse and
    # the annealed coef let v5 lunge instead of squatting like the mocap.
    # Pre-lift envs use this coef; post-lift envs keep the schedule above.
    # 14 (was 8): at 8 the off-manifold reach excursion (v9-v12 pike) stayed
    # cheaper than descending on-manifold; the excursion is brief, so the
    # per-frame style price has to be steep to matter.
    amp_prelift_reward_coef=14.0,
    amp_motion_files="dataset/dualarm",
    save_interval=50,
    num_steps_per_env=24,
    max_iterations=50_000,
  )
