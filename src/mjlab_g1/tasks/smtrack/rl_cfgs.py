"""PPO baseline cfg for SMTrack (brief §0): BeyondMimic hyperparameters from mjlab's
`tasks/tracking/config/g1/rl_cfg.py`, with the paper's PPO-Sim network sizes and the same
actor inputs as SuperMimic (history + goal)."""

from __future__ import annotations

from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg
from mjlab.tasks.tracking.config.g1.rl_cfg import unitree_g1_tracking_ppo_runner_cfg

ACTOR_HIDDEN = (2048, 1024, 512, 256)  # Table A.4 (PPO-Sim shares SuperMimic's sizes)
CRITIC_HIDDEN = (256, 128)
PPO_NUM_ENVS = 8192  # paper's PPO-Sim; SuperMimic uses 4096


# Qualified names: rsl_rl resolves bare names by scanning `rsl_rl.__path__`, which is the
# repo's rsl_rl/ project dir (setup.py and all) when the repo root is on sys.path
# (`python -m pytest`, `python -c` from the root).
MLP_MODEL = "rsl_rl.models:MLPModel"
# PPO-Sim shares SuperMimic's actor *architecture* (motion encoder -> tanh latent ->
# decoder), not just its widths; ACTOR_HIDDEN are the encoder/decoder widths.
ENC_DEC_MODEL = "mjlab_g1.rl.supermimic.ppo_models:EncDecModel"
GAUSSIAN = "rsl_rl.modules:GaussianDistribution"
PPO_CLASS = "rsl_rl.algorithms:PPO"
# torch.compile for the actor/critic (rsl_rl rejects CUDA-graph modes).
TORCH_COMPILE_MODE = "default"


def ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  cfg = unitree_g1_tracking_ppo_runner_cfg()
  cfg.actor = RslRlModelCfg(
    hidden_dims=ACTOR_HIDDEN,
    activation="elu",
    obs_normalization=True,
    distribution_cfg={"class_name": GAUSSIAN, "init_std": 1.0, "std_type": "scalar"},
    class_name=ENC_DEC_MODEL,
  )
  cfg.critic = RslRlModelCfg(
    hidden_dims=CRITIC_HIDDEN, activation="elu", obs_normalization=True, class_name=MLP_MODEL
  )
  cfg.algorithm.class_name = PPO_CLASS
  cfg.obs_groups = {"actor": ("actor_hist", "actor_goal"), "critic": ("critic",)}
  cfg.logger = "tensorboard"
  cfg.upload_model = False
  cfg.clip_actions = None  # tracking actions reach |a| ~ 4-20 (mocap knees, wrists)
  cfg.experiment_name = "smtrack_ppo"
  return cfg
