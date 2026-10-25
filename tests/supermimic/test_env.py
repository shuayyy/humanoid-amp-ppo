"""SMTrack env contract (brief §2), box reference (§3) and train/eval script smoke (§5, §6).

CPU only, N=4. Run: GIT_PYTHON_REFRESH=quiet MUJOCO_GL=disable WANDB_MODE=disabled \
  PYTHONPATH=src .venv/bin/python -m pytest tests/supermimic/test_env.py -q
"""

from __future__ import annotations

import json
import os

import numpy as np
import pytest
import torch

os.environ.setdefault("MUJOCO_GL", "disable")
os.environ.setdefault("MJLAB_WARP_QUIET", "1")

from mjlab.utils.lab_api.math import (  # noqa: E402
  matrix_from_quat,
  quat_apply,
  quat_apply_inverse,
  quat_inv,
  quat_mul,
  yaw_quat,
)
from mjlab_g1.envs.smtrack_env import SMTrackEnv  # noqa: E402
from mjlab_g1.rl.supermimic import features  # noqa: E402
from mjlab_g1.tasks.smtrack import make_env_cfg  # noqa: E402
from mjlab_g1.tasks.smtrack.env_cfgs import MOTION_FILES  # noqa: E402
from mjlab_g1.tasks.smtrack.mdp import BOX_HALF_EXTENTS  # noqa: E402

# Small tensors: extra intra-op threads only add contention on a shared CPU node.
torch.set_num_threads(min(4, torch.get_num_threads()))

N = 4
DIMS = {  # actor_hist, actor_goal, critic, state, xi
  "G1-SMTrack-Walk": (930, 640, 926, 71, 33),
  "G1-SMTrack-Lift": (1020, 730, 1031, 84, 35),
}


def _build(task: str, play: bool = False, n: int = N) -> SMTrackEnv:
  cfg = make_env_cfg(task, play=play)
  cfg.scene.num_envs = n
  cfg.seed = 0
  env = SMTrackEnv(cfg, device="cpu")
  env.reset()
  return env


@pytest.fixture(scope="module", params=list(DIMS))
def env(request) -> SMTrackEnv:
  e = _build(request.param)
  e.task_id = request.param  # type: ignore[attr-defined]
  for _ in range(20):
    e.step(torch.zeros(N, 29))
  yield e
  e.close()


@pytest.fixture(scope="module")
def lift_play_env() -> SMTrackEnv:
  e = _build("G1-SMTrack-Lift", play=True, n=2)
  yield e
  e.close()


def _zero_step(env: SMTrackEnv):
  return env.step(torch.zeros(env.num_envs, 29))


def test_obs_and_state_shapes(env):
  hist, goal, critic, s_dim, xi_dim = DIMS[env.task_id]
  obs = env.observation_manager.compute()
  assert obs["actor_hist"].shape == (N, hist)
  assert obs["actor_goal"].shape == (N, goal)
  assert obs["critic"].shape == (N, critic)
  assert env.get_state().shape == (N, s_dim)
  assert env.pre_reset_state.shape == (N, s_dim)
  assert env.get_xi().shape == (N, xi_dim)
  assert env.xi_lo.shape == env.xi_hi.shape == (xi_dim,)
  assert env.has_box == (env.task_id == "G1-SMTrack-Lift")
  assert torch.isfinite(obs["actor_hist"]).all() and torch.isfinite(obs["critic"]).all()


def test_state_matches_entity_accessors(env):
  _zero_step(env)  # step() ends with forward(): derived quantities match qpos now
  s = env.get_state()
  d = env.scene["robot"].data
  origins = env.scene.env_origins
  torch.testing.assert_close(s[:, :3], d.root_link_pos_w - origins, atol=1e-5, rtol=0)
  torch.testing.assert_close(s[:, 3:7], d.root_link_quat_w, atol=1e-5, rtol=0)
  torch.testing.assert_close(s[:, 7:36], d.joint_pos, atol=1e-6, rtol=0)
  torch.testing.assert_close(s[:, 36:39], d.root_link_lin_vel_w, atol=1e-4, rtol=0)
  # Root angular velocity must be WORLD frame (MuJoCo's free-joint qvel is body-local).
  torch.testing.assert_close(s[:, 39:42], d.root_link_ang_vel_w, atol=1e-4, rtol=0)
  torch.testing.assert_close(quat_apply_inverse(s[:, 3:7], s[:, 39:42]), d.root_link_ang_vel_b, atol=1e-4, rtol=0)
  torch.testing.assert_close(s[:, 42:71], d.joint_vel, atol=1e-6, rtol=0)
  if env.has_box:
    b = env.scene["box"].data
    torch.testing.assert_close(s[:, 71:74], b.root_link_pos_w - origins, atol=1e-5, rtol=0)
    torch.testing.assert_close(s[:, 74:78], b.root_link_quat_w, atol=1e-5, rtol=0)
    torch.testing.assert_close(s[:, 78:81], b.root_link_lin_vel_w, atol=1e-4, rtol=0)
    torch.testing.assert_close(s[:, 81:84], b.root_link_ang_vel_w, atol=1e-4, rtol=0)


def test_ref_index_advances_and_reward_terms(env):
  k0 = env.get_ref_index()
  _, rew, term, trunc, _ = _zero_step(env)
  k1 = env.get_ref_index()
  alive = ~(term | trunc)
  last = env.motion_command.num_frames - 1
  assert torch.equal(k1[alive], (k0[alive] + 1).clamp(max=last))
  assert k1.dtype == torch.long and (k1 <= last).all()
  # Per-term rewards are dt-scaled and sum to the step reward.
  terms = env.get_reward_terms()
  assert terms.shape == (N, len(env.reward_term_names))
  torch.testing.assert_close(terms.sum(-1), rew, atol=1e-6, rtol=0)
  assert "self_collisions" not in env.reward_term_names
  assert env.reward_manager.get_term_cfg("joint_limit").weight == -2.0


def test_xi_within_dr_bounds(env):
  xi = env.get_xi()
  assert ((xi >= env.xi_lo - 1e-6) & (xi <= env.xi_hi + 1e-6)).all()
  assert xi[:, 0].std() > 0, "foot friction should differ across envs (mode=reset DR)"
  enc = features.encode_xi(xi, env.xi_lo, env.xi_hi)
  assert (enc.abs() <= 1.0 + 1e-5).all()
  torch.testing.assert_close(xi[:, 1:30], env.scene["robot"].data.encoder_bias)


def test_pre_reset_state_on_forced_termination(env):
  robot = env.scene["robot"]
  d = robot.data
  root = torch.cat([d.root_link_pose_w, d.root_link_vel_w], dim=-1).clone()
  root[0, 2] += 1.0  # anchor z error > 0.25 -> anchor_pos termination
  robot.write_root_state_to_sim(root[0:1], env_ids=torch.tensor([0]))
  lifted_z = float(root[0, 2] - env.scene.env_origins[0, 2])
  _, _, term, trunc, _ = _zero_step(env)
  assert term[0] and not trunc[0]
  pre, cur = env.pre_reset_state, env.get_state()
  # Terminal state kept: still ~1 m up after one 20 ms step; the env itself was reset.
  assert abs(float(pre[0, 2]) - lifted_z) < 0.05
  assert abs(float(cur[0, 2]) - float(pre[0, 2])) > 0.5
  others = ~(term | trunc)
  torch.testing.assert_close(pre[others], cur[others])


def _force_fall(env: SMTrackEnv, env_id: int) -> None:
  """Lift one env's root 1 m so anchor_pos terminates it on the next step."""
  robot = env.scene["robot"]
  d = robot.data
  root = torch.cat([d.root_link_pose_w, d.root_link_vel_w], dim=-1)[env_id : env_id + 1].clone()
  root[0, 2] += 1.0
  robot.write_root_state_to_sim(root, env_ids=torch.tensor([env_id]))


def test_auto_reset_leaves_other_envs_untouched(env):
  """B1: an env auto-resetting inside step() must not advance the others' reference twice
  or push an extra history frame (what a partial env.reset(env_ids) would do)."""
  dims = features.obs_term_dims(env.has_box)
  for victim in (0, 3):
    hist0 = features.unflatten_history(env.obs_buf["actor_hist"].clone(), 10, dims)
    k0, xi0 = env.get_ref_index(), env.get_xi()
    _force_fall(env, victim)
    _, _, term, trunc, _ = _zero_step(env)
    assert term[victim]
    hist1 = features.unflatten_history(env.obs_buf["actor_hist"], 10, dims)
    k1, xi1 = env.get_ref_index(), env.get_xi()
    others = ~(term | trunc)
    assert others.any()
    last = env.motion_command.num_frames - 1
    assert torch.equal(k1[others], (k0[others] + 1).clamp(max=last))
    # Exactly one new frame per step: the old frames shifted by one slot.
    torch.testing.assert_close(hist1[others, :-1], hist0[others, 1:], atol=0, rtol=0)
    # The reset env's history is refilled with its first post-reset frame.
    torch.testing.assert_close(
      hist1[victim], hist1[victim, -1:].expand(10, -1), atol=0, rtol=0
    )
    # Per-episode DR: resampled for the reset env only.
    torch.testing.assert_close(xi1[others], xi0[others], atol=0, rtol=0)
    assert not torch.equal(xi1[victim], xi0[victim])


def test_box_mass_and_inertia_scaled_together(env):
  if not env.has_box:
    pytest.skip("lift only")
  m, bid = env.sim.model, env._box_bid
  mass = m.body_mass[:, bid] / env.sim.get_default_field("body_mass")[bid]
  inertia = m.body_inertia[:, bid] / env.sim.get_default_field("body_inertia")[bid]
  torch.testing.assert_close(inertia, mass[:, None].expand_as(inertia), atol=1e-5, rtol=0)
  torch.testing.assert_close(env.get_xi()[:, 33], mass, atol=1e-6, rtol=0)
  assert ((mass >= 0.8 - 1e-6) & (mass <= 1.2 + 1e-6)).all() and mass.std() > 0


def test_in_hand_start_box_follows_robot_heading():
  """With training reset noise, an in-hand start puts the box where the perturbed
  robot's hands are: the reference box carried by the robot's xy + yaw offset."""
  cfg = make_env_cfg("G1-SMTrack-Lift")
  cfg.scene.num_envs = N
  cfg.seed = 3
  env = SMTrackEnv(cfg, device="cpu")
  cmd = env.motion_command
  g = cmd.grasp_frame
  starts = torch.tensor([g - 20, g, g + 100, g + 250])
  cmd._adaptive_sampling = lambda ids: cmd.time_steps.__setitem__(ids, starts[ids])  # type: ignore[method-assign]
  env.reset()
  k = env.get_ref_index() - 1  # the reset frame (the command already advanced once)
  assert torch.equal(k, starts)
  s = env.get_state()
  root_q = s[:, 3:7]
  ref_root = cmd.motion.body_pos_w[k, 0]
  dyaw = yaw_quat(quat_mul(root_q, quat_inv(cmd.motion.body_quat_w[k, 0])))
  rel = quat_apply(dyaw, cmd.box_ref_pos[k] - ref_root)
  held = k >= g
  exp_xy = torch.where(held[:, None], s[:, :2] + rel[:, :2], cmd.box_ref_pos[k, :2])
  exp_q = torch.where(held[:, None], quat_mul(dyaw, cmd.box_ref_quat[k]), cmd.box_ref_quat[k])
  torch.testing.assert_close(s[:, 71:73], exp_xy, atol=1e-5, rtol=0)
  torch.testing.assert_close(s[:, 73], cmd.box_ref_pos[k, 2], atol=1e-6, rtol=0)
  torch.testing.assert_close(s[:, 74:78], exp_q, atol=1e-5, rtol=0)
  assert torch.count_nonzero(s[:, 78:84]) == 0
  # The test only means something if the reset noise actually moved the held starts.
  shift = (s[held, 71:73] - cmd.box_ref_pos[k[held], :2]).norm(dim=-1)
  assert (shift > 1e-3).all()
  env.close()


def test_clip_end_truncates_without_wrap(env):
  from mjlab.rl import RslRlVecEnvWrapper

  vec = RslRlVecEnvWrapper(env)
  cmd = env.motion_command
  last = cmd.num_frames - 1
  ids = torch.tensor([1, 2])
  cmd.reset_to_frame(ids, last - 2)
  env.sim.forward()
  zeros = torch.zeros(N, 29)
  for step in range(3):
    k = env.get_ref_index()
    _, _, dones, extras = vec.step(zeros)
    tout = extras["time_outs"]
    if step < 2:
      assert not tout[ids].any(), f"truncated early at step {step}"
      assert torch.equal(env.get_ref_index()[ids], k[ids] + 1)
    else:
      assert (k[ids] == last).all()
      assert tout[ids].all() and dones[ids].all()
      assert not env.reset_terminated[ids].any()
  # The truncated envs restarted from a sampled frame instead of wrapping in place.
  assert (env.episode_length_buf[ids] == 0).all()


def test_goal_matches_mjlab_reference(env):
  """features.goal on env tensors == the brief §2 definition from mjlab's command data."""
  cmd = env.motion_command
  s, k = env.get_state(), env.get_ref_index()
  ks = (k[:, None] + torch.arange(10)).clamp(max=cmd.num_frames - 1)
  m = cmd.motion
  q_a = cmd.robot_anchor_quat_w[:, None].expand(N, 10, 4)
  rel = quat_mul(quat_inv(q_a), m.body_quat_w[ks, cmd.motion_anchor_body_index])
  parts = [m.joint_pos[ks], m.joint_vel[ks], matrix_from_quat(rel)[..., :2].reshape(N, 10, 6)]
  if env.has_box:
    q_p = s[:, None, 3:7].expand(N, 10, 4).contiguous()
    parts.append(quat_apply_inverse(yaw_quat(q_p), cmd.box_ref_pos[ks] - s[:, None, :3]))
    rb = quat_mul(quat_inv(q_p), cmd.box_ref_quat[ks])
    parts.append(matrix_from_quat(rb)[..., :2].reshape(N, 10, 6))
  expected = torch.cat(parts, -1).reshape(N, -1)
  goal = env.observation_manager.compute()["actor_goal"]
  torch.testing.assert_close(goal, expected, atol=2e-4, rtol=0)


def test_actor_hist_matches_features_obs_frame(lift_play_env):
  env = lift_play_env
  for _ in range(3):
    env.step(0.3 * torch.randn(env.num_envs, 29))
  hist = env.observation_manager.compute()["actor_hist"]
  frames = features.unflatten_history(hist, 10, features.obs_term_dims(True), layout="mjlab")
  s, xi = env.get_state(), env.get_xi()
  expected = features.obs_frame(s, None, env.action_manager.action, xi, has_box=True)
  torch.testing.assert_close(frames[:, -1], expected, atol=1e-4, rtol=0)


def test_play_cfg_is_nominal(lift_play_env):
  env = lift_play_env
  xi = env.get_xi()
  nominal = torch.tensor([1.0] + [0.0] * 32 + [1.0, 1.0])
  torch.testing.assert_close(xi, nominal.expand_as(xi), atol=1e-6, rtol=0)
  # xi bounds stay the training ranges so encode_xi matches training.
  assert env.xi_lo[0] == pytest.approx(0.3) and env.xi_hi[0] == pytest.approx(1.2)
  assert env.cfg.commands["motion"].sampling_mode == "start"


def test_box_placed_at_reference_on_reset(lift_play_env):
  env = lift_play_env
  cmd = env.motion_command
  g = cmd.grasp_frame
  for frame in (0, g + 50):
    ids = torch.arange(env.num_envs)
    cmd.reset_to_frame(ids, frame)
    s = env.get_state()
    torch.testing.assert_close(s[:, 71:74], cmd.box_ref_pos[frame].expand(env.num_envs, 3))
    torch.testing.assert_close(s[:, 74:78], cmd.box_ref_quat[frame].expand(env.num_envs, 4))
    assert torch.count_nonzero(s[:, 78:84]) == 0
  # Auto-reset path: box at box_ref[k_start]; the command already advanced once (mjlab).
  env.reset()
  k_start = env.get_ref_index() - 1
  torch.testing.assert_close(env.get_state()[:, 71:74], cmd.box_ref_pos[k_start])


def test_box_reference_file():
  data = np.load(MOTION_FILES["lift"])
  src = np.load(MOTION_FILES["lift"].with_name("dual_arm1.npz"))
  for key in src.files:
    np.testing.assert_array_equal(data[key], src[key])
  num_frames = data["joint_pos"].shape[0]
  g = int(data["grasp_frame"])
  assert 0 < g < num_frames // 2
  pos = data["box_pos_w"]
  np.testing.assert_allclose(data["box_half_extents"], BOX_HALF_EXTENTS, atol=1e-6)
  np.testing.assert_allclose(pos[:g, 2], BOX_HALF_EXTENTS[2], atol=1e-6)  # resting on the floor
  assert (pos[:, 2] >= BOX_HALF_EXTENTS[2] - 1e-6).all()
  assert np.abs(np.diff(pos, axis=0)).max() < 0.05  # continuous (no jump at the grasp)
  assert pos[-1, 2] > 0.6  # lifted
  np.testing.assert_allclose(np.linalg.norm(data["box_quat_w"], axis=-1), 1.0, atol=1e-5)
  for key in ("box_lin_vel_w", "box_ang_vel_w"):
    assert data[key].shape == (num_frames, 3)


def test_amp_registry_unaffected():
  from mjlab_g1.tasks.registry import list_tasks

  tasks = list_tasks()
  assert "Mjlab-G1-DualArm" in tasks and not any("SMTrack" in t for t in tasks)


@pytest.mark.parametrize("learner", ["ppo", "supermimic"])
@pytest.mark.parametrize("task", list(DIMS))
def test_train_and_eval_smoke(task, learner, tmp_path):
  from mjlab_g1.scripts import eval_track, train_track

  log_dir = train_track.main(
    [task, "--learner", learner, "--smoke", "--device", "cpu", "--log-root", str(tmp_path)]
  )
  ckpts = sorted(log_dir.glob("model_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
  assert ckpts and (log_dir / "params" / "agent.json").exists()
  if learner == "supermimic":
    # Rewards are scored on refreshed kinematics, so the analytic reward is exact.
    first = json.loads((log_dir / "metrics.jsonl").read_text().splitlines()[0])
    assert first["parity/r_maxabs"] < 1e-5
  out = tmp_path / "eval.json"
  eval_track.main(
    ["--task", task, "--learner", learner, "--checkpoint", str(ckpts[-1]), "--episodes", "2",
     "--device", "cpu", "--output", str(out)]
  )
  metrics = json.loads(out.read_text())["metrics"]
  for key in ("success_rate", "fall_rate", "norm_return", "body_pos_err_mean_m", "joint_pos_err_rad"):
    assert metrics[key] is not None
  if task == "G1-SMTrack-Lift":
    assert "box_success" in metrics and "box_pos_err_m" in metrics
