"""Where along the clip does a tracking policy fail?

Starts one env at each of several reference frames (exact reference state, no noise, play
cfg), runs every env deterministically to the clip end, and reports per start frame: steps
survived, the frame where it failed, which termination fired, and (lift) whether the box
ended at its reference height. Mid-clip starts that succeed while early starts fail point
at a phase the policy never mastered.

  PYTHONPATH=src python src/mjlab_g1/scripts/phase_map_track.py --task G1-SMTrack-Lift \
    --learner ppo --checkpoint logs/smtrack/.../model_1000.pt --stride 25
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def main() -> None:
  ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  ap.add_argument("--task", required=True)
  ap.add_argument("--learner", choices=("ppo", "supermimic"), required=True)
  ap.add_argument("--checkpoint", required=True, type=Path)
  ap.add_argument("--stride", type=int, default=25, help="start-frame spacing")
  ap.add_argument("--device", default="cpu")
  ap.add_argument("--output", type=Path, default=None)
  args = ap.parse_args()

  from mjlab.rl import RslRlVecEnvWrapper
  from mjlab_g1.envs.smtrack_env import SMTrackEnv
  from mjlab_g1.scripts.eval_track import load_agent_cfg
  from mjlab_g1.scripts.train_track import make_runner
  from mjlab_g1.tasks.smtrack import make_env_cfg

  cfg = make_env_cfg(args.task, play=True)
  num_frames = int(__import__("numpy").load(cfg.commands["motion"].motion_file)["joint_pos"].shape[0])
  starts = list(range(0, num_frames - 1, args.stride))
  cfg.scene.num_envs = len(starts)
  cfg.auto_reset = False
  env = SMTrackEnv(cfg, device=args.device)
  vec = RslRlVecEnvWrapper(env)
  agent = load_agent_cfg(argparse.Namespace(agent_cfg=None, checkpoint=args.checkpoint, learner=args.learner, task=args.task))
  if args.learner == "supermimic":
    agent.setdefault("memory", {})["compile"] = False
  runner = make_runner(args.learner, vec, agent, None, args.device)
  runner.load(str(args.checkpoint), map_location=args.device)
  policy = runner.get_inference_policy(args.device)

  cmd, tm, dev = env.motion_command, env.termination_manager, env.device
  ids = torch.arange(len(starts), device=dev)
  env._reset_idx(ids)  # clears histories / last action / episode counters
  for i, f in enumerate(starts):
    cmd.reset_to_frame(ids[i : i + 1], f)
  env.scene.write_data_to_sim()
  env.sim.forward()
  cmd.update_relative_body_poses()

  n = len(starts)
  active = torch.ones(n, dtype=torch.bool, device=dev)
  fail_frame = torch.full((n,), -1, dtype=torch.long, device=dev)
  survived = torch.zeros(n, dtype=torch.long, device=dev)
  reason = [""] * n
  box_ok = [None] * n
  obs = vec.get_observations()
  with torch.inference_mode():
    for _ in range(num_frames - 1):
      k = env.get_ref_index().clone()
      obs, _, _, extras = vec.step(policy(obs).to(dev))
      env._manual_reset_pending[:] = False  # finished envs are frozen, never reset
      term = env.reset_terminated & active
      trunc = extras["time_outs"].to(torch.bool) & active & ~term
      survived += active.long()
      for i in torch.nonzero(term).flatten().tolist():
        fail_frame[i] = k[i]
        reason[i] = ",".join(t for t in tm.active_terms if bool(tm.get_term(t)[i]))
      if env.has_box:
        for i in torch.nonzero(trunc).flatten().tolist():
          box_z = env.get_state()[i, 73].item()
          ref_z = env.motion_lib.box_pos_w[num_frames - 1, 2].item()
          box_ok[i] = box_z >= ref_z - 0.1
      active &= ~(term | trunc)
      if not active.any():
        break

  rows = []
  print(f"{args.task} {args.learner} {args.checkpoint.parent.name}/{args.checkpoint.name}"
        f"  (clip {num_frames} frames, grasp {getattr(env.motion_lib, 'grasp_frame', None)})")
  print(f"{'start':>5s} {'survived':>8s} {'result':>8s} {'failed@frame':>12s}  reason / box")
  for i, f in enumerate(starts):
    ok = fail_frame[i] < 0
    row = {"start": f, "survived": int(survived[i]), "success": bool(ok), "fail_frame": int(fail_frame[i]),
           "reason": reason[i], "box_ok": box_ok[i]}
    rows.append(row)
    box = "" if box_ok[i] is None else ("box lifted" if box_ok[i] else "box NOT lifted")
    print(f"{f:5d} {row['survived']:8d} {'OK' if ok else 'FAIL':>8s} {row['fail_frame'] if not ok else '':>12} "
          f" {reason[i] or box}")
  frac = sum(r["success"] for r in rows) / len(rows)
  print(f"success from {sum(r['success'] for r in rows)}/{len(rows)} start frames ({100 * frac:.0f}%)")
  if args.output:
    args.output.write_text(json.dumps({"checkpoint": str(args.checkpoint), "rows": rows}, indent=1))


if __name__ == "__main__":
  main()
