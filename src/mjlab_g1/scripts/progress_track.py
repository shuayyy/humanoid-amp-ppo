"""One-table progress report for SMTrack runs: latest training metrics + latest eval.

Training columns come from each run's tensorboard (both learners log the env's
Episode_Termination/* and Metrics/motion/* under the same tags); eval columns from the newest
eval_model_<it>.json (deterministic: frame 0, no DR / noise).

  PYTHONPATH=src python src/mjlab_g1/scripts/progress_track.py --runs 'ppo_s*' 'sm_s*'
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import re
import time
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

FALLS = ("Episode_Termination/anchor_pos", "Episode_Termination/anchor_ori")


def _num_envs(run: Path) -> int:
  txt = (run / "params" / "env.yaml").read_text()
  return int(re.search(r"^\s+num_envs:\s*(\d+)", txt, re.M).group(1))


def train_row(run: Path, learner: str) -> dict:
  ea = EventAccumulator(str(run), size_guidance={"scalars": 0})
  ea.Reload()
  have = set(ea.Tags()["scalars"])

  def last(*tags: str) -> float | None:
    for t in tags:
      if t in have and ea.Scalars(t):
        return ea.Scalars(t)[-1].value
    return None

  def last_step(tag: str) -> int | None:
    return ea.Scalars(tag)[-1].step if tag in have and ea.Scalars(tag) else None

  it = last_step("Metrics/motion/error_body_pos")
  n = _num_envs(run)
  per_iter = 24 if learner == "ppo" else 2.5
  falls = [last(t) for t in FALLS]
  return {
    "iter": it,
    "env_steps_M": None if it is None else (it + 1) * per_iter * n / 1e6,
    "hours": (time.time() - (run / "params" / "env.yaml").stat().st_mtime) / 3600,
    "ep_reward": last("Train/mean_reward", "sim/mean_episode_return"),
    "ep_len": last("Train/mean_episode_length", "sim/mean_episode_length"),
    "norm_ret": last("sim/norm_return"),
    "body_err_cm": None if last("Metrics/motion/error_body_pos") is None else 100 * last("Metrics/motion/error_body_pos"),
    "joint_err": last("Metrics/motion/error_joint_pos"),
    "falls": None if all(f is None for f in falls) else sum(f or 0.0 for f in falls),
    "box_drops": last("Episode_Termination/box_pos"),
    "clip_ends": last("Episode_Termination/clip_end"),
  }


def eval_row(run: Path) -> dict:
  evals = sorted(run.glob("eval_model_*.json"), key=lambda p: int(re.findall(r"\d+", p.stem)[-1]))
  if not evals:
    return {}
  m = json.loads(evals[-1].read_text())["metrics"]
  return {
    "ckpt": int(re.findall(r"\d+", evals[-1].stem)[-1]),
    "success": m.get("success_rate"),
    "steps": m.get("episode_steps"),
    "e_norm_ret": m.get("norm_return"),
    "e_body_cm": None if m.get("anchor_rel_body_pos_err_mean_m") is None else 100 * m["anchor_rel_body_pos_err_mean_m"],
    "e_joint": m.get("joint_pos_err_rad"),
    "peak_qd": m.get("peak_joint_speed"),
    "box_ok": m.get("box_success"),
  }


def fmt(v: object, spec: str) -> str:
  return "—" if v is None else format(v, spec)


def main() -> None:
  ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  ap.add_argument("--log-root", default="logs/smtrack")
  ap.add_argument("--runs", nargs="*", default=["*"], help="fnmatch patterns on the run name")
  args = ap.parse_args()

  head = ("| Task | Learner | Run | It | Env steps | Hours | Ep reward | Ep length | Norm ret | "
          "Body err | Falls | Box drops || Eval ckpt | Success | Eval steps | Eval norm ret | "
          "Eval body err | Peak |q̇| | Box ok |")
  print(head)
  print("|" + "---|" * (head.count("|") - 1))
  for run in sorted(Path(args.log_root).glob("*/*/*")):
    name = run.name.split("_", 2)[-1]
    if not any(fnmatch.fnmatch(name, p) for p in args.runs) or not (run / "params").exists():
      continue
    if (run / "OBSOLETE").exists():  # superseded runs (e.g. trained on an older task version)
      continue
    learner, task = run.parent.name, run.parent.parent.name.replace("G1-SMTrack-", "")
    t, e = train_row(run, learner), eval_row(run)
    cells = [
      task, learner, name, fmt(t["iter"], "d"), fmt(t["env_steps_M"], ".1f") + " M",
      fmt(t["hours"], ".2f"), fmt(t["ep_reward"], ".2f"), fmt(t["ep_len"], ".0f"),
      fmt(t["norm_ret"], ".3f"), fmt(t["body_err_cm"], ".1f") + " cm", fmt(t["falls"], ".2f"),
      fmt(t["box_drops"], ".2f"), "", fmt(e.get("ckpt"), "d"), fmt(e.get("success"), ".2f"),
      fmt(e.get("steps"), ".0f"), fmt(e.get("e_norm_ret"), ".3f"),
      fmt(e.get("e_body_cm"), ".1f") + " cm", fmt(e.get("peak_qd"), ".1f"), fmt(e.get("box_ok"), ".2f"),
    ]
    print("| " + " | ".join(cells) + " |")


if __name__ == "__main__":
  main()
