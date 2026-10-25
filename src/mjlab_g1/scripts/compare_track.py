"""Compare PPO and SuperMimic runs on the SMTrack tasks.

Collects every `eval_model_<it>.json` (written by eval_track.py / eval_sweep.sbatch) under
logs/smtrack/<task>/<learner>/<run>/, places each checkpoint on a wall-clock axis (checkpoint
mtime minus the run's start) and an env-step axis, and writes a markdown summary plus
learning-curve plots.

  PYTHONPATH=src python src/mjlab_g1/scripts/compare_track.py --runs 'ppo_s*' 'sm_s*' \
    --out docs/supermimic/results
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

CURVE_METRICS = ("success_rate", "norm_return", "body_pos_err_mean_m", "box_success")
TABLE_METRICS = (
  "success_rate",
  "fall_rate",
  "norm_return",
  "body_pos_err_mean_m",
  "joint_pos_err_rad",
  "peak_joint_speed",
  "box_success",
  "box_pos_err_m",
  "peak_box_vz",
  "box_tilt_end_deg",
  "box_rise_5_to_55cm_s",
)


def _steps_per_iter(learner: str, agent: dict) -> float:
  if learner == "ppo":
    return float(agent.get("num_steps_per_env", 24))
  return float(agent.get("env_steps_per_iter", 2.5))


def load_run(run: Path) -> dict | None:
  evals = sorted(run.glob("eval_model_*.json"), key=lambda p: int(re.findall(r"\d+", p.stem)[-1]))
  if not evals:
    return None
  learner, task = run.parent.name, run.parent.parent.name
  agent = json.loads((run / "params" / "agent.json").read_text())
  # env.yaml carries python tags (tuples), so read scene.num_envs (first match) directly.
  env_txt = (run / "params" / "env.yaml").read_text()
  num_envs = int(re.search(r"^\s+num_envs:\s*(\d+)", env_txt, re.M).group(1))
  t0 = (run / "params" / "env.yaml").stat().st_mtime
  points = []
  for p in evals:
    it = int(re.findall(r"\d+", p.stem)[-1])
    ckpt = run / f"model_{it}.pt"
    d = json.loads(p.read_text())
    points.append(
      {
        "iteration": it,
        "hours": (ckpt.stat().st_mtime - t0) / 3600.0 if ckpt.exists() else np.nan,
        "env_steps": (it + 1) * _steps_per_iter(learner, agent) * num_envs,
        **{k: v for k, v in d["metrics"].items() if isinstance(v, (int, float)) or v is None},
      }
    )
  return {"run": run.name, "task": task, "learner": learner, "num_envs": num_envs, "points": points}


def _fmt(vals: list[float | None]) -> str:
  v = np.array([x for x in vals if x is not None and np.isfinite(x)], dtype=float)
  if v.size == 0:
    return "—"
  return f"{v.mean():.3f}" if v.size == 1 else f"{v.mean():.3f} ± {v.std():.3f}"


def summarize(runs: list[dict]) -> str:
  by = defaultdict(list)
  for r in runs:
    by[(r["task"], r["learner"])].append(r)
  lines = ["| Task | Learner | Seeds | Final it | Hours | " + " | ".join(TABLE_METRICS) + " |"]
  lines.append("|" + "---|" * (5 + len(TABLE_METRICS)))
  for (task, learner), rs in sorted(by.items()):
    finals = [r["points"][-1] for r in rs]
    row = [
      task,
      learner,
      str(len(rs)),
      _fmt([f["iteration"] for f in finals]),
      _fmt([f["hours"] for f in finals]),
    ]
    row += [_fmt([f.get(m) for f in finals]) for m in TABLE_METRICS]
    lines.append("| " + " | ".join(row) + " |")
  return "\n".join(lines)


def plot(runs: list[dict], out: Path) -> list[Path]:
  import matplotlib

  matplotlib.use("Agg")
  import matplotlib.pyplot as plt

  colors = {"ppo": "#1f77b4", "supermimic": "#d62728"}
  paths = []
  for task in sorted({r["task"] for r in runs}):
    metrics = [m for m in CURVE_METRICS if any(m in p for r in runs if r["task"] == task for p in r["points"])]
    fig, axes = plt.subplots(2, len(metrics), figsize=(4.2 * len(metrics), 7), squeeze=False)
    for j, metric in enumerate(metrics):
      for i, xkey in enumerate(("hours", "env_steps")):
        ax = axes[i][j]
        for r in runs:
          if r["task"] != task:
            continue
          xs = [p[xkey] for p in r["points"]]
          ys = [p.get(metric) for p in r["points"]]
          ax.plot(xs, ys, "-o", ms=3, color=colors.get(r["learner"], "k"), alpha=0.8,
                  label=f"{r['learner']} {r['run']}")
        ax.set_xlabel("wall-clock hours" if xkey == "hours" else "env steps")
        ax.set_title(metric)
        ax.grid(alpha=0.3)
        if xkey == "env_steps":
          ax.set_xscale("log")
    axes[0][0].legend(fontsize=7)
    fig.suptitle(task)
    fig.tight_layout()
    path = out / f"{task}_curves.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    paths.append(path)
  return paths


def main() -> None:
  ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  ap.add_argument("--log-root", default="logs/smtrack")
  ap.add_argument("--runs", nargs="*", default=["*"], help="fnmatch patterns on the run-name suffix")
  ap.add_argument("--out", default="docs/supermimic/results")
  args = ap.parse_args()

  out = Path(args.out)
  out.mkdir(parents=True, exist_ok=True)
  runs = []
  for run in sorted(Path(args.log_root).glob("*/*/*")):
    suffix = run.name.split("_", 2)[-1]  # "<date>_<time>_<run_name>"
    if not any(fnmatch.fnmatch(suffix, pat) for pat in args.runs) or (run / "OBSOLETE").exists():
      continue
    r = load_run(run)
    if r is not None:
      runs.append(r)
  if not runs:
    raise SystemExit("no evaluated runs found")
  table = summarize(runs)
  (out / "summary.md").write_text(table + "\n")
  (out / "runs.json").write_text(json.dumps(runs, indent=1, default=float))
  plots = plot(runs, out)
  print(table)
  print("plots:", *map(str, plots))


if __name__ == "__main__":
  main()
