"""Sample-efficiency figure: randomized success vs env steps, PPO-Sim vs SuperMimic.

Reads every eval_rand_model_<it>.json (eval_track.py --randomize) under the task's run dirs,
converts iterations to env steps from each run's params (steps per iteration x num_envs) and
plots one line per run, one panel per start-sampling variant (run-name suffix). Success is
box_success for the lift task (no termination and box at height), success_rate otherwise;
--metric norm_return plots the normalized return instead (tracking quality once success saturates).

  PYTHONPATH=src python src/mjlab_g1/scripts/plot_rand_success.py \
    --task G1-SMTrack-Lift --output docs/supermimic/lift_rand_success.png
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from mjlab_g1.scripts.progress_track import _num_envs

LEARNERS = {"supermimic": ("SuperMimic", "#2a78d6"), "ppo": ("PPO-Sim", "#eb6834")}
MARKERS = {"s0": "o", "s1": "s"}
VARIANTS = {
  "u50": "50/50 uniform/adaptive start sampling",
  "b2": "mjlab adaptive start sampling",
  "": "mjlab adaptive start sampling",
}
INK, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"


def env_steps_per_iter(run: Path) -> float:
  agent = json.loads((run / "params" / "agent.json").read_text())
  per_env = agent.get("env_steps_per_iter") or agent["num_steps_per_env"]
  return per_env * _num_envs(run)


def load_curves(task_dir: Path, metric: str) -> dict[tuple[str, str, str], list[tuple[float, float]]]:
  curves: dict[tuple[str, str, str], list[tuple[float, float]]] = {}
  for run in sorted(task_dir.glob("*/*")):
    if (run / "OBSOLETE").exists() or not (run / "params").exists():
      continue
    m = re.search(r"_(?:ppo|sm)_(s\d)(?:_(\w+))?$", run.name)
    if m is None or (m.group(2) or "") not in VARIANTS:
      continue
    variant = m.group(2) or ""
    per_iter = env_steps_per_iter(run)
    pts = []
    for f in run.glob("eval_rand_model_*.json"):
      it = int(re.findall(r"\d+", f.stem)[-1])
      metrics = json.loads(f.read_text())["metrics"]
      y = metrics.get("box_success", metrics["success_rate"]) if metric == "success" else metrics[metric]
      pts.append(((it + 1) * per_iter, y))
    if pts:
      curves[(run.parent.name, variant, m.group(1))] = sorted(pts)
  return curves


def place_end_labels(ax, ends: list[tuple[str, float, float]], fontsize: float = 8.0) -> None:
  """Direct labels right of each line's last point, nudged apart where they would overlap."""
  to_pt = 72.0 / ax.figure.dpi
  width = lambda text: 0.6 * fontsize * len(text)
  placed: list[tuple[float, float, float]] = []  # (x0, x1, y) in points
  for text, x, y in sorted(ends, key=lambda e: -e[2]):
    px, py = (v * to_pt for v in ax.transData.transform((x, y)))
    x0, x1, ty = px + 6, px + 6 + width(text), py
    for ox0, ox1, oy in placed:
      if x0 < ox1 and ox0 < x1 and abs(ty - oy) < 1.4 * fontsize:
        ty = oy - 1.4 * fontsize
    placed.append((x0, x1, ty))
    ax.annotate(text, (x, y), xytext=(6, ty - py), textcoords="offset points", va="center",
                fontsize=fontsize, color=INK)


def main() -> None:
  ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  ap.add_argument("--task", default="G1-SMTrack-Lift")
  ap.add_argument("--log-root", type=Path, default=Path("logs/smtrack"))
  ap.add_argument("--output", type=Path, default=Path("docs/supermimic/lift_rand_success.png"))
  ap.add_argument("--metric", choices=("success", "norm_return"), default="success")
  args = ap.parse_args()

  import matplotlib

  matplotlib.use("Agg")
  import matplotlib.pyplot as plt

  curves = load_curves(args.log_root / args.task, args.metric)
  variants = [v for v in VARIANTS if any(k[1] == v for k in curves)]
  from matplotlib.ticker import FixedLocator, NullFormatter

  fig, axes = plt.subplots(1, len(variants), figsize=(max(7.0, 5.2 * len(variants)), 3.8), sharex=True,
                           sharey=True, squeeze=False, facecolor=SURFACE)
  ends: dict[str, list[tuple[str, float, float]]] = {v: [] for v in variants}
  for ax, variant in zip(axes[0], variants):
    ax.set_facecolor(SURFACE)
    for (learner, v, seed), pts in sorted(curves.items()):
      if v != variant:
        continue
      name, color = LEARNERS[learner]
      x, y = zip(*pts)
      ax.plot(x, y, color=color, lw=2, marker=MARKERS.get(seed, "o"), ms=6,
              markeredgecolor=SURFACE, markeredgewidth=1.5, label=f"{name} {seed}")
      ends[variant].append((f"{name} {seed}", x[-1], y[-1]))
    ax.set_xscale("log")
    ax.set_xlim(7e6, 1.5e9)
    ax.xaxis.set_major_locator(FixedLocator([1e7, 3e7, 1e8, 3e8, 1e9]))
    ax.xaxis.set_major_formatter(lambda v, _: f"{v / 1e6:.0f}M" if v < 1e9 else "1B")
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_ylim(-0.02, 1.05)
    ax.set_title(VARIANTS[variant], fontsize=10, color=INK, loc="left")
    ax.set_xlabel("environment steps", fontsize=9, color=MUTED)
    ax.grid(True, which="major", color=GRID, lw=0.8)
    ax.tick_params(colors=MUTED, labelsize=8)
    for side in ("top", "right"):
      ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
      ax.spines[side].set_color(AXIS)
  ylabel = "success" if args.metric == "success" else "normalized return"
  axes[0][0].set_ylabel(f"{ylabel} (DR + noise, 64 episodes)", fontsize=9, color=MUTED)
  handles, labels = axes[0][0].get_legend_handles_labels()
  fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, fontsize=8,
             bbox_to_anchor=(0.5, -0.02))
  fig.suptitle(f"{args.task}: randomized {ylabel} vs environment steps", fontsize=11, color=INK,
               x=0.01, ha="left")
  fig.tight_layout(rect=(0, 0.06, 1, 1))
  for ax, variant in zip(axes[0], variants):  # after layout, so label spacing uses final axes
    place_end_labels(ax, ends[variant])
  args.output.parent.mkdir(parents=True, exist_ok=True)
  fig.savefig(args.output, dpi=160, facecolor=SURFACE)
  print(f"wrote {args.output} ({sum(len(p) for p in curves.values())} points, {len(curves)} runs)")


if __name__ == "__main__":
  main()
