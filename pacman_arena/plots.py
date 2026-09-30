"""Charts from a saved headless run.

    python -m pacman_arena.plots results/<run>/ --out docs/cumulative_time.png

Cumulative wall-clock time per game step (path analysis + model call): the mean across games
as a solid line with a 95% bootstrap confidence band. A game contributes to a step only while it
is still in play, and each curve stops once fewer than --min-games games remain.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

# Categorical slots 1-3 of the validated reference palette, assigned in lineup order.
SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]
SURFACE, INK, INK_2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"


def slug(name: str) -> str:
    return name.lower().replace(":", "-").replace("/", "-").replace(" ", "-")


def cumulative_seconds(csv_path: Path) -> list[np.ndarray]:
    """One array per game: cumulative seconds after each step."""
    games: dict[tuple[str, str], list[float]] = {}
    with csv_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            games.setdefault((row["seed"], row["round"]), []).append(float(row["step_ms"]) / 1000.0)
    return [np.cumsum(steps) for steps in games.values()]


def mean_and_ci(curves: list[np.ndarray], min_games: int, samples: int = 2000, seed: int = 0):
    """Per step: mean over games still in play, and a 95% bootstrap CI of that mean."""
    longest = max(len(c) for c in curves)
    grid = np.full((len(curves), longest), np.nan)
    for i, curve in enumerate(curves):
        grid[i, :len(curve)] = curve
    alive = (~np.isnan(grid)).sum(axis=0)
    steps = np.arange(1, longest + 1)[alive >= min_games]
    rng = np.random.default_rng(seed)
    mean, low, high = [], [], []
    for column in grid[:, alive >= min_games].T:
        values = column[~np.isnan(column)]
        boot = values[rng.integers(0, len(values), size=(samples, len(values)))].mean(axis=1)
        mean.append(values.mean())
        low.append(np.percentile(boot, 2.5))
        high.append(np.percentile(boot, 97.5))
    return steps, np.array(mean), np.array(low), np.array(high)


def plot_cumulative_time(run_dir: Path, out: Path, models: list[str] | None, min_games: int) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    config = json.loads((run_dir / "config.json").read_text())
    names = [n for n in config["models"] if config["models"][n]["runtime"] != "none"]
    if models:
        names = [n for n in names if n in models]
    fig, ax = plt.subplots(figsize=(10, 5.6), dpi=160)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    ends = []
    for color, name in zip(SERIES_COLORS, names):
        curves = cumulative_seconds(run_dir / f"{slug(name)}.csv")
        steps, mean, low, high = mean_and_ci(curves, min_games)
        ax.fill_between(steps, low, high, color=color, alpha=0.5, linewidth=0)
        ax.plot(steps, mean, color=color, linewidth=2, label=f"{name}  (n = {len(curves)} games)")
        ax.plot(steps[-1], mean[-1], "o", color=color, markersize=6, markeredgecolor=SURFACE, markeredgewidth=2)
        ends.append((mean[-1], steps[-1], name))
    top = max(e[0] for e in ends)
    for value, step, name in sorted(ends):  # direct labels at each curve's end, in ink, never series color
        ax.annotate(f"{name}\n{value:.0f} s at step {step}", (step, value), xytext=(8, 0),
                    textcoords="offset points", va="center", fontsize=9, color=INK_2)
    ax.set_xlabel("game step", color=MUTED, fontsize=10)
    ax.set_ylabel("cumulative time (seconds)", color=MUTED, fontsize=10)
    ax.set_ylim(0, top * 1.08)
    ax.set_xlim(0, max(e[1] for e in ends) * 1.18)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)
    ax.set_title(f"Mean across games still in play (solid) with 95% bootstrap CI (band); curves end when "
                 f"fewer than {min_games} games remain", loc="left", fontsize=9.5, color=INK_2, pad=10)
    legend = ax.legend(loc="upper left", frameon=False, fontsize=9)
    for text in legend.get_texts():
        text.set_color(INK_2)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.suptitle("Cumulative decision time per game step", x=ax.get_position().x0, y=0.97, ha="left",
                 fontsize=14, fontweight="bold", color=INK)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor=SURFACE)
    print(f"chart -> {out}")


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m pacman_arena.plots", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", type=Path, help="a headless run folder (with config.json and <model>.csv)")
    parser.add_argument("--out", type=Path, default=None, help="PNG path (default <run_dir>/cumulative_time.png)")
    parser.add_argument("--models", help="comma-separated model names to plot (default: all model-backed)")
    parser.add_argument("--min-games", type=int, default=5, help="stop a curve when fewer games remain")
    args = parser.parse_args()
    plot_cumulative_time(args.run_dir, args.out or args.run_dir / "cumulative_time.png",
                         args.models.split(",") if args.models else None, args.min_games)


if __name__ == "__main__":
    main()
