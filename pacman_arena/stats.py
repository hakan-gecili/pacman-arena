"""Per-step decision logs and per-model summary metrics."""
from __future__ import annotations

import csv
import json
import random
import statistics
from pathlib import Path
from typing import Any

from .game import Game

LOG_FIELDS = ["seed", "round", "step", "replan", "letter", "chosen_path", "move", "probabilities", "options",
              "greedy_agrees", "invalid", "pellet_eaten", "life_lost", "score", "lives", "latency_ms",
              "step_ms", "input_tokens", "prompt"]


def log_row(seed: int, round_index: int, game: Game, d: dict[str, Any], ate: bool, died: bool) -> dict[str, Any]:
    """One row per step. Model fields are filled only on steps where the model was asked."""
    asked = bool(d["replan"])
    return {
        "seed": seed, "round": round_index, "step": game.steps, "replan": d["replan"] or "",
        "letter": d["letter"] if asked else "", "chosen_path": d.get("chosen_path", "") if asked else "",
        "move": d["move"],
        "probabilities": json.dumps({k: round(v, 4) for k, v in d["probabilities"].items()}) if asked else "",
        "options": json.dumps([{k: v for k, v in o.items() if k != "cells"} for o in d["options"]],
                              separators=(",", ":")) if asked else "",
        "greedy_agrees": "" if d["greedy_agrees"] is None else d["greedy_agrees"], "invalid": d["invalid"],
        "pellet_eaten": int(ate), "life_lost": int(died), "score": game.score, "lives": game.lives,
        "latency_ms": round(d["latency_ms"], 3), "step_ms": round(d.get("step_ms", 0.0), 3),
        "input_tokens": d["input_tokens"],
        "prompt": d["prompt"] if asked else "",
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=LOG_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    return values[min(len(values) - 1, round((len(values) - 1) * p))]


def bootstrap_ci(values: list[float], seed: int, samples: int = 1000) -> list[float]:
    """95% bootstrap confidence interval of the mean."""
    if not values:
        return [0.0, 0.0]
    rng = random.Random(seed)
    means = [statistics.fmean(rng.choice(values) for _ in values) for _ in range(samples)]
    return [round(percentile(means, 0.025), 2), round(percentile(means, 0.975), 2)]


def summarize(games: list[Game], rows: list[dict[str, Any]], seed: int) -> dict[str, Any]:
    decisions = [r for r in rows if r["replan"]]
    latencies = [float(r["latency_ms"]) for r in decisions]
    agrees = [r["greedy_agrees"] for r in decisions if r["greedy_agrees"] != ""]
    scores = [float(g.score) for g in games]
    game_seconds: dict[tuple[str, str], float] = {}
    for r in rows:
        key = (str(r["seed"]), str(r["round"]))
        game_seconds[key] = game_seconds.get(key, 0.0) + float(r["step_ms"]) / 1000.0
    return {
        "games": len(games),
        "mean_score": round(statistics.fmean(scores), 2),
        "mean_score_95ci": bootstrap_ci(scores, seed),
        "board_clear_rate": round(sum(g.cleared for g in games) / len(games), 4),
        "mean_lives_left": round(statistics.fmean(g.lives for g in games), 3),
        "lives_lost_per_100_steps": round(statistics.fmean(100.0 * g.lives_lost / max(1, g.steps) for g in games), 4),
        "mean_steps": round(statistics.fmean(g.steps for g in games), 1),
        "mean_game_seconds": round(statistics.fmean(game_seconds.values()), 2) if game_seconds else 0.0,
        "decisions": len(decisions),
        "decision_latency_median_ms": round(statistics.median(latencies), 2) if latencies else 0.0,
        "decision_latency_p95_ms": round(percentile(latencies, 0.95), 2),
        "invalid_answers": sum(int(r["invalid"]) for r in decisions),
        "agreement_with_greedy": round(statistics.fmean(agrees), 4) if agrees else None,
        "max_input_tokens": max((int(r["input_tokens"]) for r in decisions), default=0),
    }
