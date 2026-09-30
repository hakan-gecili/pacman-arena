"""Turns model choices into moves: follow the chosen path, re-plan when needed."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .game import Game, distance_map, legal, step
from .models import Contestant
from .payload import LETTERS, LayaBudget, fair_payload
from .paths import analyze_path, compress


@dataclass
class PlanSettings:
    depth: int = 10            # path look-ahead in steps
    per_direction: int = 3     # options kept per first move before fitting Laya's budget
    max_options: int = 10      # upper bound on options before fitting
    replan: str = "junction"   # "junction": re-plan at junctions/danger/plan end; "step": every step
    danger_margin: int = 1     # re-plan when the rest of the plan's margin drops to this or below


class Pilot:
    """Asks its contestant for a path, then walks it until a re-plan trigger fires."""

    def __init__(self, contestant: Contestant, budget: LayaBudget, settings: PlanSettings):
        self.contestant, self.budget, self.settings = contestant, budget, settings
        self.plan: list[str] = []
        self.last: dict[str, Any] = {}
        self.latencies: list[float] = []
        self.truncations = 0

    def reset(self) -> None:
        """Drop the plan (called after losing a life, when everyone respawns)."""
        self.plan = []

    def replan_reason(self, game: Game) -> str | None:
        if not self.plan:
            return "plan_done"
        if self.settings.replan == "step":
            return "every_step"
        if self.plan[0] not in legal(game.pac):
            return "plan_invalid"
        cells, p = [], game.pac
        for move in self.plan:
            p = step(p, move)
            cells.append(p)
        remaining = analyze_path(game, tuple(self.plan), tuple(cells), [distance_map(g) for g in game.ghosts])
        if remaining.margin <= self.settings.danger_margin:
            return "danger"
        if len(legal(game.pac)) >= 3:
            return "junction"
        return None

    def act(self, game: Game) -> dict[str, Any]:
        """Return the decision record for this step; its "move" is what the game should play."""
        reason = self.replan_reason(game)
        decision: dict[str, Any] = {"replan": reason, "latency_ms": 0.0, "prompt": "", "invalid": 0,
                                    "input_tokens": 0, "greedy_agrees": None}
        if reason:
            s = self.settings
            payload = fair_payload(game, self.budget, s.depth, s.per_direction, s.max_options)
            self.truncations += payload.truncated
            answer = self.contestant.choose(payload)
            options, letters = payload.options, LETTERS[:len(payload.options)]
            greedy = max(options, key=lambda o: o.value)
            if answer["letter"] in letters:
                pick = options[letters.index(answer["letter"])]
            else:  # unparseable answer: fall back to the heuristic choice and count it
                pick, decision["invalid"] = greedy, 1
            self.latencies.append(answer["latency_ms"])
            decision.update(
                prompt=answer["prompt"], probabilities=answer["probabilities"], letter=answer["letter"],
                latency_ms=answer["latency_ms"], input_tokens=answer["input_tokens"],
                greedy_agrees=int(pick is greedy), chosen_path=compress(pick.moves), chosen_cells=pick.cells,
                origin=game.pac,
                options=[{"label": letters[i], "moves": compress(o.moves), "pellets": o.pellets,
                          "margin": o.margin, "value": round(o.value, 2), "cells": o.cells}
                         for i, o in enumerate(options)])
            self.plan = list(pick.moves)
        else:  # still following the previous plan; keep its details for display
            decision.update({k: self.last.get(k) for k in ("options", "probabilities", "letter", "chosen_cells", "origin")})
        decision["move"] = self.plan.pop(0)
        self.last = decision
        return decision
