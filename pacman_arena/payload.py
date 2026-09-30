"""The single input every model receives at a decision: instruction, state text, lettered options.

Laya's checkpoint silently truncates long inputs, so the option list is fitted to Laya's token
budget and *every* model receives that same fitted payload. Only the wrapper differs per model
(see models.py).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .game import UNREACHABLE, Game, distance_map
from .paths import PathOption, candidate_options, compress, ghost_moves_by, shortlist

LETTERS = "ABCDEFGHIJKLMNOPQRST"  # Ollama returns at most 20 top logprobs

INSTRUCTIONS = "Choose the path that maximizes Pac-Man's score."
RULES = ("rules: +10 per pellet; touching a ghost loses a life; 0 lives is game over; ghosts move "
         "1 cell per 2 Pac-Man steps and mostly chase Pac-Man; margin = steps to spare before a "
         "chasing ghost can reach the path, 0 or less is UNSAFE.\n"
         "strategy: prefer SAFE paths with the most pellets, eaten soonest; avoid dead ends when a "
         "ghost is near; if no path has pellets, head toward the nearest pellet.")


@dataclass
class Payload:
    instructions: str
    state: str
    criteria: dict[str, str]                      # letter -> option description
    options: list[PathOption] = field(repr=False)  # same order as criteria
    laya_tokens: int = 0
    truncated: bool = False


def option_text(o: PathOption, level: int = 0) -> str:
    """Describe one path. Levels 1-2 are terser; used only when an option exceeds Laya's per-option cap."""
    parts = [compress(o.moves), f"{o.pellets} pellets" + (f" from step {o.first_pellet}" if o.first_pellet else "")]
    if o.margin <= 0:
        parts.append(f"UNSAFE margin {o.margin} (G{o.danger_ghost} at step {o.danger_step}), "
                     f"{o.safe_pellets} safe pellets" if level < 2 else f"UNSAFE {o.margin} G{o.danger_ghost}")
    else:
        parts.append(f"SAFE margin {o.margin if o.margin < UNREACHABLE else 'large'}")
    for r in o.ghosts_on_path:
        parts.append(f"G{r['ghost']} on it at step {r['step']} {r['motion']}" if level < 2
                     else f"G{r['ghost']}@{r['step']} {r['motion']}")
    if level == 0:
        parts.append("ends in dead end" if o.end_dead_end else
                     f"ends with {o.end_exits} exit{'s' if o.end_exits != 1 else ''}")
    if o.pellets == 0 and level < 2:
        parts.append(f"nearest pellet {o.beyond_pellet} further")
    return "; ".join(parts)


def build_payload(game: Game, options: list[PathOption], describe=option_text) -> Payload:
    labels = {id(o): LETTERS[i] for i, o in enumerate(options)}
    from_pac = distance_map(game.pac)
    lines = [RULES,
             f"pacman {game.pac}; score {game.score}; lives {game.lives}; pellets left {len(game.pellets)}; "
             f"last move {game.last_move or 'none'}; ghosts move next step: "
             f"{'yes' if ghost_moves_by(game.tick, 1) else 'no'}"]
    for g, ghost in enumerate(game.ghosts):
        now = from_pac.get(ghost, UNREACHABLE)
        before = from_pac.get(game.ghost_prev[g], UNREACHABLE)
        trend = "closing in" if now < before else "moving away" if now > before else "steady"
        on = [f"{labels[id(o)]}@{r['step']}" for o in options for r in o.ghosts_on_path if r["ghost"] == g]
        lines.append(f"G{g} {ghost} heading {game.ghost_headings[g] or 'none'}, distance {now}, {trend}, "
                     f"on paths {' '.join(on) if on else 'none'}")
    return Payload(instructions=INSTRUCTIONS, state="\n".join(lines),
                   criteria={labels[id(o)]: describe(o) for o in options}, options=options)


class LayaBudget:
    """Mirrors laya_mlx.common.build_prefix/build_sequence to detect any truncation.

    Laya input: [CLS] "choice question: <instructions>" [SEP] ([MASK] option)* [SEP] state [SEP]
    - instructions + options must fit `head_max_len` (192) and each option 48 tokens;
    - the whole sequence must fit `max_len` (512), otherwise the state is cut.
    """

    PER_OPTION = 48

    def __init__(self, tokenizer: Any, config: dict):
        self.tok = tokenizer
        self.max_len = config.get("max_len", 512)
        self.head_max = config.get("head_max_len", 192)

    def count(self, text: str) -> int:
        return len(self.tok(text, add_special_tokens=False)["input_ids"])

    def describe(self, o: PathOption) -> str:
        """Least-terse wording of an option that fits the per-option cap."""
        for level in range(3):
            text = option_text(o, level)
            if self.count(f" X: {text}") <= self.PER_OPTION:
                break
        return text

    def fits(self, payload: Payload) -> tuple[bool, int]:
        head = self.count(f"choice question: {payload.instructions}")
        opts = [1 + self.count(f" {k}: {v}") for k, v in payload.criteria.items()]
        state = self.count(payload.state)
        total = 1 + head + 1 + sum(opts) + 1 + state + 1
        ok = (max(opts) <= 1 + self.PER_OPTION and self.head_max - sum(opts) >= max(16, head)
              and total <= self.max_len)
        return ok, total


def fair_payload(game: Game, budget: LayaBudget, depth: int, per_direction: int, max_options: int) -> Payload:
    """Shortlist paths, then drop the lowest-valued ones until Laya would see everything untruncated."""
    options = shortlist(candidate_options(game, depth), per_direction, max_options)
    while True:
        payload = build_payload(game, options, budget.describe)
        ok, total = budget.fits(payload)
        if ok or len(options) <= 2:
            payload.laya_tokens, payload.truncated = total, not ok
            return payload
        options.remove(min(options, key=lambda o: o.value))
