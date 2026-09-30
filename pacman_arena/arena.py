"""Command line: run the models side by side (live window) or headless, and save every run."""
from __future__ import annotations

import argparse
import csv
import json
import platform
import shutil
import statistics
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import datetime
from importlib import metadata
from pathlib import Path
from typing import Any

from .game import COLS, ROWS, Game
from .models import (KNOWN_DIGESTS, LAYA_MODEL, LAYA_REVISION, NIMBLE_MODEL, OLLAMA_URL, TEV_MODEL, Contestant,
                     greedy_contestant, laya_contestant, load_laya, nimble_prompt, ollama_contestant,
                     ollama_models, tev_prompt)
from .payload import INSTRUCTIONS, LETTERS, RULES, LayaBudget, fair_payload
from .pilot import Pilot, PlanSettings
from .render import MODEL_COLORS, draw_board
from .stats import log_row, summarize, write_csv

CONTESTANTS = ("laya", "tev", "nimble", "greedy")
TIEBREAK = "score, then lives left, then fewer steps"
PACKAGES = ("laya-mlx", "mlx", "numpy", "pygame-ce", "requests", "huggingface_hub", "tokenizers")


# ---------------------------------------------------------------- setup

def build_contestants(args: argparse.Namespace) -> tuple[list[Contestant], LayaBudget]:
    """Laya is always loaded: its tokenizer defines the shared input budget, even if it does not play."""
    agent = load_laya(args.laya_model, args.laya_revision, args.dtype)
    budget = LayaBudget(agent.tok, agent.cfg)
    factories = {
        "laya": lambda: laya_contestant(agent, args.laya_model, args.laya_revision),
        "tev": lambda: ollama_contestant("tev", args.tev_model, args.ollama_url, tev_prompt),
        "nimble": lambda: ollama_contestant("nimble", args.nimble_model, args.ollama_url, nimble_prompt),
        "greedy": greedy_contestant,
    }
    contestants = [factories[key]() for key in args.models]
    for contestant in contestants:
        warm_up(contestant, budget, args)
    return contestants, budget


def warm_up(contestant: Contestant, budget: LayaBudget, args: argparse.Namespace) -> None:
    """One untimed call, so model loading / graph compilation never counts as decision time."""
    contestant.choose(fair_payload(Game(args.seed), budget, args.depth, args.per_direction, args.max_options))


def settings(args: argparse.Namespace) -> PlanSettings:
    return PlanSettings(args.depth, args.per_direction, args.max_options, args.replan, args.danger_margin)


def environment() -> dict[str, Any]:
    versions = {}
    for package in PACKAGES:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    return {"python": platform.python_version(), "platform": platform.platform(),
            "machine": platform.machine(), "packages": versions}


def make_run_dir(args: argparse.Namespace, mode: str, contestants: list[Contestant]) -> Path:
    run_dir = args.out / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{mode}"
    run_dir.mkdir(parents=True, exist_ok=False)
    config = {
        "mode": mode, "argv": sys.argv[1:], "models": {c.name: c.info for c in contestants},
        "settings": asdict(settings(args)), "seed": args.seed, "max_steps": args.max_steps, "tiebreak": TIEBREAK,
        "instructions": INSTRUCTIONS, "rules": RULES, "environment": environment(),
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    print(f"results -> {run_dir}", flush=True)
    return run_dir


def slug(name: str) -> str:
    return name.lower().replace(":", "-").replace("/", "-").replace(" ", "-")


def rank_key(game: Game) -> tuple[int, int, int]:
    return game.score, game.lives, -game.steps


def round_winner(games: dict[str, Game]) -> str | None:
    """Name of the sole best game, or None when the best result is shared."""
    best = max(rank_key(g) for g in games.values())
    winners = [name for name, g in games.items() if rank_key(g) == best]
    return winners[0] if len(winners) == 1 else None


def play_step(game: Game, pilot: Pilot) -> tuple[dict[str, Any], bool, bool]:
    started = time.perf_counter()
    decision = pilot.act(game)
    decision["step_ms"] = (time.perf_counter() - started) * 1000.0  # path analysis + model call, if any
    ate, died = game.advance(decision["move"])
    if died:
        pilot.reset()
    return decision, ate, died


# ---------------------------------------------------------------- modes

def run_headless(args: argparse.Namespace) -> int:
    contestants, budget = build_contestants(args)
    run_dir = make_run_dir(args, "headless", contestants)
    seeds = ([int(v) for v in args.seeds.split(",")] if args.seeds
             else list(range(args.seed, args.seed + args.games)))
    results: dict[str, Any] = {}
    games: dict[str, dict[int, Game]] = {}
    for contestant in contestants:
        warm_up(contestant, budget, args)  # models play one after another; make sure this one is loaded
        rows, played, truncated = [], [], 0
        for round_index, seed in enumerate(seeds):
            game, pilot = Game(seed), Pilot(contestant, budget, settings(args))
            while not game.done and game.steps < args.max_steps:
                decision, ate, died = play_step(game, pilot)
                rows.append(log_row(seed, round_index, game, decision, ate, died))
            played.append(game)
            truncated += pilot.truncations
            games.setdefault(contestant.name, {})[seed] = game
            print(f"  {contestant.name:<16} seed {seed:>3}: score {game.score:>4}  lives {game.lives}  "
                  f"steps {game.steps}{'  cleared' if game.cleared else ''}", flush=True)
        write_csv(run_dir / f"{slug(contestant.name)}.csv", rows)
        results[contestant.name] = summarize(played, rows, args.seed) | {"truncated_decisions": truncated}
    wins = dict.fromkeys([c.name for c in contestants] + ["tie"], 0)
    per_seed = {}
    for seed in seeds:
        round_games = {c.name: games[c.name][seed] for c in contestants}
        winner = round_winner(round_games)
        wins[winner or "tie"] += 1
        per_seed[seed] = {name: {"score": g.score, "lives_left": g.lives, "steps": g.steps, "cleared": g.cleared}
                          for name, g in round_games.items()} | {"winner": winner or "tie"}
    results["head_to_head"] = {"seeds": seeds, "wins": wins, "tiebreak": TIEBREAK, "per_seed": per_seed}
    (run_dir / "summary.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print_summary(results, contestants)
    print(f"-> {run_dir}")
    return 0


def print_summary(results: dict[str, Any], contestants: list[Contestant]) -> None:
    print(f"\n{'model':<18}{'mean score':>12}{'cleared':>9}{'lives left':>12}{'median ms':>11}{'wins':>6}")
    wins = results["head_to_head"]["wins"]
    for c in contestants:
        r = results[c.name]
        print(f"{c.name:<18}{r['mean_score']:>12}{r['board_clear_rate']:>9.0%}{r['mean_lives_left']:>12}"
              f"{r['decision_latency_median_ms']:>11}{wins[c.name]:>6}")
    print(f"ties: {wins['tie']}")


def run_live(args: argparse.Namespace) -> int:
    contestants, budget = build_contestants(args)
    run_dir = make_run_dir(args, "live", contestants)
    import pygame
    pygame.init()
    n, gap, top = len(contestants), 30, 96
    panel_h = 52 + 16 * min(args.max_options, 8)
    cell = args.cell
    if not cell:  # largest cell (at most 28 px) that fits every board on this display
        info, cell = pygame.display.Info(), 28
        if info.current_w > 0 and info.current_h > 0:
            cell = min(28, (info.current_w - 40 - (n + 1) * gap) // (n * COLS),
                       (info.current_h - 90 - top - panel_h) // ROWS)
        cell = max(12, cell)
    board_w, board_h = COLS * cell, ROWS * cell
    screen = pygame.display.set_mode((n * board_w + (n + 1) * gap, top + board_h + panel_h + 12))
    pygame.display.set_caption(" vs ".join(c.name for c in contestants) + " — Pac-Man")
    fonts = {"small": pygame.font.SysFont("menlo", 12), "big": pygame.font.SysFont("menlo", 16, bold=True),
             "title": pygame.font.SysFont("menlo", 24 if board_w >= 480 else 18, bold=True),
             "head": pygame.font.SysFont("menlo", 14)}
    frames_dir = run_dir / "frames" if args.record else None
    if frames_dir:
        frames_dir.mkdir()

    names = [c.name for c in contestants]
    wins = dict.fromkeys(names + ["tie"], 0)
    rows: dict[str, list[dict[str, Any]]] = {name: [] for name in names}
    finished: dict[str, list[Game]] = {name: [] for name in names}
    round_rows: list[dict[str, Any]] = []

    def new_round(seed: int) -> dict[str, tuple[Game, Pilot]]:
        return {c.name: (Game(seed), Pilot(c, budget, settings(args))) for c in contestants}

    def save() -> None:
        """Rewritten after every round, so quitting mid-session keeps all finished rounds."""
        fields = ["round", "seed", "winner"] + [f"{slug(name)}_{col}" for name in names
                                                for col in ("score", "lives", "steps", "cleared", "decisions",
                                                            "median_ms")]
        with (run_dir / "rounds.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(round_rows)
        summary: dict[str, Any] = {}
        for name in names:
            write_csv(run_dir / f"{slug(name)}.csv", rows[name])
            summary[name] = summarize(finished[name], rows[name], args.seed)
        summary["head_to_head"] = {"rounds": len(round_rows), "wins": wins, "tiebreak": TIEBREAK}
        (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    clock, seed, round_index, frame, hold, running = pygame.time.Clock(), args.seed, 0, 0, 0, True
    sides = new_round(seed)
    decisions: dict[str, dict[str, Any]] = {name: {} for name in names}
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key in (pygame.K_ESCAPE, pygame.K_q)):
                running = False
        if hold:  # showing a finished round
            hold -= 1
            if hold == 0:
                round_index += 1
                if args.rounds and round_index >= args.rounds:
                    running = False
                else:
                    seed += 1
                    sides, decisions = new_round(seed), {name: {} for name in names}
        else:
            for name, (game, pilot) in sides.items():
                if not game.done and game.steps < args.max_steps:
                    decisions[name], ate, died = play_step(game, pilot)
                    rows[name].append(log_row(seed, round_index, game, decisions[name], ate, died))
            if all(g.done or g.steps >= args.max_steps for g, _ in sides.values()):
                winner = round_winner({name: g for name, (g, _) in sides.items()})
                wins[winner or "tie"] += 1
                row: dict[str, Any] = {"round": round_index + 1, "seed": seed, "winner": winner or "tie"}
                for name, (g, pilot) in sides.items():
                    finished[name].append(g)
                    median = round(statistics.median(pilot.latencies), 1) if pilot.latencies else 0.0
                    row |= {f"{slug(name)}_score": g.score, f"{slug(name)}_lives": g.lives,
                            f"{slug(name)}_steps": g.steps, f"{slug(name)}_cleared": int(g.cleared),
                            f"{slug(name)}_decisions": len(pilot.latencies), f"{slug(name)}_median_ms": median}
                round_rows.append(row)
                save()
                print(f"round {round_index + 1} (seed {seed}): "
                      + " | ".join(f"{name} {g.score}" for name, (g, _) in sides.items())
                      + f" -> {winner or 'tie'}", flush=True)
                hold = max(1, args.speed * 3)  # keep the result on screen for ~3 s
        screen.fill((18, 18, 26))
        header = (f"Same maze, same seed ({seed}), same state and path options for every model   |   "
                  f"look-ahead {args.depth} steps   |   ties {wins['tie']}")
        screen.blit(fonts["head"].render(header, True, (200, 200, 210)), (gap, 14))
        for i, c in enumerate(contestants):
            game, pilot = sides[c.name]
            draw_board(pygame, screen, fonts, gap + i * (board_w + gap), top, cell, c.name, MODEL_COLORS[c.key],
                       game, pilot.latencies, decisions[c.name], panel_h, wins[c.name])
        pygame.display.flip()
        if frames_dir and running:
            pygame.image.save(screen, str(frames_dir / f"frame_{frame:06d}.png"))
            frame += 1
        clock.tick(args.speed)
    pygame.quit()
    if frames_dir and frame:
        encode_video(frames_dir, args.speed, run_dir / "game.mp4")
    print(f"-> {run_dir}")
    return 0


def encode_video(frames_dir: Path, fps: int, output: Path) -> None:
    cmd = ["ffmpeg", "-y", "-framerate", str(fps), "-i", str(frames_dir / "frame_%06d.png"),
           "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(output)]
    if shutil.which("ffmpeg"):
        subprocess.run(cmd, check=True, capture_output=True)
        print(f"video -> {output}")
    else:
        print("Frames saved. Install ffmpeg (brew install ffmpeg) and run:\n  " + " ".join(cmd))


def show_inputs(args: argparse.Namespace) -> int:
    contestants, budget = build_contestants(args)
    payload = fair_payload(Game(args.seed), budget, args.depth, args.per_direction, args.max_options)
    print(f"# Shared payload: {len(payload.criteria)} options, Laya input {payload.laya_tokens} tokens, "
          f"truncated={payload.truncated}\n")
    for c in contestants:
        answer = c.choose(payload)
        probabilities = {k: round(v, 3) for k, v in answer["probabilities"].items()}
        print(f"===== {c.name} =====\n{answer['prompt']}\n-> picks {answer['letter']}  {probabilities}  "
              f"({answer['latency_ms']:.0f} ms)\n")
    return 0


def check_setup(args: argparse.Namespace) -> int:
    """Verify Python packages, Ollama models (against the published digests) and the Laya checkpoint."""
    ok = True
    env = environment()
    print(f"python {env['python']} on {env['platform']}")
    if env["machine"] != "arm64" or sys.platform != "darwin":
        ok = False
        print("  ! laya-mlx needs an Apple Silicon Mac (MLX)")
    for package, version in env["packages"].items():
        print(f"  {package:<16} {version or 'MISSING'}")
        ok &= version is not None
    try:
        pulled = ollama_models(args.ollama_url)
    except SystemExit as error:
        print(f"  ! {error}")
        return 1
    for model in dict.fromkeys([args.tev_model, args.nimble_model, *KNOWN_DIGESTS]):
        digest = pulled.get(model) or pulled.get(f"{model}:latest")
        expected = KNOWN_DIGESTS.get(model)
        state = ("not pulled" if not digest else "matches published digest" if digest == expected
                 else "pulled (digest differs from the published runs)" if expected else "pulled")
        print(f"  ollama {model:<14} {state}")
        ok &= digest is not None or model not in (args.tev_model, args.nimble_model)
    agent = load_laya(args.laya_model, args.laya_revision, args.dtype)
    print(f"  laya {args.laya_model}@{(args.laya_revision or 'latest')[:12]} loaded from {agent.model_dir}")
    print("setup OK" if ok else "setup incomplete")
    return 0 if ok else 1


# ---------------------------------------------------------------- CLI

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m pacman_arena",
        description="Pac-Man decision arena: Laya vs tev1 vs nimble under identical information.")
    parser.add_argument("--models", default="laya,tev,nimble",
                        help=f"comma-separated lineup from {', '.join(CONTESTANTS)} (default laya,tev,nimble)")
    parser.add_argument("--tev-model", default=TEV_MODEL, help=f"tev1:4b or tev1:0.8b (default {TEV_MODEL})")
    parser.add_argument("--nimble-model", default=NIMBLE_MODEL, help=f"default {NIMBLE_MODEL}")
    parser.add_argument("--laya-model", default=LAYA_MODEL)
    parser.add_argument("--laya-revision", default=LAYA_REVISION, help="Hugging Face commit ('' = latest)")
    parser.add_argument("--dtype", default="float16", choices=["float16", "float32"], help="Laya weights dtype")
    parser.add_argument("--ollama-url", default=OLLAMA_URL)

    mode = parser.add_argument_group("mode")
    mode.add_argument("--headless", action="store_true", help="no window: every model plays the same seeds")
    mode.add_argument("--show-inputs", action="store_true", help="print each model's exact input for one state")
    mode.add_argument("--check", action="store_true", help="verify packages, models and digests, then exit")

    game = parser.add_argument_group("game")
    game.add_argument("--seed", type=int, default=1, help="first seed; later rounds/games use seed+1, seed+2, ...")
    game.add_argument("--games", type=int, default=20, help="headless: number of seeds")
    game.add_argument("--seeds", help="headless: explicit comma-separated seeds")
    game.add_argument("--rounds", type=int, default=0, help="live: stop after N rounds (0 = until you quit)")
    game.add_argument("--max-steps", type=int, default=2000, help="step cap per game")

    plan = parser.add_argument_group("decision making")
    plan.add_argument("--depth", type=int, default=10, help="path look-ahead in steps")
    plan.add_argument("--per-direction", type=int, default=3, help="paths kept per first move before fitting")
    plan.add_argument("--max-options", type=int, default=10, help="options before fitting Laya's token budget")
    plan.add_argument("--replan", default="junction", choices=["junction", "step"])
    plan.add_argument("--danger-margin", type=int, default=1, help="re-plan when the plan's margin <= this")

    out = parser.add_argument_group("output")
    out.add_argument("--out", type=Path, default=Path("runs"), help="runs go to <out>/<YYYYmmdd_HHMMSS>_<mode>/")
    out.add_argument("--speed", type=int, default=8, help="live: steps per second (also the video fps)")
    out.add_argument("--cell", type=int, default=0, help="live: board cell size in px (0 = fit the screen)")
    out.add_argument("--record", action="store_true", help="live: save frames and encode game.mp4")

    args = parser.parse_args(argv)
    args.models = [m.strip() for m in args.models.split(",") if m.strip()]
    unknown = [m for m in args.models if m not in CONTESTANTS]
    if unknown or not args.models or len(set(args.models)) != len(args.models):
        parser.error(f"--models must be distinct names from {', '.join(CONTESTANTS)}")
    args.laya_revision = args.laya_revision or None
    if not 2 <= args.max_options <= len(LETTERS):
        parser.error(f"--max-options must be between 2 and {len(LETTERS)}")
    if min(args.depth, args.per_direction, args.games, args.max_steps, args.speed) < 1:
        parser.error("--depth, --per-direction, --games, --max-steps and --speed must be positive")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.check:
        return check_setup(args)
    if args.show_inputs:
        return show_inputs(args)
    return run_headless(args) if args.headless else run_live(args)
