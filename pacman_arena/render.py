"""pygame drawing: arcade-style sprites and one board + decision panel per model."""
from __future__ import annotations

import math
import statistics
from typing import Any

from .game import COLS, DIRS, ROWS, UNREACHABLE, WALLS, Game

GHOST_COLORS = [(255, 0, 0), (255, 184, 255), (0, 255, 255), (255, 184, 82)]  # Blinky, Pinky, Inky, Clyde
MODEL_COLORS = {"laya": (80, 220, 120), "tev": (255, 150, 60), "nimble": (120, 170, 255), "greedy": (200, 200, 200)}
FACING_DEGREES = {"right": 0, "down": 90, "left": 180, "up": 270}  # screen angles, y points down


def draw_pacman(pygame: Any, screen: Any, center: tuple[int, int], radius: int, facing: str | None, frame: int) -> None:
    """Yellow disc with a wedge mouth toward `facing`; the mouth opens and closes with `frame`."""
    mouth = (50, 30, 8, 30)[frame % 4]
    base = FACING_DEGREES.get(facing or "left", 180)
    cx, cy = center
    points = [center]
    for degrees in list(range(mouth // 2, 361 - mouth // 2, 10)) + [360 - mouth // 2]:
        angle = math.radians(base + degrees)
        points.append((cx + radius * math.cos(angle), cy + radius * math.sin(angle)))
    pygame.draw.polygon(screen, (255, 230, 0), points)


def draw_ghost(pygame: Any, screen: Any, center: tuple[int, int], radius: int, color: tuple[int, int, int],
               heading: str | None, frame: int) -> None:
    """Arcade-style ghost: domed head, wavy skirt, eyes looking toward `heading`."""
    cx, cy = center
    top_y, bottom = cy - radius // 4, cy + radius
    points = [(cx - radius, bottom), (cx - radius, top_y)]
    for degrees in range(180, 361, 15):
        angle = math.radians(degrees)
        points.append((cx + radius * math.cos(angle), top_y + radius * math.sin(angle)))
    points.append((cx + radius, bottom))
    feet, shift = 3, frame % 2
    for i in range(1, 2 * feet):
        points.append((cx + radius - i * radius / feet, bottom - (radius // 3 if (i + shift) % 2 else 0)))
    pygame.draw.polygon(screen, color, points)
    dx, dy = DIRS[heading] if heading else (0, 0)
    eye_w, eye_h = max(4, int(radius * 0.55)), max(5, int(radius * 0.7))
    for side in (-1, 1):
        ex, ey = cx + side * int(radius * 0.4), top_y - radius // 8
        pygame.draw.ellipse(screen, (255, 255, 255), (ex - eye_w // 2 + dx * 2, ey - eye_h // 2 + dy * 2, eye_w, eye_h))
        pygame.draw.circle(screen, (33, 33, 222), (ex + dx * (eye_w // 3 + 1), ey + dy * (eye_h // 3 + 1)),
                           max(2, int(radius * 0.2)))


def draw_board(pygame: Any, screen: Any, fonts: dict[str, Any], x0: int, y0: int, cell: int, name: str,
               color: tuple, game: Game, latencies: list[float], decision: dict[str, Any], panel_h: int,
               wins: int) -> None:
    board_w, board_h = COLS * cell, ROWS * cell
    center = lambda p: (x0 + p[0] * cell + cell // 2, y0 + p[1] * cell + cell // 2)  # noqa: E731
    screen.blit(fonts["title"].render(name, True, color), (x0, y0 - 34))
    wins_text = fonts["big"].render(f"wins {wins}", True, (220, 220, 220))
    screen.blit(wins_text, (x0 + board_w - wins_text.get_width(), y0 - 30))
    pygame.draw.rect(screen, (10, 10, 16), (x0, y0, board_w, board_h))
    for x, y in WALLS:
        pygame.draw.rect(screen, (40, 60, 170), (x0 + x * cell, y0 + y * cell, cell, cell))
    start = decision.get("origin") or game.pac  # paths are drawn from where they were planned
    for option in decision.get("options") or []:
        pygame.draw.lines(screen, (60, 60, 82), False, [center(start)] + [center(c) for c in option["cells"]], 2)
    if decision.get("chosen_cells"):
        pygame.draw.lines(screen, color, False, [center(start)] + [center(c) for c in decision["chosen_cells"]],
                          max(2, cell // 7))
    for p in game.pellets:
        pygame.draw.circle(screen, (240, 220, 190), center(p), max(2, cell // 9))
    for index, ghost in enumerate(game.ghosts):
        draw_ghost(pygame, screen, center(ghost), cell // 2 - 2, GHOST_COLORS[index % len(GHOST_COLORS)],
                   game.ghost_headings[index], game.steps)
    draw_pacman(pygame, screen, center(game.pac), cell // 2 - 2, game.last_move, game.steps)
    if game.done:
        overlay = pygame.Surface((board_w, board_h), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 150))
        screen.blit(overlay, (x0, y0))
        banner = fonts["title"].render("BOARD CLEARED" if game.cleared else "GAME OVER", True, (255, 255, 255))
        screen.blit(banner, (x0 + (board_w - banner.get_width()) // 2, y0 + board_h // 2 - 16))

    y = y0 + board_h + 8
    median = statistics.median(latencies) if latencies else 0.0
    screen.blit(fonts["big"].render(f"score {game.score}   lives {game.lives}   steps {game.steps}",
                                    True, (235, 235, 235)), (x0, y))
    screen.blit(fonts["small"].render(f"decisions {len(latencies)}   median decision {median:.0f} ms   "
                                      f"pellets left {len(game.pellets)}", True, (160, 160, 170)), (x0, y + 24))
    probs = decision.get("probabilities") or {}
    columns = (board_w - 78) // max(1, fonts["small"].size("M")[0])  # characters beside the probability bar
    for i, option in enumerate(decision.get("options") or []):
        row = y + 46 + i * 16
        if row > y0 + board_h + panel_h - 14:
            break
        selected, p = option["label"] == decision.get("letter"), probs.get(option["label"], 0.0)
        pygame.draw.rect(screen, (45, 45, 58), (x0, row + 2, 70, 11))
        pygame.draw.rect(screen, color if selected else (115, 115, 135), (x0, row + 2, int(70 * p), 11))
        margin = option["margin"] if option["margin"] < UNREACHABLE else "∞"
        prefix = f"{option['label']} {p:.2f}  "
        suffix = f"  pellets {option['pellets']:>2}  margin {margin:>2}"
        width = columns - len(prefix) - len(suffix)
        if width < 12:
            suffix = f" p{option['pellets']} m{margin}"
            width = columns - len(prefix) - len(suffix)
        moves = option["moves"] if len(option["moves"]) <= width else option["moves"][:max(1, width - 1)] + "…"
        screen.blit(fonts["small"].render(f"{prefix}{moves:<{max(1, width)}}{suffix}", True,
                                          (255, 255, 255) if selected else (165, 165, 175)), (x0 + 78, row))
