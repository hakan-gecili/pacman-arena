"""Maze, game rules and ghost behaviour.

The game is turn-based: each call to `Game.advance` is one Pac-Man step. Ghosts move one cell
on every second step. All randomness comes from a per-game `random.Random(seed)`, so a seed
fully determines how ghosts behave for a given sequence of Pac-Man moves.
"""
from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass, field

Cell = tuple[int, int]

MAZE = [
    "###################",
    "#G.......#.......G#",
    "#.##.###.#.###.##.#",
    "#.................#",
    "#.##.#.#####.#.##.#",
    "#....#...#...#....#",
    "####.###.#.###.####",
    "#.................#",
    "#.##.#.#####.#.##.#",
    "#..#.....P.....#..#",
    "##.#.#.#####.#.#.##",
    "#....#...#...#....#",
    "#.######.#.######.#",
    "#.................#",
    "###################",
]
ROWS, COLS = len(MAZE), len(MAZE[0])
assert all(len(row) == COLS for row in MAZE)

DIRS: dict[str, Cell] = {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0)}
OPPOSITE = {"up": "down", "down": "up", "left": "right", "right": "left"}
DELTA_TO_DIR = {delta: name for name, delta in DIRS.items()}

CELLS = {(x, y): ch for y, row in enumerate(MAZE) for x, ch in enumerate(row)}
WALLS = {p for p, ch in CELLS.items() if ch == "#"}
OPEN = set(CELLS) - WALLS
PAC_START = next(p for p, ch in CELLS.items() if ch == "P")
GHOST_STARTS = [p for p, ch in CELLS.items() if ch == "G"]
PELLETS_START = {p for p, ch in CELLS.items() if ch in ".PG"} - {PAC_START}

LIVES = 3
PELLET_POINTS = 10
GHOST_MOVE_EVERY = 2      # ghosts move on every second Pac-Man step
GHOST_CHASE_PROB = 0.6    # otherwise a ghost picks a random legal move
UNREACHABLE = 99


def step(p: Cell, direction: str) -> Cell:
    dx, dy = DIRS[direction]
    return p[0] + dx, p[1] + dy


def legal(p: Cell) -> list[str]:
    """Directions from `p` that do not hit a wall, in DIRS order."""
    return [d for d in DIRS if step(p, d) in OPEN]


def distance_map(src: Cell) -> dict[Cell, int]:
    """Maze (BFS) distance from `src` to every reachable open cell."""
    dist, queue = {src: 0}, deque([src])
    while queue:
        p = queue.popleft()
        for direction in DIRS:
            nxt = step(p, direction)
            if nxt in OPEN and nxt not in dist:
                dist[nxt] = dist[p] + 1
                queue.append(nxt)
    return dist


@dataclass
class Game:
    seed: int
    rng: random.Random = field(init=False)
    pac: Cell = PAC_START
    ghosts: list[Cell] = field(default_factory=lambda: list(GHOST_STARTS))
    ghost_headings: list[str | None] = field(default_factory=lambda: [None] * len(GHOST_STARTS))
    ghost_prev: list[Cell] = field(default_factory=lambda: list(GHOST_STARTS))  # cell before each ghost's last move
    pellets: set[Cell] = field(default_factory=lambda: set(PELLETS_START))
    last_move: str | None = None
    score: int = 0
    lives: int = LIVES
    tick: int = 0
    steps: int = 0
    lives_lost: int = 0
    pellets_eaten: int = 0
    cleared: bool = False

    def __post_init__(self) -> None:
        self.rng = random.Random(self.seed)

    @property
    def done(self) -> bool:
        return self.lives == 0 or self.cleared

    def advance(self, direction: str) -> tuple[bool, bool]:
        """Play one Pac-Man step; returns (ate_pellet, lost_life)."""
        if direction not in legal(self.pac):
            direction = legal(self.pac)[0]
        self.pac = step(self.pac, direction)
        self.last_move = direction
        ate = self.pac in self.pellets
        if ate:
            self.pellets.remove(self.pac)
            self.score += PELLET_POINTS
            self.pellets_eaten += 1
        self.tick += 1
        if self.pac not in self.ghosts and self.tick % GHOST_MOVE_EVERY == 0:
            self._move_ghosts()
        died = self.pac in self.ghosts
        if died:
            self.lives -= 1
            self.lives_lost += 1
            self.ghost_headings = [None] * len(self.ghosts)
            if self.lives:  # respawn everyone; eaten pellets stay eaten
                self.pac, self.ghosts = PAC_START, list(GHOST_STARTS)
            self.ghost_prev = list(self.ghosts)
        self.cleared = not self.pellets
        self.steps += 1
        return ate, died

    def _move_ghosts(self) -> None:
        to_pac = distance_map(self.pac)
        moved = []
        for index, ghost in enumerate(self.ghosts):
            options = legal(ghost)
            if self.rng.random() < GHOST_CHASE_PROB:
                move = min(options, key=lambda d: to_pac.get(step(ghost, d), UNREACHABLE))
            else:
                move = self.rng.choice(options)
            new = step(ghost, move)
            self.ghost_headings[index] = move
            self.ghost_prev[index] = ghost
            moved.append(new)
        self.ghosts = moved
