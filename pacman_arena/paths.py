"""What Pac-Man "sees": every path it could take, annotated with pellets and ghost danger.

The models never see the raw maze. Code enumerates candidate paths and computes facts about
each one; the models then choose between those paths.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from .game import DIRS, GHOST_MOVE_EVERY, UNREACHABLE, Cell, Game, distance_map, legal, step


def enumerate_paths(start: Cell, depth: int, limit: int = 20000) -> list[tuple[tuple[str, ...], tuple[Cell, ...]]]:
    """All simple (non-self-crossing) paths of up to `depth` moves; a path ends early at a dead end."""
    paths: list[tuple[tuple[str, ...], tuple[Cell, ...]]] = []

    def extend(cells: list[Cell], moves: list[str]) -> None:
        if len(paths) >= limit:
            return
        options = [(d, step(cells[-1], d)) for d in legal(cells[-1])]
        options = [(d, nxt) for d, nxt in options if nxt not in cells]
        if len(moves) == depth or not options:
            if moves:
                paths.append((tuple(moves), tuple(cells[1:])))
            return
        for direction, nxt in options:
            extend(cells + [nxt], moves + [direction])

    extend([start], [])
    return paths


def ghost_moves_by(tick: int, steps: int) -> int:
    """Ghost moves completed after Pac-Man's next `steps` moves, starting from game tick `tick`."""
    return (tick + steps) // GHOST_MOVE_EVERY - tick // GHOST_MOVE_EVERY


def ghost_arrival(tick: int, distance: int) -> int:
    """Earliest Pac-Man step by which a ghost chasing directly could cover `distance` cells."""
    steps = 0
    while ghost_moves_by(tick, steps) < distance:
        steps += 1
    return steps


def escape_routes(cell: Cell, came_from: Cell | None = None) -> int:
    """Exits from `cell` (other than the one we came from) that lead to a junction, not a dead end."""
    count = 0
    for direction in legal(cell):
        first = step(cell, direction)
        if first == came_from:
            continue
        seen, queue, useful = {cell, first}, deque([first]), False
        while queue:
            pos = queue.popleft()
            if len(legal(pos)) >= 3:
                useful = True
                break
            exits = [step(pos, d) for d in legal(pos) if step(pos, d) not in seen]
            seen.update(exits)
            queue.extend(exits)
        count += useful
    return count


def compress(moves: tuple[str, ...]) -> str:
    """("up", "up", "left") -> "up2 left1"."""
    runs: list[list] = []
    for move in moves:
        if runs and runs[-1][0] == move:
            runs[-1][1] += 1
        else:
            runs.append([move, 1])
    return " ".join(f"{move}{count}" for move, count in runs)


@dataclass
class PathOption:
    moves: tuple[str, ...]
    cells: tuple[Cell, ...]
    pellets: int
    pellet_cells: frozenset[Cell]
    first_pellet: int | None      # step on which the first pellet is eaten
    safe_pellets: int             # pellets eaten before the first step a ghost could intercept
    margin: int                   # worst-case steps to spare before a chasing ghost can reach the path
    danger_ghost: int | None      # ghost that sets the margin
    danger_step: int | None       # path step where that ghost could intercept
    ghosts_on_path: list[dict] = field(default_factory=list)
    end_dead_end: bool = False
    end_exits: int = 0
    beyond_pellet: int = 0        # maze distance from the path end to the nearest pellet not on the path
    value: float = 0.0            # heuristic score, used only to shortlist and for the greedy baseline

    @property
    def direction(self) -> str:
        return self.moves[0]


def analyze_path(game: Game, moves: tuple[str, ...], cells: tuple[Cell, ...],
                 ghost_dists: list[dict[Cell, int]]) -> PathOption:
    pellet_steps = [i + 1 for i, cell in enumerate(cells) if cell in game.pellets]
    # margin: for every ghost and every cell of the path, compare when Pac-Man gets there (step i+1)
    # with the earliest step a ghost that chases directly could get there. The minimum wins.
    margin, danger_ghost, danger_step = UNREACHABLE, None, None
    for g, dist in enumerate(ghost_dists):
        for i, cell in enumerate(cells):
            slack = ghost_arrival(game.tick, dist.get(cell, UNREACHABLE)) - (i + 1)
            if slack < margin:
                margin, danger_ghost, danger_step = slack, g, i + 1
    unsafe_from = danger_step if margin <= 0 else len(cells) + 1
    on_path = []
    for g, ghost in enumerate(game.ghosts):
        if ghost in cells:
            k = cells.index(ghost)
            heading = game.ghost_headings[g]
            behind = cells[k - 1] if k > 0 else game.pac
            ahead = cells[k + 1] if k + 1 < len(cells) else None
            nxt = step(ghost, heading) if heading else None
            motion = ("toward you" if nxt == behind else
                      "moving away" if nxt is not None and nxt == ahead else
                      "leaving" if heading else "still")
            on_path.append({"ghost": g, "step": k + 1, "motion": motion})
    end = cells[-1]
    end_dist = distance_map(end)
    rest = game.pellets - set(cells)
    return PathOption(
        moves=moves, cells=cells, pellets=len(pellet_steps),
        pellet_cells=frozenset(c for c in cells if c in game.pellets),
        first_pellet=pellet_steps[0] if pellet_steps else None,
        safe_pellets=sum(s < unsafe_from for s in pellet_steps),
        margin=margin, danger_ghost=danger_ghost, danger_step=danger_step, ghosts_on_path=on_path,
        end_dead_end=len(legal(end)) <= 1,
        end_exits=escape_routes(end, cells[-2] if len(cells) > 1 else game.pac),
        beyond_pellet=min((end_dist.get(p, UNREACHABLE) for p in rest), default=0),
    )


def path_value(o: PathOption) -> float:
    """Transparent heuristic: safe pellets, eaten early, with penalties for danger and dead ends."""
    value = 10.0 * o.safe_pellets - (o.first_pellet or 0) * 0.5
    if o.pellets == 0:
        value -= o.beyond_pellet
    if o.margin <= 0:
        value -= 100.0 - 10.0 * o.margin
    else:
        value += 3.0 * min(o.margin, 4)
    if o.end_dead_end and o.margin < 6:
        value -= 15.0
    return value


def candidate_options(game: Game, depth: int) -> list[PathOption]:
    ghost_dists = [distance_map(g) for g in game.ghosts]
    options = [analyze_path(game, moves, cells, ghost_dists) for moves, cells in enumerate_paths(game.pac, depth)]
    for o in options:
        o.value = path_value(o)
    return options


def shortlist(options: list[PathOption], per_direction: int, max_options: int) -> list[PathOption]:
    """Per first move: the best-valued path, the safest path, then the next best; near-duplicates dropped."""
    chosen: list[PathOption] = []
    for direction in DIRS:
        group = sorted((o for o in options if o.direction == direction), key=lambda o: -o.value)
        picked: list[PathOption] = []
        seen = set()
        for o in group[:1] + sorted(group, key=lambda o: -o.margin)[:1] + group[1:]:
            key = (o.pellet_cells, o.margin)  # same pellets at the same risk
            if key in seen or len(picked) >= per_direction:
                continue
            seen.add(key)
            picked.append(o)
        chosen.extend(picked)
    if len(chosen) > max_options:
        keep = set(map(id, sorted(chosen, key=lambda o: -o.value)[:max_options]))
        chosen = [o for o in chosen if id(o) in keep]
    return chosen
