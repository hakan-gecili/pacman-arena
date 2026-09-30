"""Game rules and ghost timing (no models needed)."""
from pacman_arena.game import GHOST_STARTS, LIVES, PAC_START, PELLETS_START, Game, distance_map, legal, step
from pacman_arena.paths import compress, enumerate_paths, ghost_arrival, ghost_moves_by


def test_maze_layout():
    assert len(PELLETS_START) == 141
    assert len(GHOST_STARTS) == 2
    assert set(legal(PAC_START)) == {"left", "right"}


def test_ghosts_move_every_second_step():
    game = Game(seed=3)
    game.advance("left")
    assert game.ghosts == GHOST_STARTS  # tick 1: ghosts wait
    game.advance("left")
    assert game.ghosts != GHOST_STARTS  # tick 2: ghosts move one cell each
    assert all(abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1 for a, b in zip(game.ghosts, GHOST_STARTS))


def test_ghost_timing_helpers():
    assert [ghost_moves_by(0, s) for s in range(5)] == [0, 0, 1, 1, 2]
    assert [ghost_moves_by(1, s) for s in range(5)] == [0, 1, 1, 2, 2]
    assert ghost_arrival(0, 3) == 6 and ghost_arrival(1, 3) == 5


def test_same_seed_same_game():
    moves = ["left", "left", "left", "up", "up", "right", "right"] * 5
    a, b = Game(seed=7), Game(seed=7)
    for move in moves:
        a.advance(move)
        b.advance(move)
    assert (a.pac, a.ghosts, a.score, a.lives) == (b.pac, b.ghosts, b.score, b.lives)


def test_eating_and_dying():
    game = Game(seed=1)
    ate, _ = game.advance("left")
    assert ate and game.score == 10
    game.ghosts = [step(game.pac, "left")]
    game.ghost_headings, game.ghost_prev = [None], list(game.ghosts)
    _, died = game.advance("left")
    assert died and game.lives == LIVES - 1 and game.pac == PAC_START


def test_paths_are_simple_and_bounded():
    paths = enumerate_paths(PAC_START, depth=8)
    assert paths
    for moves, cells in paths:
        assert len(moves) == len(cells) <= 8
        assert len(set(cells)) == len(cells) and PAC_START not in cells
    assert compress(("up", "up", "left")) == "up2 left1"
    assert distance_map(PAC_START)[PAC_START] == 0
