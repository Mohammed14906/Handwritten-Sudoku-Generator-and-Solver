import numpy as np
from src.search import CorrectionSearchProblem, search
from src.csp import find_conflicts


def _make_grid_and_probs(grid, true_grid=None):
    """Helper: create probability matrix where true digit has high prob, others low."""
    probs = np.full((81, 9), 0.01, dtype=np.float32)
    for r in range(9):
        for c in range(9):
            idx = r * 9 + c
            d = grid[r, c]
            if d != 0:
                probs[idx, d - 1] = 0.85
                # If true_grid differs, give the true digit second-highest prob
                if true_grid is not None and true_grid[r, c] != d:
                    true_d = true_grid[r, c]
                    probs[idx, true_d - 1] = 0.10
    # Normalize each row
    probs = probs / probs.sum(axis=1, keepdims=True)
    return probs


# A known valid complete grid
VALID_GRID = np.array([
    [5, 3, 4, 6, 7, 8, 9, 1, 2],
    [6, 7, 2, 1, 9, 5, 3, 4, 8],
    [1, 9, 8, 3, 4, 2, 5, 6, 7],
    [8, 5, 9, 7, 6, 1, 4, 2, 3],
    [4, 2, 6, 8, 5, 3, 7, 9, 1],
    [7, 1, 3, 9, 2, 4, 8, 5, 6],
    [9, 6, 1, 5, 3, 7, 2, 8, 4],
    [2, 8, 7, 4, 1, 9, 6, 3, 5],
    [3, 4, 5, 2, 8, 6, 1, 7, 9],
])


# --- Normal cases ---

def test_single_misread():
    corrupted = VALID_GRID.copy()
    corrupted[0, 0] = 3  # was 5, now 3 (conflicts with (0,1) which is also 3)
    probs = _make_grid_and_probs(corrupted, VALID_GRID)
    for strategy in ["UCS", "Greedy", "A*"]:
        problem = CorrectionSearchProblem(corrupted, probs, max_corrections=3)
        result, stats = search(problem, strategy)
        assert stats["found"], f"{strategy} failed to find correction"
        assert result is not None
        corrections = {(r, c): d for (r, c, d) in result}
        assert (0, 0) in corrections
        assert corrections[(0, 0)] == 5


def test_two_misreads():
    corrupted = VALID_GRID.copy()
    corrupted[0, 0] = 3  # was 5
    corrupted[1, 0] = 7  # was 6, now 7 (conflicts with (1,1) which is 7)
    probs = _make_grid_and_probs(corrupted, VALID_GRID)
    problem = CorrectionSearchProblem(corrupted, probs, max_corrections=3)
    result, stats = search(problem, "UCS")
    assert stats["found"]
    assert len(result) == 2


def test_ucs_optimal():
    corrupted = VALID_GRID.copy()
    corrupted[0, 0] = 3  # was 5
    probs = _make_grid_and_probs(corrupted, VALID_GRID)
    problem = CorrectionSearchProblem(corrupted, probs, max_corrections=3)
    _, ucs_stats = search(problem, "UCS")
    _, greedy_stats = search(problem, "Greedy")
    assert ucs_stats["found"]
    assert ucs_stats["solution_cost"] <= greedy_stats.get("solution_cost", float("inf"))


def test_astar_fewer_nodes():
    corrupted = VALID_GRID.copy()
    corrupted[0, 0] = 3
    corrupted[1, 0] = 7
    probs = _make_grid_and_probs(corrupted, VALID_GRID)

    problem1 = CorrectionSearchProblem(corrupted, probs, max_corrections=3)
    _, ucs_stats = search(problem1, "UCS")

    problem2 = CorrectionSearchProblem(corrupted, probs, max_corrections=3)
    _, astar_stats = search(problem2, "A*")

    assert astar_stats["nodes_expanded"] <= ucs_stats["nodes_expanded"]


def test_no_conflicts():
    probs = _make_grid_and_probs(VALID_GRID)
    problem = CorrectionSearchProblem(VALID_GRID, probs, max_corrections=3)
    result, stats = search(problem, "UCS")
    assert stats["found"]
    assert len(result) == 0
    assert stats["nodes_expanded"] == 0


# --- Edge cases ---

def test_max_corrections_limit():
    corrupted = VALID_GRID.copy()
    corrupted[0, 0] = 3
    corrupted[1, 0] = 7
    corrupted[2, 0] = 9
    corrupted[3, 0] = 4
    probs = _make_grid_and_probs(corrupted, VALID_GRID)
    problem = CorrectionSearchProblem(corrupted, probs, max_corrections=2)
    result, stats = search(problem, "UCS")
    assert not stats["found"]


def test_ambiguous_probabilities():
    corrupted = VALID_GRID.copy()
    corrupted[0, 0] = 3  # was 5
    probs = np.full((81, 9), 0.10, dtype=np.float32)
    # Give correct digit slightly higher prob so it's in top-k
    probs[0, 5 - 1] = 0.15  # digit 5 slightly favoured
    probs[0, 3 - 1] = 0.20  # digit 3 (current wrong) still highest
    probs = probs / probs.sum(axis=1, keepdims=True)
    problem = CorrectionSearchProblem(corrupted, probs, max_corrections=3)
    result, stats = search(problem, "UCS")
    assert stats["found"]


def test_correction_fixes_multiple_conflicts():
    corrupted = VALID_GRID.copy()
    # Cell (0,0)=5. Change to 3 which conflicts in row (0,1)=3 and might conflict in col/box
    corrupted[0, 0] = 3
    probs = _make_grid_and_probs(corrupted, VALID_GRID)
    problem = CorrectionSearchProblem(corrupted, probs, max_corrections=3)
    result, stats = search(problem, "A*")
    assert stats["found"]
    assert len(result) == 1  # one correction should fix all


# --- Impossible case ---

def test_unsolvable_grid():
    # Many misreads that can't be fixed in 1 correction
    corrupted = VALID_GRID.copy()
    corrupted[0, 0] = 3
    corrupted[1, 0] = 7
    corrupted[2, 0] = 9
    corrupted[3, 0] = 4
    corrupted[4, 0] = 8
    probs = _make_grid_and_probs(corrupted, VALID_GRID)
    problem = CorrectionSearchProblem(corrupted, probs, max_corrections=1)
    result, stats = search(problem, "UCS")
    assert not stats["found"]
