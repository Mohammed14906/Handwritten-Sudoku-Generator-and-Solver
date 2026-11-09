import numpy as np
from src.integration import solve_from_grid
from src.csp import is_valid_complete


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


def _make_probs(grid, true_grid=None):
    probs = np.full((81, 9), 0.01, dtype=np.float32)
    for r in range(9):
        for c in range(9):
            idx = r * 9 + c
            d = grid[r, c]
            if d != 0:
                probs[idx, d - 1] = 0.85
                if true_grid is not None and true_grid[r, c] != d:
                    probs[idx, true_grid[r, c] - 1] = 0.10
    probs = probs / probs.sum(axis=1, keepdims=True)
    return probs


def _make_puzzle_grid():
    """Create a puzzle (some cells empty) from VALID_GRID."""
    puzzle = VALID_GRID.copy()
    # Set some cells to 0 (empty)
    empty_cells = [(0, 2), (0, 3), (1, 1), (1, 2), (2, 0), (3, 1), (3, 2)]
    for r, c in empty_cells:
        puzzle[r, c] = 0
    return puzzle


# --- Normal cases ---

def test_clean_puzzle_e2e():
    puzzle = _make_puzzle_grid()
    probs = _make_probs(puzzle, VALID_GRID)
    result = solve_from_grid(puzzle, probs, "BT+FC+AC3", "A*")
    assert result["solved"]
    assert is_valid_complete(result["solution"])
    assert len(result["corrections"]) == 0


def test_one_misread_e2e():
    puzzle = _make_puzzle_grid()
    puzzle[0, 0] = 3  # was 5
    probs = _make_probs(puzzle, VALID_GRID)
    result = solve_from_grid(puzzle, probs, "BT+FC+AC3", "A*")
    assert result["solved"]
    assert is_valid_complete(result["solution"])
    assert len(result["corrections"]) >= 1


def test_two_misreads_e2e():
    puzzle = _make_puzzle_grid()
    puzzle[0, 0] = 3  # was 5
    puzzle[0, 1] = 4  # was 3 (now conflicts)
    probs = _make_probs(puzzle, VALID_GRID)
    result = solve_from_grid(puzzle, probs, "BT+FC+AC3", "A*")
    assert result["solved"]
    assert is_valid_complete(result["solution"])


def test_all_method_combos():
    puzzle = _make_puzzle_grid()
    puzzle[0, 0] = 3  # one misread
    probs = _make_probs(puzzle, VALID_GRID)
    for csp_method in ["BT", "BT+FC", "BT+FC+AC3"]:
        for search_strategy in ["UCS", "Greedy", "A*"]:
            result = solve_from_grid(puzzle, probs, csp_method, search_strategy)
            assert result["solved"], f"Failed with {csp_method}/{search_strategy}"


def test_correction_log_format():
    puzzle = _make_puzzle_grid()
    puzzle[0, 0] = 3
    probs = _make_probs(puzzle, VALID_GRID)
    result = solve_from_grid(puzzle, probs)
    for entry in result["corrections"]:
        assert "cell" in entry
        assert "read_digit" in entry
        assert "new_digit" in entry
        assert "cost" in entry
        assert isinstance(entry["cost"], float)


# --- Edge cases ---

def test_high_confidence_misread():
    puzzle = _make_puzzle_grid()
    puzzle[0, 0] = 3
    probs = _make_probs(puzzle)  # no true_grid, so wrong digit has highest prob
    # Override: give true digit a decent probability
    probs[0, 5 - 1] = 0.3  # digit 5 prob
    probs[0] = probs[0] / probs[0].sum()
    result = solve_from_grid(puzzle, probs, "BT+FC+AC3", "A*")
    assert result["solved"]


def test_many_empty_cells():
    puzzle = np.zeros((9, 9), dtype=int)
    # Only 17 givens from VALID_GRID
    given_positions = [(0, 0), (0, 4), (1, 1), (1, 5), (2, 2), (2, 6),
                       (3, 3), (3, 7), (4, 4), (4, 8), (5, 0), (5, 5),
                       (6, 1), (6, 6), (7, 2), (7, 7), (8, 3)]
    for r, c in given_positions:
        puzzle[r, c] = VALID_GRID[r, c]
    probs = _make_probs(puzzle)
    result = solve_from_grid(puzzle, probs, "BT+FC+AC3", "A*")
    assert result["solved"]


def test_perfect_nn():
    """When NN is perfect, no search needed."""
    puzzle = _make_puzzle_grid()
    probs = _make_probs(puzzle)
    result = solve_from_grid(puzzle, probs)
    assert result["solved"]
    assert result["search_stats"]["nodes_expanded"] == 0


# --- Impossible cases ---

def test_too_many_misreads():
    puzzle = _make_puzzle_grid()
    puzzle[0, 0] = 3
    puzzle[0, 1] = 4
    puzzle[1, 0] = 9
    puzzle[3, 0] = 5
    puzzle[4, 0] = 7
    probs = _make_probs(puzzle, VALID_GRID)
    result = solve_from_grid(puzzle, probs, max_corrections=2)
    assert not result["solved"]


def test_contradictory_uniform_probs():
    puzzle = _make_puzzle_grid()
    puzzle[0, 0] = 3
    puzzle[0, 1] = 3  # two 3s in row
    probs = np.full((81, 9), 1.0 / 9, dtype=np.float32)
    result = solve_from_grid(puzzle, probs, "BT+FC+AC3", "A*", max_corrections=3)
    # Should still attempt to solve; may or may not succeed
    assert isinstance(result["solved"], bool)
