import numpy as np
from src.csp import SudokuCSP, is_valid_complete, find_conflicts


EASY_PUZZLE = np.array([
    [5, 3, 0, 0, 7, 0, 0, 0, 0],
    [6, 0, 0, 1, 9, 5, 0, 0, 0],
    [0, 9, 8, 0, 0, 0, 0, 6, 0],
    [8, 0, 0, 0, 6, 0, 0, 0, 3],
    [4, 0, 0, 8, 0, 3, 0, 0, 1],
    [7, 0, 0, 0, 2, 0, 0, 0, 6],
    [0, 6, 0, 0, 0, 0, 2, 8, 0],
    [0, 0, 0, 4, 1, 9, 0, 0, 5],
    [0, 0, 0, 0, 8, 0, 0, 7, 9],
])

EASY_SOLUTION = np.array([
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

# Medium puzzle (~28 givens)
MEDIUM_PUZZLE = np.array([
    [0, 0, 0, 2, 6, 0, 7, 0, 1],
    [6, 8, 0, 0, 7, 0, 0, 9, 0],
    [1, 9, 0, 0, 0, 4, 5, 0, 0],
    [8, 2, 0, 1, 0, 0, 0, 4, 0],
    [0, 0, 4, 6, 0, 2, 9, 0, 0],
    [0, 5, 0, 0, 0, 3, 0, 2, 8],
    [0, 0, 9, 3, 0, 0, 0, 7, 4],
    [0, 4, 0, 0, 5, 0, 0, 3, 6],
    [7, 0, 3, 0, 1, 8, 0, 0, 0],
])

# Hard puzzle (17 givens - near minimum)
HARD_PUZZLE = np.array([
    [0, 0, 0, 0, 0, 0, 0, 1, 0],
    [4, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 2, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 5, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 3, 0],
    [0, 0, 1, 0, 0, 0, 0, 0, 8],
    [5, 0, 0, 0, 6, 0, 9, 0, 0],
    [0, 3, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 7, 0, 0, 0, 0, 2],
])


# --- Normal cases ---

def test_easy_puzzle():
    csp = SudokuCSP(EASY_PUZZLE)
    sol, stats = csp.solve("BT+FC+AC3")
    assert sol is not None
    assert np.array_equal(sol, EASY_SOLUTION)
    assert stats["solved"]


def test_medium_puzzle():
    csp = SudokuCSP(MEDIUM_PUZZLE)
    sol, stats = csp.solve("BT+FC+AC3")
    assert sol is not None
    assert is_valid_complete(sol)


def test_hard_puzzle():
    csp = SudokuCSP(HARD_PUZZLE)
    sol, stats = csp.solve("BT+FC+AC3")
    assert sol is not None
    assert is_valid_complete(sol)


def test_all_methods_agree():
    solutions = []
    for method in ["BT", "BT+FC", "BT+FC+AC3"]:
        csp = SudokuCSP(EASY_PUZZLE)
        sol, _ = csp.solve(method)
        assert sol is not None
        solutions.append(sol)
    assert np.array_equal(solutions[0], solutions[1])
    assert np.array_equal(solutions[1], solutions[2])


def test_counters_ordering():
    bt_stats_list = []
    for method in ["BT", "BT+FC", "BT+FC+AC3"]:
        csp = SudokuCSP(MEDIUM_PUZZLE)
        _, stats = csp.solve(method)
        bt_stats_list.append(stats["backtracks"])
    # BT >= FC >= AC3 in backtracks (typically)
    assert bt_stats_list[0] >= bt_stats_list[1]


# --- Edge cases ---

def test_already_solved():
    csp = SudokuCSP(EASY_SOLUTION)
    sol, stats = csp.solve("BT+FC+AC3")
    assert sol is not None
    assert np.array_equal(sol, EASY_SOLUTION)
    assert stats["backtracks"] == 0


def test_single_empty_cell():
    puzzle = EASY_SOLUTION.copy()
    puzzle[0, 2] = 0  # remove the 4
    csp = SudokuCSP(puzzle)
    sol, stats = csp.solve("BT+FC+AC3")
    assert sol is not None
    assert sol[0, 2] == 4


def test_hard_17_givens():
    csp = SudokuCSP(HARD_PUZZLE)
    sol, stats = csp.solve("BT+FC+AC3")
    assert sol is not None
    assert is_valid_complete(sol)
    assert stats["assignments"] > 0


# --- Impossible cases ---

def test_contradictory_row():
    puzzle = np.zeros((9, 9), dtype=int)
    puzzle[0, 0] = 5
    puzzle[0, 3] = 5  # two 5s in row 0
    csp = SudokuCSP(puzzle)
    sol, stats = csp.solve("BT+FC+AC3")
    assert sol is None
    assert not stats["solved"]


def test_no_valid_solution():
    puzzle = np.zeros((9, 9), dtype=int)
    # Fill row 0 with conflicting values to make unsolvable
    puzzle[0, 0] = 1
    puzzle[0, 1] = 2
    puzzle[0, 2] = 3
    puzzle[1, 0] = 4
    puzzle[1, 1] = 5
    puzzle[1, 2] = 6
    puzzle[2, 0] = 7
    puzzle[2, 1] = 8
    puzzle[2, 2] = 9
    # Now force a contradiction in box
    puzzle[3, 0] = 1
    puzzle[3, 1] = 2
    puzzle[3, 2] = 4  # valid so far
    puzzle[4, 0] = 5
    puzzle[4, 1] = 3
    puzzle[4, 2] = 8
    puzzle[5, 0] = 6
    puzzle[5, 1] = 9
    puzzle[5, 2] = 7
    # Force column conflict
    puzzle[6, 0] = 2
    puzzle[7, 0] = 3
    puzzle[8, 0] = 8
    puzzle[6, 1] = 1  # conflicts: col 1 has 2,5,8,9,1 but we need more
    puzzle[6, 2] = 5
    puzzle[7, 2] = 2  # col 2 conflict: 3,6,8,4,7,5,2 - 9 values for 9 slots, might still work
    # Use a known unsolvable: two same digits in a column
    puzzle2 = np.zeros((9, 9), dtype=int)
    puzzle2[0, 0] = 1
    puzzle2[8, 0] = 1  # two 1s in column 0
    csp = SudokuCSP(puzzle2)
    sol, stats = csp.solve("BT+FC+AC3")
    assert sol is None


# --- Utility function tests ---

def test_is_valid_complete():
    assert is_valid_complete(EASY_SOLUTION)
    bad = EASY_SOLUTION.copy()
    bad[0, 0] = bad[0, 1]  # duplicate in row
    assert not is_valid_complete(bad)


def test_find_conflicts():
    puzzle = EASY_SOLUTION.copy()
    assert find_conflicts(puzzle) == []

    puzzle[0, 0] = puzzle[0, 1]  # make conflict
    conflicts = find_conflicts(puzzle)
    assert (0, 0) in conflicts
    assert (0, 1) in conflicts
