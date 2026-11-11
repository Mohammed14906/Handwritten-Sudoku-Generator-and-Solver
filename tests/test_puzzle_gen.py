import numpy as np
from src.csp import is_valid_complete
from src.puzzle_gen import (
    generate_full_grid, generate_puzzle, introduce_misreads,
    generate_mosaic, mosaic_to_grid, is_blank_cell
)


def test_full_grid_valid():
    grid = generate_full_grid(seed=42)
    assert is_valid_complete(grid)


def test_puzzle_valid():
    puzzle, solution = generate_puzzle(n_givens=30, seed=42)
    assert is_valid_complete(solution)
    for r in range(9):
        for c in range(9):
            if puzzle[r, c] != 0:
                assert puzzle[r, c] == solution[r, c]


def test_n_givens():
    puzzle, _ = generate_puzzle(n_givens=30, seed=42)
    assert np.count_nonzero(puzzle) == 30


def test_deterministic_seed():
    p1, s1 = generate_puzzle(n_givens=30, seed=123)
    p2, s2 = generate_puzzle(n_givens=30, seed=123)
    assert np.array_equal(p1, p2)
    assert np.array_equal(s1, s2)


def test_misread_count():
    puzzle, solution = generate_puzzle(n_givens=30, seed=42)
    corrupted, log = introduce_misreads(puzzle, solution, n_misreads=3, seed=42)
    assert len(log) == 3
    diff_count = np.sum(puzzle != corrupted)
    assert diff_count == 3


def test_misread_log_accurate():
    puzzle, solution = generate_puzzle(n_givens=30, seed=42)
    corrupted, log = introduce_misreads(puzzle, solution, n_misreads=2, seed=42)
    for r, c, true_digit, wrong_digit in log:
        assert puzzle[r, c] == true_digit
        assert corrupted[r, c] == wrong_digit
        assert true_digit != wrong_digit


def test_mosaic_shape():
    puzzle, _ = generate_puzzle(n_givens=30, seed=42)
    digit_images = {}
    rng = np.random.default_rng(0)
    for d in range(1, 10):
        digit_images[d] = rng.random((10, 28, 28)).astype(np.float32)
    mosaic = generate_mosaic(puzzle, digit_images, seed=42)
    assert mosaic.shape == (252, 252)
    assert mosaic.dtype == np.float32


def test_mosaic_to_grid_roundtrip():
    puzzle, _ = generate_puzzle(n_givens=30, seed=42)
    rng = np.random.default_rng(0)
    digit_images = {}
    for d in range(1, 10):
        digit_images[d] = rng.random((10, 28, 28)).astype(np.float32)
    mosaic = generate_mosaic(puzzle, digit_images, seed=42)
    cells = mosaic_to_grid(mosaic)
    assert len(cells) == 81
    for i, cell in enumerate(cells):
        assert cell.shape == (28, 28)
        r, c = i // 9, i % 9
        expected = mosaic[r*28:(r+1)*28, c*28:(c+1)*28]
        assert np.array_equal(cell, expected)


def test_blank_detection():
    blank = np.zeros((28, 28), dtype=np.float32)
    assert is_blank_cell(blank)
    digit = np.random.default_rng(0).random((28, 28)).astype(np.float32)
    assert not is_blank_cell(digit)
