import numpy as np
from src.puzzle_gen import generate_mosaic, mosaic_to_grid, is_blank_cell


def _make_dummy_puzzle():
    """Create a simple puzzle for mosaic testing."""
    puzzle = np.zeros((9, 9), dtype=int)
    puzzle[0, 0] = 5
    puzzle[0, 4] = 7
    puzzle[1, 1] = 6
    puzzle[2, 2] = 8
    puzzle[4, 4] = 9
    puzzle[6, 6] = 2
    puzzle[8, 8] = 1
    return puzzle


def test_mosaic_shape():
    puzzle = _make_dummy_puzzle()
    digit_images = {}
    rng = np.random.default_rng(0)
    for d in range(1, 10):
        digit_images[d] = rng.random((10, 28, 28)).astype(np.float32)
    mosaic = generate_mosaic(puzzle, digit_images, seed=42)
    assert mosaic.shape == (252, 252)
    assert mosaic.dtype == np.float32


def test_mosaic_to_grid_roundtrip():
    puzzle = _make_dummy_puzzle()
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
