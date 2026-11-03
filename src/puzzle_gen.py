"""Sudoku puzzle generator, misread injector, and mosaic image generator."""

import numpy as np
from src.csp import SudokuCSP, is_valid_complete


def generate_full_grid(seed=42):
    """Generate a complete valid Sudoku grid using randomized backtracking."""
    rng = np.random.default_rng(seed)
    grid = np.zeros((9, 9), dtype=int)

    def _fill(pos):
        if pos == 81:
            return True
        r, c = pos // 9, pos % 9
        digits = list(range(1, 10))
        rng.shuffle(digits)
        for d in digits:
            if _can_place(grid, r, c, d):
                grid[r, c] = d
                if _fill(pos + 1):
                    return True
                grid[r, c] = 0
        return False

    _fill(0)
    return grid


def _can_place(grid, r, c, d):
    if d in grid[r, :]:
        return False
    if d in grid[:, c]:
        return False
    br, bc = 3 * (r // 3), 3 * (c // 3)
    if d in grid[br:br+3, bc:bc+3]:
        return False
    return True


def _has_unique_solution(puzzle):
    """Check if puzzle has exactly one solution (use CSP to count up to 2)."""
    solutions = [0]

    def _count(grid, pos):
        if solutions[0] >= 2:
            return
        if pos == 81:
            solutions[0] += 1
            return
        r, c = pos // 9, pos % 9
        if grid[r, c] != 0:
            _count(grid, pos + 1)
            return
        for d in range(1, 10):
            if _can_place(grid, r, c, d):
                grid[r, c] = d
                _count(grid, pos + 1)
                grid[r, c] = 0

    _count(puzzle.copy(), 0)
    return solutions[0] == 1


def generate_puzzle(n_givens, seed=42):
    """Generate a puzzle with n_givens filled cells and a unique solution."""
    rng = np.random.default_rng(seed)
    solution = generate_full_grid(seed)
    puzzle = solution.copy()

    filled = list(range(81))
    rng.shuffle(filled)

    removed = 0
    target_removals = 81 - n_givens

    for pos in filled:
        if removed >= target_removals:
            break
        r, c = pos // 9, pos % 9
        old_val = puzzle[r, c]
        if old_val == 0:
            continue
        puzzle[r, c] = 0
        if _has_unique_solution(puzzle):
            removed += 1
        else:
            puzzle[r, c] = old_val

    return puzzle, solution


CONFUSION_PAIRS = {
    1: [7, 2],
    2: [3, 7],
    3: [5, 8],
    4: [9, 6],
    5: [3, 6],
    6: [4, 5],
    7: [1, 9],
    8: [3, 9],
    9: [4, 7],
}


def introduce_misreads(puzzle, solution, n_misreads, seed=42):
    """Replace n_misreads given cells with commonly-confused digits.

    Uses a documented confusion rule: each digit maps to its most likely
    misclassifications (based on typical NN confusion patterns for MNIST).
    """
    rng = np.random.default_rng(seed)
    corrupted = puzzle.copy()

    given_cells = [(r, c) for r in range(9) for c in range(9) if puzzle[r, c] != 0]
    rng.shuffle(given_cells)

    log = []
    for r, c in given_cells:
        if len(log) >= n_misreads:
            break
        true_digit = puzzle[r, c]
        confusable = CONFUSION_PAIRS.get(true_digit, [d for d in range(1, 10) if d != true_digit])
        wrong_digit = rng.choice(confusable)
        corrupted[r, c] = wrong_digit
        log.append((r, c, true_digit, wrong_digit))

    return corrupted, log


def generate_mosaic(puzzle, digit_images, seed=42):
    """Create a 252x252 mosaic image from a puzzle grid using MNIST digit images.

    Args:
        puzzle: 9x9 int array (0 = empty)
        digit_images: dict mapping digit (1-9) to array of images (N, 28, 28)
        seed: random seed for image selection
    """
    rng = np.random.default_rng(seed)
    mosaic = np.zeros((252, 252), dtype=np.float32)

    for r in range(9):
        for c in range(9):
            digit = puzzle[r, c]
            y_start = r * 28
            x_start = c * 28
            if digit == 0:
                mosaic[y_start:y_start+28, x_start:x_start+28] = 0.0
            else:
                images = digit_images[digit]
                idx = rng.integers(0, len(images))
                mosaic[y_start:y_start+28, x_start:x_start+28] = images[idx]

    return mosaic


def mosaic_to_grid(mosaic):
    """Split a 252x252 mosaic into 81 individual 28x28 cell images."""
    cells = []
    for r in range(9):
        for c in range(9):
            y_start = r * 28
            x_start = c * 28
            cell = mosaic[y_start:y_start+28, x_start:x_start+28]
            cells.append(cell)
    return cells


def is_blank_cell(image, threshold=0.01):
    """Detect if a cell image is blank (empty cell) by average brightness."""
    return image.mean() < threshold
