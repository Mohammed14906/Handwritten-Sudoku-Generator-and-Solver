"""Sudoku mosaic image generator."""

import numpy as np


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
