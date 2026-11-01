"""Sudoku CSP solver with Backtracking, Forward Checking, and AC-3."""

import numpy as np
from copy import deepcopy


class SudokuCSP:
    def __init__(self, grid):
        self.grid = np.array(grid, dtype=int)
        self.stats = {"assignments": 0, "backtracks": 0, "arc_checks": 0, "solved": False}
        self._init_domains()
        self._init_constraints()

    def _init_domains(self):
        self.domains = {}
        for r in range(9):
            for c in range(9):
                if self.grid[r, c] != 0:
                    self.domains[(r, c)] = {self.grid[r, c]}
                else:
                    self.domains[(r, c)] = set(range(1, 10))

    def _init_constraints(self):
        self.neighbors = {}
        for r in range(9):
            for c in range(9):
                nbrs = set()
                for i in range(9):
                    if i != c:
                        nbrs.add((r, i))
                    if i != r:
                        nbrs.add((i, c))
                br, bc = 3 * (r // 3), 3 * (c // 3)
                for dr in range(3):
                    for dc in range(3):
                        nr, nc = br + dr, bc + dc
                        if (nr, nc) != (r, c):
                            nbrs.add((nr, nc))
                self.neighbors[(r, c)] = nbrs


def is_valid_complete(grid):
    grid = np.array(grid, dtype=int)
    if grid.shape != (9, 9):
        return False
    if np.any(grid < 1) or np.any(grid > 9):
        return False
    for i in range(9):
        if len(set(grid[i, :])) != 9:
            return False
        if len(set(grid[:, i])) != 9:
            return False
    for br in range(0, 9, 3):
        for bc in range(0, 9, 3):
            block = grid[br:br+3, bc:bc+3].flatten()
            if len(set(block)) != 9:
                return False
    return True


def find_conflicts(grid):
    grid = np.array(grid, dtype=int)
    conflicts = set()
    for i in range(9):
        _check_unit(grid[i, :], [(i, c) for c in range(9)], conflicts)
        _check_unit(grid[:, i], [(r, i) for r in range(9)], conflicts)
    for br in range(0, 9, 3):
        for bc in range(0, 9, 3):
            cells = []
            for dr in range(3):
                for dc in range(3):
                    cells.append((br + dr, bc + dc))
            vals = [grid[r, c] for r, c in cells]
            _check_unit(vals, cells, conflicts)
    return list(conflicts)


def _check_unit(values, cells, conflicts):
    seen = {}
    for val, cell in zip(values, cells):
        if val == 0:
            continue
        if val in seen:
            conflicts.add(cell)
            conflicts.add(seen[val])
        else:
            seen[val] = cell
