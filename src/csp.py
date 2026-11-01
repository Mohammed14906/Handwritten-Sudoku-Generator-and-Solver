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

    def solve(self, method="BT"):
        self.stats = {"assignments": 0, "backtracks": 0, "arc_checks": 0, "solved": False}
        self._init_domains()

        use_fc = method in ("BT+FC", "BT+FC+AC3")
        use_ac3 = method == "BT+FC+AC3"

        if use_ac3:
            if not self._ac3():
                return None, self.stats

        assignment = {}
        for r in range(9):
            for c in range(9):
                if self.grid[r, c] != 0:
                    assignment[(r, c)] = self.grid[r, c]

        result = self._backtrack(assignment, use_fc, use_ac3)
        if result is not None:
            solution = np.zeros((9, 9), dtype=int)
            for (r, c), v in result.items():
                solution[r, c] = v
            self.stats["solved"] = True
            return solution, self.stats
        return None, self.stats

    def _select_unassigned(self, assignment):
        min_size = 10
        best = None
        for var in self.domains:
            if var not in assignment:
                size = len(self.domains[var])
                if size < min_size:
                    min_size = size
                    best = var
        return best

    def _backtrack(self, assignment, use_fc, use_ac3):
        if len(assignment) == 81:
            return assignment

        var = self._select_unassigned(assignment)
        if var is None:
            return None

        for value in sorted(self.domains[var]):
            if self._is_consistent(var, value, assignment):
                assignment[var] = value
                self.stats["assignments"] += 1

                old_domains = deepcopy(self.domains)
                self.domains[var] = {value}

                ok = True
                if use_fc:
                    ok = self._forward_check(var, value)
                if ok and use_ac3:
                    arcs = [(nb, var) for nb in self.neighbors[var]
                            if nb not in assignment]
                    ok = self._ac3(arcs)

                if ok:
                    result = self._backtrack(assignment, use_fc, use_ac3)
                    if result is not None:
                        return result

                self.stats["backtracks"] += 1
                del assignment[var]
                self.domains = old_domains

        return None

    def _is_consistent(self, var, value, assignment):
        for nb in self.neighbors[var]:
            if nb in assignment and assignment[nb] == value:
                return False
        return True

    def _forward_check(self, var, value):
        for nb in self.neighbors[var]:
            if len(self.domains[nb]) > 1 and value in self.domains[nb]:
                self.domains[nb] = self.domains[nb] - {value}
                if len(self.domains[nb]) == 0:
                    return False
        return True

    def _ac3(self, arcs=None):
        if arcs is None:
            arcs = []
            for var in self.domains:
                for nb in self.neighbors[var]:
                    arcs.append((var, nb))

        queue = list(arcs)
        while queue:
            xi, xj = queue.pop(0)
            if self._revise(xi, xj):
                if len(self.domains[xi]) == 0:
                    return False
                for xk in self.neighbors[xi]:
                    if xk != xj:
                        queue.append((xk, xi))
        return True

    def _revise(self, xi, xj):
        revised = False
        to_remove = set()
        for val in self.domains[xi]:
            self.stats["arc_checks"] += 1
            if not any(val != v2 for v2 in self.domains[xj]):
                to_remove.add(val)
                revised = True
        self.domains[xi] -= to_remove
        return revised


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


def count_conflict_groups(grid):
    """Count distinct (unit, digit) conflict groups. Each group needs >= 1 correction."""
    grid = np.array(grid, dtype=int)
    groups = set()
    for i in range(9):
        _count_unit_groups(grid[i, :], "row", i, groups)
        _count_unit_groups(grid[:, i], "col", i, groups)
    for br in range(0, 9, 3):
        for bc in range(0, 9, 3):
            vals = grid[br:br+3, bc:bc+3].flatten()
            _count_unit_groups(vals, "box", (br, bc), groups)
    return len(groups)


def _count_unit_groups(values, unit_type, unit_id, groups):
    seen = {}
    for val in values:
        if val == 0:
            continue
        if val in seen:
            groups.add((unit_type, unit_id, val))
        else:
            seen[val] = True


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
