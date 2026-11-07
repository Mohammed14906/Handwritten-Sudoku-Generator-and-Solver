"""Correction search: UCS, Greedy, and A* for finding cheapest digit corrections."""

import heapq
import numpy as np
from src.csp import find_conflicts, count_conflict_groups, SudokuCSP


EPSILON = 1e-6


class CorrectionSearchProblem:
    def __init__(self, grid, probabilities, max_corrections=3, top_k=3):
        """
        Args:
            grid: 9x9 int array (NN's best guess, may have conflicts)
            probabilities: (81, 9) array where prob[i][d-1] = P(cell i is digit d)
            max_corrections: max number of cells to change
            top_k: number of top alternative digits to consider per cell
        """
        self.grid = np.array(grid, dtype=int)
        self.probabilities = np.array(probabilities)
        self.max_corrections = max_corrections
        self.top_k = top_k
        self.conflict_cells = self._get_conflict_cells()

    def _get_conflict_cells(self):
        conflicts = find_conflicts(self.grid)
        return set(conflicts)

    def initial_state(self):
        return frozenset()

    def is_goal(self, state):
        corrected = self._apply_corrections(state)
        if len(find_conflicts(corrected)) > 0:
            return False
        csp = SudokuCSP(corrected)
        sol, stats = csp.solve("BT+FC+AC3")
        if sol is not None:
            self.last_solution = sol
            self.last_csp_stats = stats
            return True
        return False

    def _apply_corrections(self, state):
        grid = self.grid.copy()
        for (r, c, new_digit) in state:
            grid[r, c] = new_digit
        return grid

    def actions(self, state):
        if len(state) >= self.max_corrections:
            return []

        corrected = self._apply_corrections(state)
        current_conflicts = find_conflicts(corrected)
        already_corrected = {(r, c) for (r, c, _) in state}

        acts = []
        for (r, c) in current_conflicts:
            if (r, c) in already_corrected:
                continue
            cell_idx = r * 9 + c
            current_digit = corrected[r, c]
            probs = self.probabilities[cell_idx].copy()
            probs[current_digit - 1] = -1  # exclude current digit
            top_indices = np.argsort(probs)[::-1][:self.top_k]
            for idx in top_indices:
                d = idx + 1
                cost = max(
                    self.probabilities[cell_idx, current_digit - 1]
                    - self.probabilities[cell_idx, d - 1],
                    EPSILON
                )
                acts.append(((r, c, d), cost))
        return acts

    def heuristic(self, state):
        corrected = self._apply_corrections(state)
        return count_conflict_groups(corrected)


def search(problem, strategy="UCS"):
    """
    Search for the cheapest correction set.

    Returns:
        (state, stats) where state is frozenset of corrections or None
        stats: {nodes_expanded, nodes_generated, solution_cost, found}
    """
    stats = {"nodes_expanded": 0, "nodes_generated": 0, "solution_cost": 0.0, "found": False}

    initial = problem.initial_state()

    if problem.is_goal(initial):
        stats["found"] = True
        return initial, stats

    # Priority queue: (priority, tie_breaker, cost_so_far, state)
    counter = 0
    if strategy == "UCS":
        start_priority = 0.0
    elif strategy == "Greedy":
        start_priority = problem.heuristic(initial)
    elif strategy == "A*":
        start_priority = problem.heuristic(initial)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    frontier = [(start_priority, counter, 0.0, initial)]
    visited = set()
    visited.add(initial)

    while frontier:
        priority, _, g, state = heapq.heappop(frontier)
        stats["nodes_expanded"] += 1

        if problem.is_goal(state):
            stats["found"] = True
            stats["solution_cost"] = g
            return state, stats

        for (action, action_cost) in problem.actions(state):
            r, c, d = action
            new_state = frozenset(state | {(r, c, d)})

            if new_state in visited:
                continue
            visited.add(new_state)

            new_g = g + action_cost
            stats["nodes_generated"] += 1

            if strategy == "UCS":
                new_priority = new_g
            elif strategy == "Greedy":
                new_priority = problem.heuristic(new_state)
            elif strategy == "A*":
                new_priority = new_g + problem.heuristic(new_state)

            counter += 1
            heapq.heappush(frontier, (new_priority, counter, new_g, new_state))

    return None, stats
