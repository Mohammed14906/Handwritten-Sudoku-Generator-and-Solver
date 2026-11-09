"""Integration pipeline: NN → AC-3 conflict detection → Search correction → CSP solve."""

import numpy as np
from src import nn as nn_module
from src.csp import SudokuCSP, find_conflicts
from src.search import CorrectionSearchProblem, search
from src.puzzle_gen import mosaic_to_grid, is_blank_cell


def solve_from_mosaic(mosaic, model, csp_method="BT+FC+AC3",
                      search_strategy="A*", max_corrections=3,
                      callback=None):
    """Full pipeline: mosaic image → solved puzzle.

    Args:
        callback: optional function(step_name, data_dict) called at each stage.
    """
    cell_images = mosaic_to_grid(mosaic)

    grid = np.zeros((9, 9), dtype=int)
    probabilities = np.zeros((81, 9), dtype=np.float32)
    confidences = np.zeros((9, 9), dtype=np.float32)

    digit_cells = []
    blank_cells = []
    for i, img in enumerate(cell_images):
        if is_blank_cell(img):
            probabilities[i] = np.ones(9) / 9
            blank_cells.append(i)
        else:
            digit_cells.append(i)

    if digit_cells:
        digit_images_arr = np.array([cell_images[i] for i in digit_cells])
        probs = nn_module.predict_proba(model, digit_images_arr)
        preds = probs.argmax(axis=1) + 1

        for j, i in enumerate(digit_cells):
            r, c = i // 9, i % 9
            grid[r, c] = preds[j]
            probabilities[i] = probs[j]
            confidences[r, c] = probs[j].max()

    if callback:
        top2 = np.zeros((81, 2), dtype=int)
        top2_probs = np.zeros((81, 2), dtype=np.float32)
        for i in range(81):
            sorted_idx = np.argsort(probabilities[i])[::-1]
            top2[i] = sorted_idx[:2] + 1
            top2_probs[i] = probabilities[i, sorted_idx[:2]]
        callback("nn_reading", {
            "grid": grid.copy(),
            "probabilities": probabilities.copy(),
            "confidences": confidences.copy(),
            "digit_cells": digit_cells,
            "blank_cells": blank_cells,
            "cell_images": cell_images,
            "top2_digits": top2,
            "top2_probs": top2_probs,
        })

    return solve_from_grid(grid, probabilities, csp_method, search_strategy,
                           max_corrections, callback=callback)


def solve_from_grid(grid, probabilities, csp_method="BT+FC+AC3",
                    search_strategy="A*", max_corrections=3,
                    callback=None):
    """Pipeline without image step: grid + probabilities → solved puzzle."""
    grid = np.array(grid, dtype=int)
    nn_grid = grid.copy()

    conflicts = find_conflicts(grid)
    corrections = frozenset()
    search_stats = {"nodes_expanded": 0, "nodes_generated": 0, "solution_cost": 0.0, "found": True}

    if callback:
        conflict_units = _classify_conflicts(grid, conflicts)
        callback("conflict_detection", {
            "conflicts": list(conflicts),
            "conflict_units": conflict_units,
            "n_conflicts": len(conflicts),
            "grid": grid.copy(),
        })

    if conflicts:
        problem = CorrectionSearchProblem(grid, probabilities, max_corrections)
        result, search_stats = search(problem, search_strategy)

        if callback:
            callback("search_correction", {
                "search_stats": search_stats,
                "strategy": search_strategy,
                "corrections_found": result,
                "max_corrections": max_corrections,
            })

        if search_stats["found"] and result is not None:
            corrections = result
            for (r, c, new_digit) in corrections:
                grid[r, c] = new_digit
            solution = getattr(problem, "last_solution", None)
            csp_stats = getattr(problem, "last_csp_stats", {})
            if solution is None:
                csp = SudokuCSP(grid)
                solution, csp_stats = csp.solve(csp_method)
        else:
            least_confident = _find_least_confident(nn_grid, probabilities)
            if callback:
                callback("csp_solve", {
                    "method": csp_method,
                    "csp_stats": {},
                    "solved": False,
                    "grid_in": grid.copy(),
                })
            return {
                "solution": None,
                "nn_grid": nn_grid,
                "corrections": [],
                "correction_log": "UNREADABLE: No valid correction found within budget.",
                "least_confident_cells": least_confident,
                "csp_stats": {},
                "search_stats": search_stats,
                "solved": False,
            }
    else:
        if callback:
            callback("search_correction", {
                "search_stats": search_stats,
                "strategy": search_strategy,
                "corrections_found": frozenset(),
                "max_corrections": max_corrections,
                "skipped": True,
            })
        csp = SudokuCSP(grid)
        solution, csp_stats = csp.solve(csp_method)

    if callback:
        n_empty = int(np.sum(grid == 0))
        callback("csp_solve", {
            "method": csp_method,
            "csp_stats": csp_stats,
            "solved": solution is not None,
            "grid_in": grid.copy(),
            "solution": solution.copy() if solution is not None else None,
            "empty_cells": n_empty,
        })

    correction_log = []
    for (r, c, new_digit) in corrections:
        old_digit = nn_grid[r, c]
        cell_idx = r * 9 + c
        cost = max(
            probabilities[cell_idx, old_digit - 1] - probabilities[cell_idx, new_digit - 1],
            1e-6
        )
        correction_log.append({
            "cell": (r, c),
            "read_digit": int(old_digit),
            "new_digit": int(new_digit),
            "cost": float(cost),
        })

    return {
        "solution": solution,
        "nn_grid": nn_grid,
        "corrections": correction_log,
        "correction_log": _format_log(correction_log),
        "csp_stats": csp_stats,
        "search_stats": search_stats,
        "solved": solution is not None,
    }


def _classify_conflicts(grid, conflicts):
    """Return which unit(s) each conflicting cell participates in."""
    units = {}
    for (r, c) in conflicts:
        cell_units = []
        d = grid[r, c]
        for c2 in range(9):
            if c2 != c and grid[r, c2] == d:
                cell_units.append(f"row {r}")
                break
        for r2 in range(9):
            if r2 != r and grid[r2, c] == d:
                cell_units.append(f"col {c}")
                break
        br, bc = 3 * (r // 3), 3 * (c // 3)
        for dr in range(3):
            for dc in range(3):
                nr, nc = br + dr, bc + dc
                if (nr, nc) != (r, c) and grid[nr, nc] == d:
                    cell_units.append(f"box ({br},{bc})")
                    break
            else:
                continue
            break
        units[(r, c)] = cell_units
    return units


def _find_least_confident(grid, probabilities, n=5):
    """Return the n given cells where the NN was least confident."""
    cells = []
    for r in range(9):
        for c in range(9):
            if grid[r, c] == 0:
                continue
            idx = r * 9 + c
            confidence = probabilities[idx].max()
            cells.append({"cell": (r, c), "digit": int(grid[r, c]), "confidence": float(confidence)})
    cells.sort(key=lambda x: x["confidence"])
    return cells[:n]


def _format_log(correction_log):
    if not correction_log:
        return "No corrections needed."
    lines = []
    for entry in correction_log:
        r, c = entry["cell"]
        lines.append(
            f"  Cell ({r},{c}): {entry['read_digit']} -> {entry['new_digit']} (cost={entry['cost']:.4f})"
        )
    return "Corrections applied:\n" + "\n".join(lines)
