"""Experiment runners for the three required experiments."""

import os
import time
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.csp import SudokuCSP
from src.search import CorrectionSearchProblem, search
from src.puzzle_gen import generate_puzzle, introduce_misreads
from src.integration import solve_from_grid

EXPERIMENTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "experiments")


def _ensure_dir():
    os.makedirs(EXPERIMENTS_DIR, exist_ok=True)


def _make_probs_from_solution(puzzle, solution, corrupted, seed=42):
    """Create synthetic probability matrix mimicking NN output."""
    rng = np.random.default_rng(seed)
    probs = np.full((81, 9), 0.01, dtype=np.float32)
    for r in range(9):
        for c in range(9):
            idx = r * 9 + c
            d = corrupted[r, c]
            if d == 0:
                continue
            probs[idx, d - 1] = 0.70 + rng.uniform(0, 0.15)
            true_d = solution[r, c]
            if true_d != d:
                probs[idx, true_d - 1] = 0.10 + rng.uniform(0, 0.10)
    probs = probs / probs.sum(axis=1, keepdims=True)
    return probs


def run_experiment_1(seeds=None, misread_levels=None, n_puzzles=100, n_givens=30, verbose=True):
    """Experiment 1: UCS vs Greedy vs A* -- nodes and recovery rate for 0-4 misreads."""
    _ensure_dir()
    if seeds is None:
        seeds = [1, 2, 3, 4, 5]
    if misread_levels is None:
        misread_levels = [0, 1, 2, 3, 4]
    strategies = ["UCS", "Greedy", "A*"]

    csv_path = os.path.join(EXPERIMENTS_DIR, "exp1_results.csv")
    rows = []

    for seed in seeds:
        for n_misreads in misread_levels:
            for strategy in strategies:
                expanded_list = []
                recovered = 0

                for puzzle_idx in range(n_puzzles):
                    p_seed = seed * 10000 + puzzle_idx
                    puzzle, solution = generate_puzzle(n_givens, seed=p_seed)

                    if n_misreads > 0:
                        corrupted, _ = introduce_misreads(puzzle, solution, n_misreads, seed=p_seed + 1)
                    else:
                        corrupted = puzzle

                    probs = _make_probs_from_solution(puzzle, solution, corrupted, seed=p_seed + 2)
                    result = solve_from_grid(corrupted, probs, "BT+FC+AC3", strategy, max_corrections=3)

                    expanded_list.append(result["search_stats"]["nodes_expanded"])
                    if result["solved"] and np.array_equal(result["solution"], solution):
                        recovered += 1

                rows.append({
                    "seed": seed,
                    "misreads": n_misreads,
                    "strategy": strategy,
                    "mean_nodes": np.mean(expanded_list),
                    "std_nodes": np.std(expanded_list),
                    "recovery_rate": recovered / n_puzzles,
                    "n_puzzles": n_puzzles,
                })

                if verbose:
                    print(f"  Exp1: seed={seed}, misreads={n_misreads}, {strategy}: "
                          f"nodes={np.mean(expanded_list):.1f}\u00b1{np.std(expanded_list):.1f}, "
                          f"recovery={recovered}/{n_puzzles}")

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    _plot_experiment_1(csv_path, misread_levels, strategies)
    return csv_path


def _plot_experiment_1(csv_path, misread_levels, strategies):
    import csv as csv_module
    with open(csv_path) as f:
        reader = csv_module.DictReader(f)
        data = list(reader)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for strategy in strategies:
        means = []
        stds = []
        recovery = []
        for ml in misread_levels:
            subset = [r for r in data if int(r["misreads"]) == ml and r["strategy"] == strategy]
            means.append(np.mean([float(r["mean_nodes"]) for r in subset]))
            stds.append(np.mean([float(r["std_nodes"]) for r in subset]))
            recovery.append(np.mean([float(r["recovery_rate"]) for r in subset]))
        ax1.errorbar(misread_levels, means, yerr=stds, marker="o", label=strategy, capsize=3)
        ax2.plot(misread_levels, recovery, marker="s", label=strategy)

    ax1.set_xlabel("Number of Misreads")
    ax1.set_ylabel("Nodes Expanded (mean \u00b1 std)")
    ax1.set_title("Exp 1: Search Nodes by Strategy")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel("Number of Misreads")
    ax2.set_ylabel("Recovery Rate")
    ax2.set_title("Exp 1: Recovery Rate by Strategy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-0.05, 1.05)

    plt.tight_layout()
    plt.savefig(os.path.join(EXPERIMENTS_DIR, "exp1_plot.png"), dpi=150)
    plt.close()


def run_experiment_2(seeds=None, given_levels=None, n_puzzles=100, verbose=True):
    """Experiment 2: BT vs BT+FC vs BT+FC+AC3 on puzzles with 25, 30, 35 givens."""
    _ensure_dir()
    if seeds is None:
        seeds = [1, 2, 3, 4, 5]
    if given_levels is None:
        given_levels = [25, 30, 35]
    methods = ["BT", "BT+FC", "BT+FC+AC3"]

    csv_path = os.path.join(EXPERIMENTS_DIR, "exp2_results.csv")
    rows = []

    for seed in seeds:
        for n_givens in given_levels:
            for method in methods:
                bt_list = []
                assign_list = []
                time_list = []

                for puzzle_idx in range(n_puzzles):
                    p_seed = seed * 10000 + puzzle_idx
                    puzzle, solution = generate_puzzle(n_givens, seed=p_seed)
                    csp = SudokuCSP(puzzle)

                    t0 = time.time()
                    sol, stats = csp.solve(method)
                    elapsed = time.time() - t0

                    bt_list.append(stats["backtracks"])
                    assign_list.append(stats["assignments"])
                    time_list.append(elapsed)

                rows.append({
                    "seed": seed,
                    "givens": n_givens,
                    "method": method,
                    "mean_backtracks": np.mean(bt_list),
                    "std_backtracks": np.std(bt_list),
                    "mean_assignments": np.mean(assign_list),
                    "std_assignments": np.std(assign_list),
                    "mean_time": np.mean(time_list),
                    "n_puzzles": n_puzzles,
                })

                if verbose:
                    print(f"  Exp2: seed={seed}, givens={n_givens}, {method}: "
                          f"bt={np.mean(bt_list):.1f}\u00b1{np.std(bt_list):.1f}, "
                          f"assign={np.mean(assign_list):.1f}")

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    _plot_experiment_2(csv_path, given_levels, methods)
    return csv_path


def _plot_experiment_2(csv_path, given_levels, methods):
    import csv as csv_module
    with open(csv_path) as f:
        reader = csv_module.DictReader(f)
        data = list(reader)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    x = np.arange(len(given_levels))
    width = 0.25

    for i, method in enumerate(methods):
        bt_means = []
        assign_means = []
        for gl in given_levels:
            subset = [r for r in data if int(r["givens"]) == gl and r["method"] == method]
            bt_means.append(np.mean([float(r["mean_backtracks"]) for r in subset]))
            assign_means.append(np.mean([float(r["mean_assignments"]) for r in subset]))
        ax1.bar(x + i * width, bt_means, width, label=method)
        ax2.bar(x + i * width, assign_means, width, label=method)

    ax1.set_xlabel("Number of Givens")
    ax1.set_ylabel("Backtracks (mean)")
    ax1.set_title("Exp 2: Backtracks by CSP Method")
    ax1.set_xticks(x + width)
    ax1.set_xticklabels(given_levels)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis="y")

    ax2.set_xlabel("Number of Givens")
    ax2.set_ylabel("Assignments (mean)")
    ax2.set_title("Exp 2: Assignments by CSP Method")
    ax2.set_xticks(x + width)
    ax2.set_xticklabels(given_levels)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(os.path.join(EXPERIMENTS_DIR, "exp2_plot.png"), dpi=150)
    plt.close()
