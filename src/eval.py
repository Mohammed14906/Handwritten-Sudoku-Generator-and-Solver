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


def run_nn_comparison(verbose=True):
    """Compare NN vs logistic regression baseline. Saves confusion matrix and metrics."""
    _ensure_dir()
    from src.data import load_splits
    from src import nn as nn_module
    from src import baseline as baseline_module

    X_train, y_train, X_val, y_val, X_test, y_test = load_splits()

    if verbose:
        print("Training neural network...")
    nn_model = nn_module.train(X_train, y_train, X_val, y_val,
                               epochs=30, lr=0.05, batch_size=128, seed=42, verbose=False)

    if verbose:
        print("Training logistic regression baseline...")
    lr_model = baseline_module.train(X_train, y_train, X_val, y_val, seed=42, verbose=False)

    nn_acc = nn_module.accuracy(nn_model, X_test, y_test)
    lr_acc = (baseline_module.predict(lr_model, X_test) == y_test).mean()
    nn_ece = nn_module.expected_calibration_error(nn_model, X_test, y_test)
    nn_cm = nn_module.confusion_matrix(nn_model, X_test, y_test)

    if verbose:
        print(f"\n--- NN vs Baseline Comparison ---")
        print(f"Neural Network  - Accuracy: {nn_acc:.4f}, ECE: {nn_ece:.4f}")
        print(f"Logistic Regr.  - Accuracy: {lr_acc:.4f}")

    # Save confusion matrix plot
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(nn_cm, cmap="Blues")
    ax.set_xticks(range(9))
    ax.set_yticks(range(9))
    ax.set_xticklabels(range(1, 10))
    ax.set_yticklabels(range(1, 10))
    ax.set_xlabel("Predicted Digit")
    ax.set_ylabel("True Digit")
    ax.set_title(f"NN Confusion Matrix (acc={nn_acc:.4f}, ECE={nn_ece:.4f})")
    for i in range(9):
        for j in range(9):
            val = nn_cm[i, j]
            color = "white" if val > nn_cm.max() / 2 else "black"
            ax.text(j, i, str(val), ha="center", va="center", color=color, fontsize=8)
    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig(os.path.join(EXPERIMENTS_DIR, "nn_confusion_matrix.png"), dpi=150)
    plt.close()

    # Save comparison CSV
    csv_path = os.path.join(EXPERIMENTS_DIR, "nn_vs_baseline.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "accuracy", "ece"])
        writer.writerow(["neural_network", f"{nn_acc:.4f}", f"{nn_ece:.4f}"])
        writer.writerow(["logistic_regression", f"{lr_acc:.4f}", "N/A"])

    if verbose:
        print(f"Confusion matrix saved to experiments/nn_confusion_matrix.png")
        print(f"Comparison saved to experiments/nn_vs_baseline.csv")

    return nn_model, {"nn_acc": nn_acc, "lr_acc": lr_acc, "nn_ece": nn_ece}


def run_experiment_1(seeds=None, misread_levels=None, n_puzzles=100, n_givens=30, verbose=True):
    """Experiment 1: UCS vs Greedy vs A* — nodes and recovery rate for 0-4 misreads."""
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
                          f"nodes={np.mean(expanded_list):.1f}±{np.std(expanded_list):.1f}, "
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
    ax1.set_ylabel("Nodes Expanded (mean ± std)")
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
                          f"bt={np.mean(bt_list):.1f}±{np.std(bt_list):.1f}, "
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


def run_experiment_3(seeds=None, misread_levels=None, n_puzzles=100, n_givens=30, verbose=True):
    """Experiment 3: NN-guided vs no correction vs random correction."""
    _ensure_dir()
    if seeds is None:
        seeds = [1, 2, 3, 4, 5]
    if misread_levels is None:
        misread_levels = [0, 1, 2, 3, 4]
    modes = ["nn_guided", "no_correction", "random"]

    csv_path = os.path.join(EXPERIMENTS_DIR, "exp3_results.csv")
    rows = []

    for seed in seeds:
        for n_misreads in misread_levels:
            for mode in modes:
                recovered = 0

                for puzzle_idx in range(n_puzzles):
                    p_seed = seed * 10000 + puzzle_idx
                    puzzle, solution = generate_puzzle(n_givens, seed=p_seed)

                    if n_misreads > 0:
                        corrupted, _ = introduce_misreads(puzzle, solution, n_misreads, seed=p_seed + 1)
                    else:
                        corrupted = puzzle

                    probs = _make_probs_from_solution(puzzle, solution, corrupted, seed=p_seed + 2)

                    if mode == "nn_guided":
                        result = solve_from_grid(corrupted, probs, "BT+FC+AC3", "A*", max_corrections=3)
                    elif mode == "no_correction":
                        csp = SudokuCSP(corrupted)
                        sol, _ = csp.solve("BT+FC+AC3")
                        result = {"solution": sol, "solved": sol is not None}
                    elif mode == "random":
                        rng = np.random.default_rng(p_seed + 3)
                        rand_probs = rng.random((81, 9)).astype(np.float32)
                        rand_probs = rand_probs / rand_probs.sum(axis=1, keepdims=True)
                        result = solve_from_grid(corrupted, rand_probs, "BT+FC+AC3", "A*", max_corrections=3)

                    if result["solved"] and result.get("solution") is not None:
                        if np.array_equal(result["solution"], solution):
                            recovered += 1

                rows.append({
                    "seed": seed,
                    "misreads": n_misreads,
                    "mode": mode,
                    "recovery_rate": recovered / n_puzzles,
                    "n_puzzles": n_puzzles,
                })

                if verbose:
                    print(f"  Exp3: seed={seed}, misreads={n_misreads}, {mode}: "
                          f"recovery={recovered}/{n_puzzles}")

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    _plot_experiment_3(csv_path, misread_levels, modes)
    return csv_path


def _plot_experiment_3(csv_path, misread_levels, modes):
    import csv as csv_module
    with open(csv_path) as f:
        reader = csv_module.DictReader(f)
        data = list(reader)

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(misread_levels))
    width = 0.25
    labels = {"nn_guided": "NN-Guided", "no_correction": "No Correction", "random": "Random"}

    for i, mode in enumerate(modes):
        rates = []
        for ml in misread_levels:
            subset = [r for r in data if int(r["misreads"]) == ml and r["mode"] == mode]
            rates.append(np.mean([float(r["recovery_rate"]) for r in subset]))
        ax.bar(x + i * width, rates, width, label=labels[mode])

    ax.set_xlabel("Number of Misreads")
    ax.set_ylabel("Exact Recovery Rate")
    ax.set_title("Exp 3: Correction Strategy Comparison")
    ax.set_xticks(x + width)
    ax.set_xticklabels(misread_levels)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    plt.savefig(os.path.join(EXPERIMENTS_DIR, "exp3_plot.png"), dpi=150)
    plt.close()
