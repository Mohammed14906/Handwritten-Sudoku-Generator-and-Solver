import os
import csv
from src.eval import run_experiment_1, run_experiment_2, run_experiment_3, EXPERIMENTS_DIR


def test_experiment_1_runs():
    path = run_experiment_1(seeds=[1], misread_levels=[0, 1], n_puzzles=2, verbose=False)
    assert os.path.exists(path)
    with open(path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    assert len(rows) > 0
    assert "strategy" in rows[0]
    assert "recovery_rate" in rows[0]
    assert os.path.exists(os.path.join(EXPERIMENTS_DIR, "exp1_plot.png"))


def test_experiment_2_runs():
    path = run_experiment_2(seeds=[1], given_levels=[30, 35], n_puzzles=2, verbose=False)
    assert os.path.exists(path)
    with open(path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    assert len(rows) > 0
    assert "method" in rows[0]
    assert "mean_backtracks" in rows[0]
    assert os.path.exists(os.path.join(EXPERIMENTS_DIR, "exp2_plot.png"))


def test_experiment_3_runs():
    path = run_experiment_3(seeds=[1], misread_levels=[0, 1], n_puzzles=2, verbose=False)
    assert os.path.exists(path)
    with open(path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    assert len(rows) > 0
    assert "mode" in rows[0]
    assert os.path.exists(os.path.join(EXPERIMENTS_DIR, "exp3_plot.png"))


def test_reproducibility():
    path1 = run_experiment_1(seeds=[42], misread_levels=[1], n_puzzles=3, verbose=False)
    with open(path1) as f:
        rows1 = list(csv.DictReader(f))
    path2 = run_experiment_1(seeds=[42], misread_levels=[1], n_puzzles=3, verbose=False)
    with open(path2) as f:
        rows2 = list(csv.DictReader(f))
    for r1, r2 in zip(rows1, rows2):
        assert r1["mean_nodes"] == r2["mean_nodes"]
        assert r1["recovery_rate"] == r2["recovery_rate"]
