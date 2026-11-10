# Handwritten Sudoku Reader & Solver

A system that reads handwritten Sudoku puzzles, detects misread digits using a neural network, corrects them using search algorithms, and solves the puzzle with constraint satisfaction.

## Setup

```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac

pip install -r requirements.txt
```

## Requirements

- Python 3.10+
- See `requirements.txt` for pinned dependencies

## Quick Start

### 1. Download and preprocess data

```bash
python make_data.py
```

Downloads MNIST, filters to digits 1-9, creates train/val/test splits in `data/processed/`.

### 2. Run the demo

```bash
python run_demo.py --seed 42 --n_givens 30 --misreads 2
```

This will:
- Train (or load) the neural network
- Generate a Sudoku puzzle with 30 givens
- Inject 2 misread digits
- Create a mosaic image of handwritten digits
- Run the full pipeline: NN reading → conflict detection → correction search → CSP solve

### 3. Run tests

```bash
python -m pytest tests/ -v
```

### 4. Run experiments

```python
from src.eval import run_nn_comparison, run_experiment_1, run_experiment_2, run_experiment_3

run_nn_comparison()  # NN vs logistic regression baseline
run_experiment_1()   # UCS vs Greedy vs A*
run_experiment_2()   # BT vs BT+FC vs BT+FC+AC3
run_experiment_3()   # NN-guided vs no correction vs random
```

Results saved to `experiments/`.

## CLI Options

```
python run_demo.py [options]

  --seed INT          Random seed (default: 42)
  --n_givens INT      Number of given cells (default: 30)
  --misreads INT      Number of injected misreads (default: 2)
  --csp METHOD        CSP solver: BT, BT+FC, BT+FC+AC3 (default: BT+FC+AC3)
  --search STRATEGY   Search: UCS, Greedy, A* (default: A*)
  --model_path PATH   Path to saved NN model (default: models/nn_model.pkl)
```

## Project Structure

```
src/
  data.py          - Data loading and preprocessing
  nn.py            - Pure NumPy feedforward neural network (784→256→128→9)
  baseline.py      - Logistic regression baseline
  csp.py           - Sudoku CSP solver (BT, FC, AC-3)
  search.py        - Correction search (UCS, Greedy, A*)
  puzzle_gen.py    - Puzzle and mosaic generators
  integration.py   - Full pipeline orchestration
  eval.py          - Experiment runners and plotting
tests/             - Test suite (5 normal + 3 edge + 2 impossible per module)
experiments/       - CSV results and plots
```

## Data Dictionary

| Field | Type | Description |
|-------|------|-------------|
| X_train/val/test | (N, 28, 28) float32 | Digit images, normalized 0-1 |
| y_train/val/test | (N,) int | Labels 1-9 |
| puzzle | (9, 9) int | Sudoku grid, 0 = empty |
| solution | (9, 9) int | Complete valid Sudoku |
| mosaic | (252, 252) float32 | 9×9 grid of 28×28 digit images |
| probabilities | (81, 9) float32 | NN output per cell, P(digit=d) |
| correction | dict {cell, read_digit, new_digit, cost} | One correction action |

## Dataset

- **Source**: MNIST (via scikit-learn's `fetch_openml`)
- **Original**: 60k train + 10k test, 28×28 grayscale, digits 0-9
- **Processed**: Filtered to digits 1-9, normalized, split 50k/10k/3k
- **Puzzles**: Generated (not real-world photographs)
