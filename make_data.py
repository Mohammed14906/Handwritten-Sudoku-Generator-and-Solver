"""Download MNIST, filter to digits 1-9, split, and save as .npy files."""

import os
import numpy as np
from sklearn.datasets import fetch_openml

SEED = 42
DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "processed")


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    print("Fetching MNIST from OpenML...")
    mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
    X_all, y_all = mnist.data, mnist.target.astype(int)

    # MNIST: first 60k are training set, last 10k are test set (different writers)
    X_orig_train, y_orig_train = X_all[:60000], y_all[:60000]
    X_orig_test, y_orig_test = X_all[60000:], y_all[60000:]

    # Filter out digit 0 from each split separately
    train_mask = y_orig_train != 0
    X_orig_train = X_orig_train[train_mask]
    y_orig_train = y_orig_train[train_mask]

    test_mask = y_orig_test != 0
    X_orig_test = X_orig_test[test_mask]
    y_orig_test = y_orig_test[test_mask]

    # Normalize to [0, 1] and reshape to (N, 28, 28)
    X_orig_train = (X_orig_train.astype(np.float32) / 255.0).reshape(-1, 28, 28)
    y_orig_train = y_orig_train
    X_orig_test = (X_orig_test.astype(np.float32) / 255.0).reshape(-1, 28, 28)
    y_orig_test = y_orig_test

    print(f"Training pool (digits 1-9): {len(X_orig_train)}")
    print(f"Test set (digits 1-9): {len(X_orig_test)}")

    # Shuffle training pool with fixed seed, then split into train + val
    rng = np.random.default_rng(SEED)
    indices = rng.permutation(len(X_orig_train))
    X_orig_train = X_orig_train[indices]
    y_orig_train = y_orig_train[indices]

    X_train, y_train = X_orig_train[:50000], y_orig_train[:50000]
    X_val, y_val = X_orig_train[50000:], y_orig_train[50000:]

    # Test set stays untouched (different writers)
    X_test, y_test = X_orig_test, y_orig_test

    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    np.save(os.path.join(DATA_DIR, "X_train.npy"), X_train)
    np.save(os.path.join(DATA_DIR, "y_train.npy"), y_train)
    np.save(os.path.join(DATA_DIR, "X_val.npy"), X_val)
    np.save(os.path.join(DATA_DIR, "y_val.npy"), y_val)
    np.save(os.path.join(DATA_DIR, "X_test.npy"), X_test)
    np.save(os.path.join(DATA_DIR, "y_test.npy"), y_test)

    print(f"Saved to {DATA_DIR}")


if __name__ == "__main__":
    main()
