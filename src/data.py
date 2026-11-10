import os
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "processed")


def load_splits():
    X_train = np.load(os.path.join(DATA_DIR, "X_train.npy"))
    y_train = np.load(os.path.join(DATA_DIR, "y_train.npy"))
    X_val = np.load(os.path.join(DATA_DIR, "X_val.npy"))
    y_val = np.load(os.path.join(DATA_DIR, "y_val.npy"))
    X_test = np.load(os.path.join(DATA_DIR, "X_test.npy"))
    y_test = np.load(os.path.join(DATA_DIR, "y_test.npy"))
    return X_train, y_train, X_val, y_val, X_test, y_test


def get_digit_images(digit, split="train"):
    splits = load_splits()
    split_map = {"train": (0, 1), "val": (2, 3), "test": (4, 5)}
    X, y = splits[split_map[split][0]], splits[split_map[split][1]]
    return X[y == digit]


def show_samples(n=5, seed=42):
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(seed)
    X_train, y_train, *_ = load_splits()

    fig, axes = plt.subplots(1, n, figsize=(2 * n, 2))
    indices = rng.choice(len(X_train), size=n, replace=False)
    for ax, idx in zip(axes, indices):
        ax.imshow(X_train[idx], cmap="gray")
        ax.set_title(f"Label: {y_train[idx]}")
        ax.axis("off")
    plt.tight_layout()
    plt.show()
