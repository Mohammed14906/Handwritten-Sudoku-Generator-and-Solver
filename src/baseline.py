"""Logistic regression baseline with same interface as nn.py."""

import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression


def train(X_train, y_train, X_val=None, y_val=None, seed=42, verbose=True):
    X_flat = X_train.reshape(len(X_train), -1)
    clf = LogisticRegression(
        max_iter=1000, solver="lbfgs", multi_class="multinomial",
        random_state=seed, n_jobs=-1
    )
    clf.fit(X_flat, y_train)

    if verbose and X_val is not None:
        val_acc = clf.score(X_val.reshape(len(X_val), -1), y_val)
        print(f"Logistic Regression val_acc: {val_acc:.4f}")

    model = {"clf": clf}
    return model


def predict_proba(model, X):
    X_flat = X.reshape(len(X), -1)
    return model["clf"].predict_proba(X_flat).astype(np.float32)


def predict(model, X):
    X_flat = X.reshape(len(X), -1)
    return model["clf"].predict(X_flat)


def save_model(model, path):
    with open(path, "wb") as f:
        pickle.dump(model, f)


def load_model(path):
    with open(path, "rb") as f:
        return pickle.load(f)
