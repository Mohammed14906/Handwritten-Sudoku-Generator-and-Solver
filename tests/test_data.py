import os
import hashlib
import numpy as np
import pytest

from src.data import load_splits, DATA_DIR


@pytest.fixture(scope="module")
def splits():
    return load_splits()


def _files_exist():
    return os.path.exists(os.path.join(DATA_DIR, "X_train.npy"))


pytestmark = pytest.mark.skipif(
    not _files_exist(), reason="Run make_data.py first"
)


def test_no_zeros(splits):
    _, y_train, _, y_val, _, y_test = splits
    for y in [y_train, y_val, y_test]:
        assert 0 not in y, "Digit 0 should be filtered out"


def test_shapes(splits):
    X_train, y_train, X_val, y_val, X_test, y_test = splits
    for X in [X_train, X_val, X_test]:
        assert X.ndim == 3
        assert X.shape[1] == 28 and X.shape[2] == 28
        assert X.dtype == np.float32
    for y in [y_train, y_val, y_test]:
        assert y.ndim == 1


def test_normalization(splits):
    X_train, _, X_val, _, X_test, _ = splits
    for X in [X_train, X_val, X_test]:
        assert X.min() >= 0.0
        assert X.max() <= 1.0


def test_label_range(splits):
    _, y_train, _, y_val, _, y_test = splits
    for y in [y_train, y_val, y_test]:
        assert set(np.unique(y)).issubset(set(range(1, 10)))


def test_splits_disjoint(splits):
    X_train, _, X_val, _, X_test, _ = splits

    def _hash_images(X):
        return {hashlib.md5(img.tobytes()).hexdigest() for img in X}

    h_train = _hash_images(X_train[:5000])
    h_val = _hash_images(X_val[:5000])
    h_test = _hash_images(X_test[:5000])

    assert len(h_train & h_val) == 0, "Train and val overlap"
    assert len(h_train & h_test) == 0, "Train and test overlap"
    assert len(h_val & h_test) == 0, "Val and test overlap"
