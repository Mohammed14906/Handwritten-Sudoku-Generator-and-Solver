import os
import tempfile
import numpy as np
import pytest

from src.data import load_splits, DATA_DIR
from src import nn


def _files_exist():
    return os.path.exists(os.path.join(DATA_DIR, "X_train.npy"))


pytestmark = pytest.mark.skipif(
    not _files_exist(), reason="Run make_data.py first"
)


@pytest.fixture(scope="module")
def trained_model():
    X_train, y_train, X_val, y_val, _, _ = load_splits()
    model = nn.train(X_train, y_train, X_val, y_val,
                     epochs=30, lr=0.05, batch_size=128, seed=42, verbose=False)
    return model


@pytest.fixture(scope="module")
def test_data():
    _, _, _, _, X_test, y_test = load_splits()
    return X_test, y_test


def test_output_shape(trained_model, test_data):
    X_test, _ = test_data
    probs = nn.predict_proba(trained_model, X_test[:100])
    assert probs.shape == (100, 9)


def test_probabilities_sum_to_one(trained_model, test_data):
    X_test, _ = test_data
    probs = nn.predict_proba(trained_model, X_test[:100])
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)


def test_prediction_matches_argmax(trained_model, test_data):
    X_test, _ = test_data
    X_sub = X_test[:100]
    probs = nn.predict_proba(trained_model, X_sub)
    preds = nn.predict(trained_model, X_sub)
    expected = probs.argmax(axis=1) + 1
    assert np.array_equal(preds, expected)


def test_save_load_roundtrip(trained_model, test_data):
    X_test, _ = test_data
    X_sub = X_test[:50]

    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
        path = f.name

    try:
        nn.save_model(trained_model, path)
        loaded = nn.load_model(path)
        preds_orig = nn.predict(trained_model, X_sub)
        preds_loaded = nn.predict(loaded, X_sub)
        assert np.array_equal(preds_orig, preds_loaded)
    finally:
        os.unlink(path)


def test_accuracy_above_threshold(trained_model, test_data):
    X_test, y_test = test_data
    acc = nn.accuracy(trained_model, X_test, y_test)
    assert acc > 0.95, f"Test accuracy {acc:.4f} below 0.95"
