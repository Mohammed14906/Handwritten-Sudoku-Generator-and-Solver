"""Pure NumPy feedforward neural network for digit classification (1-9)."""

import pickle
import numpy as np


def _relu(z):
    return np.maximum(0, z)


def _relu_deriv(z):
    return (z > 0).astype(z.dtype)


def _softmax(z):
    shifted = z - z.max(axis=1, keepdims=True)
    exp_z = np.exp(shifted)
    return exp_z / exp_z.sum(axis=1, keepdims=True)


def _cross_entropy_loss(probs, y):
    n = len(y)
    log_probs = np.log(probs[np.arange(n), y] + 1e-12)
    return -log_probs.mean()


def _init_weights(layer_sizes, seed=42):
    rng = np.random.default_rng(seed)
    weights = []
    biases = []
    for i in range(len(layer_sizes) - 1):
        fan_in = layer_sizes[i]
        fan_out = layer_sizes[i + 1]
        std = np.sqrt(2.0 / fan_in)
        W = rng.normal(0, std, (fan_in, fan_out)).astype(np.float32)
        b = np.zeros(fan_out, dtype=np.float32)
        weights.append(W)
        biases.append(b)
    return weights, biases


def _forward(X, weights, biases):
    activations = [X]
    pre_activations = []
    a = X
    for i in range(len(weights) - 1):
        z = a @ weights[i] + biases[i]
        pre_activations.append(z)
        a = _relu(z)
        activations.append(a)
    z_out = a @ weights[-1] + biases[-1]
    pre_activations.append(z_out)
    probs = _softmax(z_out)
    activations.append(probs)
    return activations, pre_activations


def _backward(activations, pre_activations, weights, y):
    n = len(y)
    num_layers = len(weights)
    grad_w = [None] * num_layers
    grad_b = [None] * num_layers

    dz = activations[-1].copy()
    dz[np.arange(n), y] -= 1
    dz /= n

    for i in range(num_layers - 1, -1, -1):
        grad_w[i] = activations[i].T @ dz
        grad_b[i] = dz.sum(axis=0)
        if i > 0:
            da = dz @ weights[i].T
            dz = da * _relu_deriv(pre_activations[i - 1])

    return grad_w, grad_b


def train(X_train, y_train, X_val, y_val, epochs=20, lr=0.01,
          batch_size=128, seed=42, verbose=True):
    # Labels: convert from 1-9 to 0-8 for internal indexing
    y_tr = y_train - 1
    y_v = y_val - 1

    X_flat = X_train.reshape(len(X_train), -1)
    X_val_flat = X_val.reshape(len(X_val), -1)

    layer_sizes = [784, 256, 128, 9]
    weights, biases = _init_weights(layer_sizes, seed)

    rng = np.random.default_rng(seed)
    n = len(X_flat)

    for epoch in range(epochs):
        current_lr = lr * (0.95 ** epoch)
        indices = rng.permutation(n)
        for start in range(0, n, batch_size):
            batch_idx = indices[start:start + batch_size]
            X_b = X_flat[batch_idx]
            y_b = y_tr[batch_idx]

            activations, pre_activations = _forward(X_b, weights, biases)
            grad_w, grad_b = _backward(activations, pre_activations, weights, y_b)

            for i in range(len(weights)):
                weights[i] -= current_lr * grad_w[i]
                biases[i] -= current_lr * grad_b[i]

        if verbose:
            val_acts, _ = _forward(X_val_flat, weights, biases)
            val_loss = _cross_entropy_loss(val_acts[-1], y_v)
            val_preds = val_acts[-1].argmax(axis=1)
            val_acc = (val_preds == y_v).mean()
            print(f"Epoch {epoch+1}/{epochs} - val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f}")

    # Temperature scaling on validation set
    temperature = _calibrate_temperature(X_val_flat, y_v, weights, biases)

    model = {
        "weights": weights,
        "biases": biases,
        "temperature": temperature,
        "layer_sizes": layer_sizes,
    }
    return model


def _calibrate_temperature(X_val, y_val, weights, biases):
    activations, pre_activations = _forward(X_val, weights, biases)
    logits = pre_activations[-1]

    T = 1.0
    lr_t = 0.01
    eps = 1e-4
    for _ in range(100):
        loss_plus = _cross_entropy_loss(_softmax(logits / (T + eps)), y_val)
        loss_minus = _cross_entropy_loss(_softmax(logits / (T - eps)), y_val)
        grad = (loss_plus - loss_minus) / (2 * eps)
        T -= lr_t * grad
        T = max(T, 0.1)

    return float(T)


def predict_proba(model, X):
    X_flat = X.reshape(len(X), -1)
    activations, pre_activations = _forward(X_flat, model["weights"], model["biases"])
    logits = pre_activations[-1]
    return _softmax(logits / model["temperature"])


def predict(model, X):
    probs = predict_proba(model, X)
    return probs.argmax(axis=1) + 1  # back to 1-9


def save_model(model, path):
    with open(path, "wb") as f:
        pickle.dump(model, f)


def load_model(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def accuracy(model, X, y):
    preds = predict(model, X)
    return (preds == y).mean()


def confusion_matrix(model, X, y):
    preds = predict(model, X)
    n_classes = 9
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for true, pred in zip(y, preds):
        cm[true - 1, pred - 1] += 1
    return cm


def expected_calibration_error(model, X, y, n_bins=10):
    probs = predict_proba(model, X)
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1) + 1
    correct = (predictions == y).astype(float)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        if mask.sum() == 0:
            continue
        bin_acc = correct[mask].mean()
        bin_conf = confidences[mask].mean()
        ece += mask.sum() * abs(bin_acc - bin_conf)
    return ece / len(y)
