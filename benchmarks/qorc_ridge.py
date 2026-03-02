"""
Benchmark wrapper for QORC + Ridge Regression model.

Uses the same pre-computed quantum features as the QUANTECH MLP model,
but replaces the MLP head with sklearn Ridge regression.
This is the textbook reservoir computing readout.
"""

import os
import sys
import time
import numpy as np

from sklearn.linear_model import Ridge

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


def train_qorc_ridge(X_train, y_train, X_val, y_val,
                     Q_train, Q_val, alpha=1.0):
    """
    Train Ridge regression on QORC features + classical context.

    Args:
        X_train, y_train: classical context and targets (train split)
        X_val, y_val: classical context and targets (val split)
        Q_train, Q_val: pre-computed quantum features
        alpha: Ridge regularisation

    Returns:
        dict with predictions, val_loss, train_time, inference_time, n_parameters
    """
    # Concatenate quantum + classical features
    X_tr_full = np.hstack([Q_train, X_train])
    X_vl_full = np.hstack([Q_val, X_val])

    print(f"    Input dim: {X_tr_full.shape[1]} "
          f"(quantum={Q_train.shape[1]} + classical={X_train.shape[1]})")
    print(f"    Ridge alpha: {alpha}")

    # Search for best alpha
    alphas = [0.01, 0.1, 1.0, 10.0, 100.0]
    best_model = None
    best_mse = float("inf")
    best_alpha = alpha

    t0 = time.perf_counter()
    for a in alphas:
        model = Ridge(alpha=a)
        model.fit(X_tr_full, y_train)
        pred = model.predict(X_vl_full)
        mse = float(np.mean((pred - y_val) ** 2))
        if mse < best_mse:
            best_mse = mse
            best_model = model
            best_alpha = a
    train_time = time.perf_counter() - t0

    print(f"    Best alpha: {best_alpha} (val MSE={best_mse:.6f})")

    predictions = best_model.predict(X_vl_full)
    val_loss = best_mse

    # Parameter count
    n_params = int(np.prod(best_model.coef_.shape))
    if best_model.intercept_ is not None:
        n_params += int(np.prod(np.array(best_model.intercept_).shape))

    # Inference timing
    sample = X_vl_full[:1]
    t_inf = time.perf_counter()
    for _ in range(200):
        best_model.predict(sample)
    inference_time = (time.perf_counter() - t_inf) / 200

    return {
        "model": best_model,
        "predictions": predictions,
        "val_loss": val_loss,
        "train_time": train_time,
        "inference_time": inference_time,
        "n_parameters": n_params,
        "alpha": best_alpha,
    }
