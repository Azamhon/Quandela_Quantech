"""
Random Forest baseline for swaption latent-code prediction.

Uses sklearn's MultiOutputRegressor wrapping RandomForestRegressor.
Input: flattened windowed latent context (120-dim).
Output: next latent code (20-dim).
"""

import time
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor


def train_random_forest(
    X_train, y_train, X_val, y_val, *,
    n_estimators=300, max_depth=20, min_samples_leaf=3,
    n_jobs=-1, random_state=42,
):
    """
    Train a multi-output Random Forest and return val predictions + metadata.

    Args:
        X_train : (n_train, 120) — flattened windowed features
        y_train : (n_train, 20)
        X_val   : (n_val, 120)
        y_val   : (n_val, 20)

    Returns:
        dict with keys: model, predictions, val_loss, train_time,
                        inference_time, n_parameters, description
    """
    print(f"    RF n_estimators  : {n_estimators}")
    print(f"    RF max_depth     : {max_depth}")

    base_rf = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        n_jobs=n_jobs,
        random_state=random_state,
    )
    model = MultiOutputRegressor(base_rf, n_jobs=1)

    # ── Train ──
    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    train_time = time.perf_counter() - t0

    # ── Predict ──
    predictions = model.predict(X_val)
    val_loss = float(np.mean((predictions - y_val) ** 2))

    # ── Approximate "parameter count" = total leaf nodes ──
    n_params = 0
    for est in model.estimators_:
        for tree in est.estimators_:
            n_params += tree.tree_.node_count
    n_params_str = f"~{n_params:,} nodes"

    # ── Inference timing ──
    sample = X_val[:1]
    t_inf = time.perf_counter()
    for _ in range(200):
        model.predict(sample)
    inference_time = (time.perf_counter() - t_inf) / 200

    return {
        "model": model,
        "predictions": predictions,
        "val_loss": val_loss,
        "train_time": train_time,
        "inference_time": inference_time,
        "n_parameters": n_params_str,
        "n_params_int": n_params,
    }
