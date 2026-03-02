"""
Classical ML baselines for swaption latent-code prediction.

Models:
    - Ridge Regression          (linear baseline)
    - Gradient Boosting         (ensemble of shallow trees)
    - Support Vector Regression (kernel method)
    - sklearn MLP               (neural network via sklearn)

All models map:  flattened context (120-dim) → next latent code (20-dim)
using sklearn's MultiOutputRegressor when native multi-output is unsupported.
"""

import time
import numpy as np

from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.multioutput import MultiOutputRegressor


# ─────────────────────────────────────────────────────────────
# Generic training wrapper
# ─────────────────────────────────────────────────────────────

def _train_sklearn_model(name, model, X_train, y_train, X_val, y_val):
    """
    Fit an sklearn estimator and return standardised results dict.

    If the estimator doesn't support multi-output natively, it is
    wrapped in MultiOutputRegressor.
    """
    # Wrap if needed
    try:
        model.fit(X_train[:5], y_train[:5])       # quick probe
    except ValueError:
        model = MultiOutputRegressor(model, n_jobs=-1)

    # ── Train ──
    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    train_time = time.perf_counter() - t0

    # ── Predict ──
    predictions = model.predict(X_val)
    val_loss = float(np.mean((predictions - y_val) ** 2))

    # ── Parameter count heuristic ──
    try:
        if hasattr(model, "coef_"):
            n_params = int(np.prod(model.coef_.shape))
        elif hasattr(model, "estimators_"):
            n_params = sum(
                getattr(e, "tree_", None).node_count
                if hasattr(e, "tree_") else 0
                for est_list in (model.estimators_ if isinstance(model.estimators_[0], list)
                                 else [model.estimators_])
                for e in (est_list if isinstance(est_list, list) else [est_list])
            )
        else:
            n_params = "N/A"
    except Exception:
        n_params = "N/A"

    # ── Inference timing ──
    sample = X_val[:1]
    t_inf = time.perf_counter()
    for _ in range(200):
        model.predict(sample)
    inference_time = (time.perf_counter() - t_inf) / 200

    print(f"    {name:<22} val_MSE={val_loss:.6f}  train={train_time:.1f}s")

    return {
        "model": model,
        "predictions": predictions,
        "val_loss": val_loss,
        "train_time": train_time,
        "inference_time": inference_time,
        "n_parameters": n_params,
    }


# ─────────────────────────────────────────────────────────────
# Individual model factories
# ─────────────────────────────────────────────────────────────

def train_ridge(X_train, y_train, X_val, y_val, *, alpha=1.0):
    """Ridge Regression (L2-regularised linear model)."""
    model = Ridge(alpha=alpha)
    return _train_sklearn_model("Ridge", model, X_train, y_train, X_val, y_val)


def train_gradient_boosting(X_train, y_train, X_val, y_val, *,
                             n_estimators=200, max_depth=5, lr=0.05):
    """Gradient Boosting Regressor (per-output wrapper)."""
    base = GradientBoostingRegressor(
        n_estimators=n_estimators, max_depth=max_depth,
        learning_rate=lr, random_state=42,
    )
    model = MultiOutputRegressor(base, n_jobs=-1)
    return _train_sklearn_model("Gradient Boosting", model, X_train, y_train, X_val, y_val)


def train_svr(X_train, y_train, X_val, y_val, *, C=1.0, kernel="rbf"):
    """Support Vector Regression (per-output wrapper)."""
    base = SVR(C=C, kernel=kernel, max_iter=5000)
    model = MultiOutputRegressor(base, n_jobs=-1)
    return _train_sklearn_model("SVR (RBF)", model, X_train, y_train, X_val, y_val)


def train_sklearn_mlp(X_train, y_train, X_val, y_val, *,
                       hidden=(128, 64), lr=1e-3, max_iter=500):
    """sklearn MLP Regressor (simple neural network)."""
    model = MLPRegressor(
        hidden_layer_sizes=hidden,
        learning_rate_init=lr,
        max_iter=max_iter,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=42,
    )
    return _train_sklearn_model("sklearn MLP", model, X_train, y_train, X_val, y_val)


# ─────────────────────────────────────────────────────────────
# Run all classical ML benchmarks at once
# ─────────────────────────────────────────────────────────────

def train_all_classical_ml(X_train, y_train, X_val, y_val):
    """
    Train all 4 classical ML baselines.

    Returns:
        dict mapping model name → results dict
    """
    results = {}

    print("  ── Ridge Regression ──")
    results["Ridge Regression"] = train_ridge(X_train, y_train, X_val, y_val)

    print("  ── Gradient Boosting ──")
    results["Gradient Boosting"] = train_gradient_boosting(
        X_train, y_train, X_val, y_val
    )

    print("  ── SVR (RBF kernel) ──")
    results["SVR (RBF)"] = train_svr(X_train, y_train, X_val, y_val)

    print("  ── sklearn MLP ──")
    results["sklearn MLP"] = train_sklearn_mlp(X_train, y_train, X_val, y_val)

    return results
