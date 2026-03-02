"""
Evaluation metrics for benchmark comparison.

All metrics are computed on both:
    - Latent space  (20-dim predicted vs true latent codes)
    - Surface space (224-dim decoded prices via frozen AE decoder)
"""

import time
import numpy as np
import torch
from contextlib import contextmanager


# ─────────────────────────────────────────────────────────────
# Core metrics
# ─────────────────────────────────────────────────────────────

def mse(y_true, y_pred):
    """Mean Squared Error."""
    return float(np.mean((y_true - y_pred) ** 2))


def rmse(y_true, y_pred):
    """Root Mean Squared Error."""
    return float(np.sqrt(mse(y_true, y_pred)))


def mae(y_true, y_pred):
    """Mean Absolute Error."""
    return float(np.mean(np.abs(y_true - y_pred)))


def r_squared(y_true, y_pred):
    """Coefficient of determination (R²)."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true, axis=0)) ** 2)
    return float(1.0 - ss_res / (ss_tot + 1e-10))


def max_error(y_true, y_pred):
    """Maximum absolute error across all dimensions."""
    return float(np.max(np.abs(y_true - y_pred)))


def mape(y_true, y_pred, epsilon=1e-8):
    """Mean Absolute Percentage Error (clipped denominator)."""
    return float(np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + epsilon))) * 100)


# ─────────────────────────────────────────────────────────────
# Composite evaluation
# ─────────────────────────────────────────────────────────────

def compute_all_metrics(y_true, y_pred, prefix=""):
    """Compute full metric dictionary."""
    p = f"{prefix}_" if prefix else ""
    return {
        f"{p}MSE": mse(y_true, y_pred),
        f"{p}RMSE": rmse(y_true, y_pred),
        f"{p}MAE": mae(y_true, y_pred),
        f"{p}R²": r_squared(y_true, y_pred),
        f"{p}MaxErr": max_error(y_true, y_pred),
    }


def compute_latent_and_surface_metrics(y_true_latent, y_pred_latent, ae_decoder):
    """
    Evaluate at both latent and surface levels.

    Args:
        y_true_latent : np.ndarray (N, 20)
        y_pred_latent : np.ndarray (N, 20)
        ae_decoder    : callable — frozen AE decoder

    Returns:
        dict with latent_* and surface_* metrics
    """
    # Latent-level
    metrics = compute_all_metrics(y_true_latent, y_pred_latent, prefix="latent")

    # Surface-level (decode both through frozen AE)
    with torch.no_grad():
        true_surf = ae_decoder(
            torch.tensor(y_true_latent, dtype=torch.float32)
        ).numpy()
        pred_surf = ae_decoder(
            torch.tensor(y_pred_latent, dtype=torch.float32)
        ).numpy()
    surface_metrics = compute_all_metrics(true_surf, pred_surf, prefix="surface")
    metrics.update(surface_metrics)

    return metrics


# ─────────────────────────────────────────────────────────────
# Timing utility
# ─────────────────────────────────────────────────────────────

@contextmanager
def timer():
    """Context manager that records elapsed wall-clock time."""
    record = {"elapsed": 0.0}
    start = time.perf_counter()
    yield record
    record["elapsed"] = time.perf_counter() - start


def measure_inference_time(predict_fn, x_sample, n_runs=200):
    """
    Measure average inference time per sample.

    Args:
        predict_fn : callable(x) → predictions
        x_sample   : single-sample input (1, dim) or appropriate shape
        n_runs     : number of repetitions for averaging

    Returns:
        float — average seconds per forward pass
    """
    # Warm-up
    for _ in range(10):
        predict_fn(x_sample)

    start = time.perf_counter()
    for _ in range(n_runs):
        predict_fn(x_sample)
    elapsed = time.perf_counter() - start
    return elapsed / n_runs


# ─────────────────────────────────────────────────────────────
# Report formatting
# ─────────────────────────────────────────────────────────────

def format_results_table(all_results):
    """
    Format benchmark results as aligned console table.

    Args:
        all_results: list of dicts with keys:
            'name', 'latent_MSE', 'latent_RMSE', 'surface_MSE',
            'surface_RMSE', 'R²', 'n_params', 'train_time',
            'inference_ms'
    """
    header = (
        f"{'Model':<28} | {'Lat MSE':>10} | {'Lat RMSE':>10} | "
        f"{'Surf MSE':>10} | {'Surf RMSE':>10} | {'R2':>8} | "
        f"{'Params':>10} | {'Train(s)':>10} | {'Inf(ms)':>10}"
    )
    sep = "-" * len(header)

    lines = [sep, header, sep]
    for r in all_results:
        line = (
            f"{r['name']:<28} | "
            f"{r.get('latent_MSE', 0):>10.6f} | "
            f"{r.get('latent_RMSE', 0):>10.6f} | "
            f"{r.get('surface_MSE', 0):>10.6f} | "
            f"{r.get('surface_RMSE', 0):>10.6f} | "
            f"{r.get('R2', r.get('R²', 0)):>8.4f} | "
            f"{r.get('n_params', 'N/A'):>10} | "
            f"{r.get('train_time', 0):>10.2f} | "
            f"{r.get('inference_ms', 0):>10.4f}"
        )
        lines.append(line)
    lines.append(sep)
    return "\n".join(lines)
