"""Wiener filter: ridge regression on the current and recent feature vectors."""
from __future__ import annotations

import numpy as np


def lag_matrix(x: np.ndarray, n_lags: int) -> np.ndarray:
    """Stack ``x[t], x[t-1], ..., x[t-n_lags+1]`` into one row per step (zeros before the start)."""
    n, f = x.shape
    out = np.zeros((n, f * n_lags), dtype=x.dtype)
    for lag in range(n_lags):
        out[lag:, lag * f:(lag + 1) * f] = x[: n - lag]
    return out


class LaggedRidge:
    """Causal linear decoder (a Wiener filter) fitted by ridge regression.

    ``fit`` and ``predict`` take the whole continuous feature sequence, so that the lags at
    the start of a segment are filled with the true preceding data rather than zeros.
    """

    def __init__(self, n_lags: int = 10, alpha: float = 1e3):
        self.n_lags = n_lags
        self.alpha = alpha

    def _design(self, x_seq: np.ndarray) -> np.ndarray:
        return lag_matrix(x_seq, self.n_lags)

    def fit(self, x_seq: np.ndarray, y_seq: np.ndarray, rows: slice) -> "LaggedRidge":
        a = self._design(x_seq.astype(np.float32))[rows]
        y = y_seq[rows].astype(np.float64)
        self.x_mean_, self.y_mean_ = a.mean(axis=0, dtype=np.float64), y.mean(axis=0)
        a -= self.x_mean_.astype(np.float32)
        gram = (a.T @ a).astype(np.float64)
        gram[np.diag_indices_from(gram)] += self.alpha
        self.coef_ = np.linalg.solve(gram, (a.T @ (y - self.y_mean_).astype(np.float32)).astype(np.float64))
        return self

    def predict(self, x_seq: np.ndarray) -> np.ndarray:
        return (self._design(x_seq) - self.x_mean_) @ self.coef_ + self.y_mean_


def select_ridge(x_seq: np.ndarray, y_seq: np.ndarray, fit_rows: slice, val_rows: slice,
                 lag_grid: tuple[int, ...], alpha_grid: np.ndarray, score_fn) -> tuple[dict, list[dict]]:
    """Grid search over lags and ridge penalty on a held-out validation block.

    Returns the best setting, every result and the best setting's validation predictions.

    For each lag setting the Gram matrix is eigendecomposed once, which makes the whole
    regularisation path cheap.
    """
    results, best_pred = [], None
    for n_lags in lag_grid:
        design = lag_matrix(x_seq.astype(np.float32), n_lags)  # float32 keeps memory in check
        a, y = design[fit_rows], y_seq[fit_rows].astype(np.float64)
        a_val = design[val_rows]
        del design
        x_mean, y_mean = a.mean(axis=0, dtype=np.float64), y.mean(axis=0)
        a -= x_mean.astype(np.float32)
        evals, evecs = np.linalg.eigh((a.T @ a).astype(np.float64))
        proj = evecs.T @ (a.T @ (y - y_mean).astype(np.float32)).astype(np.float64)
        del a
        a_val = a_val - x_mean
        for alpha in alpha_grid:
            coef = evecs @ (proj / (evals + alpha)[:, None])
            pred = a_val @ coef + y_mean
            val = float(score_fn(pred, y_seq[val_rows]))
            results.append({"n_lags": int(n_lags), "alpha": float(alpha), "val_score": val})
            if best_pred is None or val > max(r["val_score"] for r in results[:-1]):
                best_pred = pred
    best = max(results, key=lambda r: r["val_score"])
    return best, results, best_pred
