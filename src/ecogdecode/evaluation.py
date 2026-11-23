"""Scoring exactly as in the competition: Pearson r per finger at 1 kHz, ring finger excluded."""

from __future__ import annotations

import numpy as np

from .data import SCORED_FINGERS


def pearson_r(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    """Column-wise Pearson correlation of two (n, k) arrays (or two 1-D arrays). Constant input -> 0."""
    p = np.atleast_2d(pred.T).T.astype(np.float64)
    t = np.atleast_2d(true.T).T.astype(np.float64)
    p = p - p.mean(axis=0)
    t = t - t.mean(axis=0)
    denom = np.sqrt((p * p).sum(axis=0) * (t * t).sum(axis=0))
    return np.nan_to_num((p * t).sum(axis=0) / np.where(denom > 0, denom, np.nan))


def scored_mean(r: np.ndarray) -> float:
    """Mean r over the scored fingers (thumb, index, middle, little). Works on (..., 5) arrays."""
    return float(np.mean(np.asarray(r)[..., list(SCORED_FINGERS)]))


def score(pred: np.ndarray, true: np.ndarray) -> float:
    """The selection criterion used on validation data: mean r over the scored fingers."""
    return scored_mean(pearson_r(pred, true))


def to_sample_rate(step_values: np.ndarray, step_times: np.ndarray, n_samples: int) -> np.ndarray:
    """Hold each step prediction until the next one arrives (causal zero-order hold).

    ``step_times`` are the sample indices at which predictions become available; the output at
    sample t is the most recent prediction available at t.
    """
    idx = np.searchsorted(step_times, np.arange(n_samples), side="right") - 1
    return step_values[np.clip(idx, 0, None)]


def score_test(pred_steps: np.ndarray, step_times: np.ndarray, test_flex: np.ndarray) -> dict:
    """Official test score of step predictions. ``pred_steps`` should start with the prediction of
    the last training step (time < 0) so that the first test samples are covered causally."""
    r = pearson_r(to_sample_rate(pred_steps, step_times, len(test_flex)), test_flex)
    return {"r": r.tolist(), "score": scored_mean(r), "scored_fingers": list(SCORED_FINGERS)}


def causal_ema(x: np.ndarray, tau_ms: float, step_ms: float = 40.0) -> np.ndarray:
    """Exponential moving average over steps (axis 0): uses only the present and the past."""
    if tau_ms <= 0:
        return x
    a = 1.0 - np.exp(-step_ms / tau_ms)
    out = np.empty_like(x, dtype=np.float64)
    out[0] = x[0]
    for t in range(1, len(x)):
        out[t] = out[t - 1] + a * (x[t] - out[t - 1])
    return out


def choose_smoothing(
    val_pred: np.ndarray, val_true: np.ndarray, grid: tuple[float, ...], context: int = 0
) -> tuple[float, float]:
    """Pick the smoothing time constant with the best validation score.

    ``val_pred`` may start with ``context`` warm-up steps that are smoothed but not scored.
    Returns (tau_ms, validation score at that tau).
    """
    scores = {tau: score(causal_ema(val_pred, tau)[context:], val_true) for tau in grid}
    best = max(scores, key=scores.get)
    return best, scores[best]
