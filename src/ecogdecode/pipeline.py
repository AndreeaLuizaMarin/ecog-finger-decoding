"""Features over the continuous recording and the chronological split shared by every approach.

    |---------------- training segment (2/3) ----------------|----- test (1/3) -----|
    | warm-up | fit (first 80 %)              | validation    | official test set    |

* Settings and early stopping are chosen on the validation block only.
* With ``Protocol.refit`` the chosen model is then retrained on the whole training segment.
* The official test labels are read once, by ``score_test``.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .config import DEFAULT_FEATURES, DEFAULT_PROTOCOL, FeatureConfig, Protocol
from .data import SubjectData
from .evaluation import score_test
from .features import Standardiser, extract_features, mask_artefact_channels


@dataclass
class Prepared:
    subject: int
    cfg: FeatureConfig
    x: np.ndarray  # (n_steps, n_features) raw causal features of the whole recording
    y: np.ndarray  # (n_steps, 5) finger flexion at each step (test rows only ever used for scoring)
    ends: np.ndarray  # sample index of each step
    n_train_samples: int
    fit_rows: slice
    val_rows: slice
    train_rows: slice  # fit + validation
    test_rows: slice

    @property
    def test_times(self) -> np.ndarray:
        """Sample times (relative to the test start) of the last training step and every test step."""
        return self.ends[self.test_rows.start - 1: self.test_rows.stop] - self.n_train_samples

    def test_predictions(self, pred: np.ndarray) -> np.ndarray:
        """Step predictions from the last training step to the end, aligned with ``test_times``."""
        return pred[self.test_rows.start - 1: self.test_rows.stop]


def prepare(data: SubjectData, cfg: FeatureConfig = DEFAULT_FEATURES, protocol: Protocol = DEFAULT_PROTOCOL) -> Prepared:
    """Compute causal features over the continuous recording and define the chronological split."""
    x, ends = extract_features(data.ecog, cfg)
    test_flex = data.test_flex if data.test_flex is not None else np.full((len(data.test_ecog), 5), np.nan)
    y = np.concatenate([data.train_flex, test_flex])[ends].astype(np.float32)
    n_train_steps = int(np.searchsorted(ends, data.n_train))  # steps that end inside the training segment
    n_val = int(round(protocol.val_fraction * (n_train_steps - protocol.warmup_steps)))
    split = n_train_steps - n_val
    return Prepared(data.subject, cfg, x, y, ends, data.n_train, slice(protocol.warmup_steps, split),
                    slice(split, n_train_steps), slice(protocol.warmup_steps, n_train_steps),
                    slice(n_train_steps, len(ends)))


@dataclass
class Scaled:
    x: np.ndarray  # standardised, artefact-masked features
    y: np.ndarray  # standardised targets (test rows are meaningless placeholders)
    x_scaler: Standardiser
    y_scaler: Standardiser


def adaptive_normalise(x_std: np.ndarray, start: int, steps: float) -> np.ndarray:
    """Causal re-normalisation of standardised features with exponentially weighted running mean and
    variance (time constant ``steps``), starting at row ``start`` from the training statistics (0, 1).
    Each row uses statistics from the rows before it only, so a slow drift of the recording is tracked
    online without looking ahead."""
    a = 1.0 / steps
    out = x_std.copy()
    mean, var = np.zeros(x_std.shape[1]), np.ones(x_std.shape[1])
    for t in range(start, len(x_std)):
        out[t] = (x_std[t] - mean) / np.sqrt(var + 1e-6)
        delta = x_std[t] - mean
        mean = mean + a * delta
        var = (1 - a) * (var + a * delta * delta)
    return out.astype(np.float32)


def standardise(prep: Prepared, rows: slice, adaptive_steps: float = 0.0) -> Scaled:
    """Z-score features and targets with statistics from ``rows`` only, optionally keep re-normalising the
    features online after the end of ``rows`` (``adaptive_steps`` > 0), then mask artefactual channels."""
    xs = Standardiser().fit(prep.x[rows])
    ys = Standardiser().fit(prep.y[rows])
    x = xs.transform(prep.x)
    if adaptive_steps > 0:
        x = adaptive_normalise(x, rows.stop, adaptive_steps)
    x, _ = mask_artefact_channels(x, prep.cfg.n_per_channel, prep.cfg.artefact_z)
    return Scaled(x, ys.transform(np.nan_to_num(prep.y)), xs, ys)


def evaluate(prep: Prepared, pred_steps: np.ndarray, test_flex: np.ndarray) -> dict:
    """Official test score of a full-sequence prediction (n_steps, 5)."""
    return score_test(prep.test_predictions(pred_steps), prep.test_times, test_flex)


def top_electrodes(prep: Prepared, rows: slice, k: int) -> np.ndarray:
    """The ``k`` electrodes whose features correlate most with finger flexion in ``rows`` (training data):
    for each electrode, the largest absolute Pearson r between any of its features and any finger."""
    npc = prep.cfg.n_per_channel
    x, y = prep.x[rows], np.nan_to_num(prep.y[rows])
    xz = (x - x.mean(0)) / (x.std(0) + 1e-9)
    yz = (y - y.mean(0)) / (y.std(0) + 1e-9)
    r = np.abs(xz.T @ yz) / len(x)  # (features, fingers)
    per_electrode = r.max(axis=1).reshape(-1, npc).max(axis=1)
    return np.sort(np.argsort(per_electrode)[::-1][:k])


def keep_electrodes(prep: Prepared, electrodes: np.ndarray) -> Prepared:
    """A copy of ``prep`` whose features come from the given electrodes only."""
    npc = prep.cfg.n_per_channel
    cols = (np.asarray(electrodes)[:, None] * npc + np.arange(npc)).ravel()
    return replace(prep, x=prep.x[:, cols])
