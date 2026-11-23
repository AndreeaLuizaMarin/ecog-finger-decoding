"""Causal ECoG feature extraction.

Every operation uses only present and past samples (forward-only IIR filters and trailing
windows), so the feature vector at time t never depends on data after t. This is what an
online decoder can compute, and it is checked in ``tests/``.

Pipeline: median re-referencing -> line-noise notches -> per band: band-pass, square, trailing
mean -> log. The local motor potential (LMP, trailing mean of the re-referenced signal) is
added as an extra feature. LMP plus band power follows earlier finger-decoding work on these
recordings (Kubanek et al., 2009; Liang and Bougrain, 2012).
"""
from __future__ import annotations

import numpy as np
from scipy import signal

from .config import DEFAULT_FEATURES, FeatureConfig


def step_ends(n_samples: int, hop: int) -> np.ndarray:
    """Index of the last sample covered by each feature vector."""
    return np.arange(hop - 1, n_samples, hop)


def trailing_mean(x: np.ndarray, ends: np.ndarray, window: int) -> np.ndarray:
    """Mean of ``x[e - window + 1 : e + 1]`` for each end index ``e`` (shorter at the very start)."""
    csum = np.concatenate([np.zeros((1, x.shape[1])), np.cumsum(x, axis=0, dtype=np.float64)])
    starts = np.maximum(ends + 1 - window, 0)
    return (csum[ends + 1] - csum[starts]) / (ends + 1 - starts)[:, None]


def rereference(x: np.ndarray, method: str = "median") -> np.ndarray:
    """Subtract the instantaneous median (or mean) across channels; uses the current sample only.

    The median keeps one artefactual electrode from contaminating all the others, which the
    common average would do.
    """
    if method == "median":
        return x - np.median(x, axis=1, keepdims=True)
    if method == "average":
        return x - x.mean(axis=1, keepdims=True)
    raise ValueError(f"unknown reference {method!r}")


def notch_filters(cfg: FeatureConfig) -> list[tuple[np.ndarray, np.ndarray]]:
    return [signal.iirnotch(cfg.line_freq * k, Q=30.0, fs=cfg.fs) for k in cfg.notch_harmonics
            if cfg.line_freq * k < cfg.fs / 2]


def band_filters(cfg: FeatureConfig) -> list[np.ndarray]:
    return [signal.butter(cfg.filter_order, band, btype="bandpass", fs=cfg.fs, output="sos") for band in cfg.bands]


def extract_features(ecog: np.ndarray, cfg: FeatureConfig = DEFAULT_FEATURES) -> tuple[np.ndarray, np.ndarray]:
    """Features of a continuous recording: causal by default, zero-phase and centred if ``cfg.offline``.

    Returns ``features`` (n_steps, n_channels * cfg.n_per_channel), float32, ordered channel-major
    (all features of channel 0, then channel 1, ...), and ``ends``, the last sample used per step.
    """
    x = rereference(ecog.astype(np.float64), cfg.reference)
    for b, a in notch_filters(cfg):
        x = signal.filtfilt(b, a, x, axis=0) if cfg.offline else signal.lfilter(b, a, x, axis=0)
    ends = step_ends(x.shape[0], cfg.hop)
    window = int(round(cfg.window_ms * cfg.fs / 1000))
    # offline: the window is centred on the step instead of ending at it
    win_ends = np.minimum(ends + window // 2, x.shape[0] - 1) if cfg.offline else ends
    feats = [trailing_mean(x, win_ends, window)] if cfg.include_lmp else []
    for sos in band_filters(cfg):
        y = signal.sosfiltfilt(sos, x, axis=0) if cfg.offline else signal.sosfilt(sos, x, axis=0)
        feats.append(np.log(trailing_mean(y * y, win_ends, window) + 1e-12))
    out = np.stack(feats, axis=-1)  # (n_steps, n_channels, n_per_channel)
    return out.reshape(out.shape[0], -1).astype(np.float32), ends


class Standardiser:
    """Z-scoring with statistics from the fitting data only."""

    def fit(self, x: np.ndarray) -> "Standardiser":
        self.mean_ = x.mean(axis=0)
        self.std_ = x.std(axis=0) + 1e-8
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        return ((x - self.mean_) / self.std_).astype(np.float32)

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        return x * self.std_ + self.mean_


def mask_artefact_channels(x_std: np.ndarray, n_per_channel: int, z: float = 6.0) -> tuple[np.ndarray, np.ndarray]:
    """Online bad-channel handling on standardised features.

    At each step, a channel with any feature beyond ``z`` training standard deviations is treated
    as artefactual and all its features are set to the training mean (zero). Only the current
    step is used, so this runs online. Returns the masked features and the boolean mask.
    """
    x = x_std.reshape(x_std.shape[:-1] + (-1, n_per_channel)).copy()
    bad = (np.abs(x) > z).any(axis=-1)
    x[bad] = 0.0
    return x.reshape(x_std.shape), bad
