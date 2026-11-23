"""FingerFlex (Lomtev, Kovalev and Timchenko, 2023), re-implemented: wavelet power into a convolutional U-Net.

Sources: the paper (arXiv:2211.01960 v2, CC BY 4.0) and the authors' notebooks ``prepare_data.ipynb`` and
``Lightning_BCI-autoencoder.ipynb`` (https://github.com/Irautak/FingerFlex). That repository has no licence, so no
code was copied; the values below are those of the notebooks, which produced the published numbers.

Two forms share this module:

* ``offline=True``, the published method (used only by ``scripts/offline_study.py``): each recording z-scored with
  its own statistics, zero-phase FIR band-pass and notch, centred Morlet wavelets, a network with 'same' padding and
  linear upsampling, and a centred Gaussian smoother on the output. Every output uses samples after it.
* ``offline=False``, approach 14 (live): the same pipeline made causal. Statistics come from the training rows,
  filters are forward-only IIR filters with the same pass and stop bands, each Morlet wavelet is delayed by half its
  length (one-sided support), convolutions are left-padded, pooling windows end at the current sample and
  upsampling holds the last completed coarse sample. The output at time t depends on ECoG up to t only (tested).
  The published 200 ms shift between ECoG and fingers is dropped: a causal decoder that kept it would ignore the most
  recent 200 ms of signal (the paper reports the shift is not critical between 0 and 200 ms).
"""
from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from scipy import interpolate, signal
from torch import nn

FS = 1000
DECIM = 10  # wavelet power is kept every 10 ms (100 Hz)
FREQS = np.logspace(np.log10(40), np.log10(300), 40)  # 40 Morlet wavelets, 40-300 Hz
N_CYCLES = 7.0  # MNE default
NOTCH = tuple(50.0 * k for k in range(1, 10))  # the notebooks notch 50 Hz harmonics (the data are 60 Hz US mains)
SHIFT_OFFLINE = 20  # 200 ms at 100 Hz: ECoG at t is paired with the fingers at t + 200 ms (published)


@dataclass(frozen=True)
class FingerFlexConfig:
    """Training settings from ``Lightning_BCI-autoencoder.ipynb``."""

    window: int = 256  # 2.56 s at 100 Hz; every start offset is a training window (stride 1)
    batch_size: int = 128
    lr: float = 8.42e-5
    weight_decay: float = 1e-6
    max_epochs: int = 20
    dropout: float = 0.1
    seed: int = 0
    patience: int = 5  # runs that choose the epoch on validation stop after this many epochs without improvement
    # (the notebooks train 20 epochs and keep the best checkpoint; stopping early only saves time)


# --------------------------------------------------------------------------- features
def morlet_wavelets(freqs: np.ndarray = FREQS, fs: int = FS, n_cycles: float = N_CYCLES) -> list[np.ndarray]:
    """Complex Morlet wavelets as in MNE (``mne.time_frequency.morlet``, zero_mean=False): +-5 SD of the envelope."""
    out = []
    for f in freqs:
        sigma_t = n_cycles / (2.0 * np.pi * f)
        t = np.arange(0.0, 5.0 * sigma_t, 1.0 / fs)
        t = np.r_[-t[::-1], t[1:]]
        w = np.exp(2j * np.pi * f * t) * np.exp(-(t ** 2) / (2.0 * sigma_t ** 2))
        out.append((w / (np.sqrt(0.5) * np.linalg.norm(w))).astype(np.complex64))
    return out


def wavelet_power(x: np.ndarray, offline: bool, decim: int = DECIM) -> np.ndarray:
    """Morlet power of ``x`` (n_samples, n_channels), kept every ``decim`` samples: (n_steps, n_channels, n_freqs).

    Offline: centred wavelets (each value uses samples on both sides). Causal: the same wavelet applied so that its
    support ends at the current sample, i.e. delayed by half its length.
    """
    n = x.shape[0]
    keep = np.arange(0, n, decim)
    xc = x.T.astype(np.float32)  # (channels, samples)
    out = np.empty((len(keep), x.shape[1], len(FREQS)), dtype=np.float32)
    for j, w in enumerate(morlet_wavelets()):
        full = signal.oaconvolve(xc, w[None, :], mode="full", axes=1)  # (channels, n + len(w) - 1)
        start = (len(w) - 1) // 2 if offline else 0  # centred ('same') or causal alignment
        out[:, :, j] = (np.abs(full[:, start:start + n][:, keep]) ** 2).T
    return out


def _fir(numtaps: int, cutoff, pass_zero) -> np.ndarray:
    return signal.firwin(numtaps, cutoff, pass_zero=pass_zero, window="hamming", fs=FS)


def _zero_phase(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Linear-phase FIR with its group delay removed (MNE's one-pass zero-phase filtering)."""
    return signal.oaconvolve(x, h[:, None], mode="same", axes=0)


def preprocess(ecog: np.ndarray, offline: bool, mean: np.ndarray | None = None,
               std: np.ndarray | None = None) -> np.ndarray:
    """z-score per channel, common median reference, band-pass 40-300 Hz, notch at the 50 Hz harmonics.

    Offline (as published): statistics of this recording itself, MNE-style zero-phase FIR filters (331-tap band-pass
    with -6 dB points at 35 and 337.5 Hz; 6601-tap multi-band notch). Causal: ``mean``/``std`` from training data,
    forward-only 4th-order Butterworth band-pass and IIR notches.
    """
    x = ecog.astype(np.float64)
    mu = x.mean(0) if offline else mean
    sd = x.std(0) if offline else std
    x = (x - mu) / sd
    x = x - np.median(x, axis=1, keepdims=True)
    if offline:
        x = _zero_phase(x, _fir(331, [35.0, 337.5], pass_zero=False))
        edges = [e for f in NOTCH for e in (f - f / 400, f + f / 400)]
        x = _zero_phase(x, _fir(6601, edges, pass_zero=True))
    else:
        x = signal.sosfilt(signal.butter(4, [40.0, 300.0], btype="bandpass", fs=FS, output="sos"), x, axis=0)
        for f in NOTCH:
            b, a = signal.iirnotch(f, Q=30.0, fs=FS)
            x = signal.lfilter(b, a, x, axis=0)
    return x


def recording_features(ecog: np.ndarray, offline: bool, stats_end: int | None = None) -> np.ndarray:
    """Wavelet power of one recording, (n_samples // 10, n_electrodes * 40), electrode-major as in the notebooks.

    Offline: the recording is normalised with its own statistics (the train and test recordings separately).
    Causal: pass the continuous recording (training followed by test) and ``stats_end``, the number of leading
    samples whose statistics normalise it (the training rows only)."""
    if offline:
        x = preprocess(ecog, offline=True)
    else:
        ref = ecog[:stats_end]
        x = preprocess(ecog, offline=False, mean=ref.mean(0), std=ref.std(0))
    p = wavelet_power(x, offline=offline)
    return p.reshape(len(p), -1)


class RobustScaler:
    """sklearn's RobustScaler(unit_variance=True, quantile_range=(10, 90)) per feature, without clipping."""

    def fit(self, x: np.ndarray) -> "RobustScaler":
        lo, med, hi = np.percentile(x, [10, 50, 90], axis=0)
        self.center_ = med
        self.scale_ = (hi - lo) / (2 * 1.2815515655446004) + 1e-12  # normal quantile range 10-90 = 2.563 SD
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        return ((x - self.center_) / self.scale_).astype(np.float32)


def glove_100hz(flex_1khz: np.ndarray) -> np.ndarray:
    """The 25 Hz glove samples (every 40th of the super-sampled 1 kHz series), cubic-interpolated to 100 Hz."""
    knots = flex_1khz[::40]
    t = np.arange(len(knots)) * 4
    grid = np.arange(len(flex_1khz) // DECIM)
    grid = np.minimum(grid, t[-1])
    return interpolate.interp1d(t, knots, kind="cubic", axis=0)(grid).astype(np.float32)


# --------------------------------------------------------------------------- network
class _Block(nn.Module):
    """Conv1d (no bias) -> LayerNorm over channels -> GELU -> dropout -> max-pool (``pool`` > 1)."""

    def __init__(self, c_in: int, c_out: int, k: int, pool: int, causal: bool, dropout: float):
        super().__init__()
        self.conv = nn.Conv1d(c_in, c_out, k, bias=False, padding=0 if causal else "same")
        self.norm = nn.LayerNorm(c_out)
        self.drop = nn.Dropout(dropout)
        self.k, self.pool, self.causal = k, pool, causal

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (batch, channels, time)
        if self.causal:
            x = F.pad(x, (self.k - 1, 0))
        x = self.drop(F.gelu(self.norm(self.conv(x).transpose(1, 2)).transpose(1, 2)))
        if self.pool > 1:
            # causal: pooling windows [2j, 2j+1] end at the sample the coarse value is assigned to (see _up)
            x = F.max_pool1d(x, self.pool, self.pool)
        return x


def _up(x: torch.Tensor, causal: bool) -> torch.Tensor:
    """Double the time resolution. Offline: linear interpolation (uses the next coarse sample). Causal: fine sample m
    takes coarse sample (m - 1) // 2, the last one whose pooling window had ended by then (0 before the first)."""
    if not causal:
        return F.interpolate(x, scale_factor=2, mode="linear", align_corners=False)
    held = x.repeat_interleave(2, dim=-1)
    return F.pad(held, (1, 0))[..., :-1]


class _Reduce(nn.Module):
    """The first block (electrodes x wavelets -> 32 channels, kernel 3) on time-major input.

    Mathematically the notebook's Conv1d(n_inputs, 32, 3, bias=False) with the same weights, computed as one matrix
    product per time step followed by a shift-and-add, which avoids transposing the large input (about 3x faster on
    a CPU)."""

    def __init__(self, n_inputs: int, c_out: int, causal: bool, dropout: float, k: int = 3):
        super().__init__()
        self.proj = nn.Linear(n_inputs, k * c_out, bias=False)  # tap j of every output channel
        self.norm = nn.LayerNorm(c_out)
        self.drop = nn.Dropout(dropout)
        self.k, self.c_out, self.causal = k, c_out, causal

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (batch, time, n_inputs) -> (batch, c_out, time)
        z = self.proj(x).unflatten(-1, (self.k, self.c_out))  # (batch, time, k, c_out)
        n = z.shape[1]
        left = self.k - 1 if self.causal else (self.k - 1) // 2  # taps reaching back (and, offline, ahead)
        zp = F.pad(z, (0, 0, 0, 0, left, self.k - 1 - left))
        out = sum(zp[:, j:j + n, j] for j in range(self.k))  # y[t] = sum_j W_j x[t + j - left]
        return self.drop(F.gelu(self.norm(out))).transpose(1, 2)


class FingerFlexNet(nn.Module):
    """The FingerFlex encoder-decoder (``AutoEncoder1D``): 660,293 parameters for 64 electrodes x 40 wavelets.

    channels (32, 32, 64, 64, 128, 128), kernels (7, 7, 5, 5, 5), five max-pool / upsample stages, U-Net skips.
    Input (batch, time, n_electrodes * 40), output (batch, 5, time); time must be a multiple of 32.
    """

    CHANNELS = (32, 32, 64, 64, 128, 128)
    KERNELS = (7, 7, 5, 5, 5)

    def __init__(self, n_inputs: int, n_outputs: int = 5, causal: bool = True, dropout: float = 0.1):
        super().__init__()
        c, k = self.CHANNELS, self.KERNELS
        self.causal = causal
        self.reduce = _Reduce(n_inputs, c[0], causal, dropout)
        self.down = nn.ModuleList(_Block(c[i], c[i + 1], k[i], 2, causal, dropout) for i in range(5))
        # up block i takes the previous output (plus its skip) and returns c[4 - i] channels
        ins = [c[5], 2 * c[4], 2 * c[3], 2 * c[2], 2 * c[1]]
        outs = [c[4], c[3], c[2], c[1], c[0]]
        kernels = [k[4], k[3], k[2], k[1], k[0]]
        self.up = nn.ModuleList(_Block(i, o, kk, 1, causal, dropout) for i, o, kk in zip(ins, outs, kernels))
        self.head = nn.Conv1d(2 * c[0], n_outputs, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.reduce(x)
        skips = [h]
        for block in self.down:
            h = block(h)
            skips.append(h)
        skips.pop()  # the deepest output is not a skip
        for block in self.up:
            h = torch.cat([_up(block(h), self.causal), skips.pop()], dim=1)
        return self.head(h)


# --------------------------------------------------------------------------- training
def _loss(pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
    """Half MSE plus half the mean cosine distance over time (per window and finger), as in the notebook."""
    cos = F.cosine_similarity(pred, true, dim=-1, eps=1e-8).mean()
    return 0.5 * F.mse_loss(pred, true) + 0.5 * (1.0 - cos)


@torch.no_grad()
def predict(model: FingerFlexNet, x: np.ndarray, chunk: int = 32) -> np.ndarray:
    """Run the network over a whole recording x (time, features) at once; returns (time, 5).

    The input is zero-padded at the end to a multiple of 32 samples (the notebooks truncate instead); with causal
    convolutions the padding cannot affect earlier outputs."""
    model.eval()
    n = len(x)
    pad = (-n) % chunk
    xt = torch.from_numpy(np.ascontiguousarray(np.pad(x, ((0, pad), (0, 0)))))[None]
    return model(xt)[0, :, :n].T.numpy()


def train(model: FingerFlexNet, x: np.ndarray, y: np.ndarray, rows: slice, cfg: FingerFlexConfig,
          epochs: int, on_epoch=None) -> list[dict]:
    """Train on every window of ``cfg.window`` samples inside ``rows`` (stride 1, shuffled each epoch).

    ``x`` (time, features) and ``y`` (time, 5) are aligned at 100 Hz. ``on_epoch(epoch, model)`` may return a dict
    of scores recorded for that epoch (the model is in eval mode during the call); ``"stop": True`` ends training."""
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    xt, yt = torch.from_numpy(np.ascontiguousarray(x)), torch.from_numpy(np.ascontiguousarray(y))
    starts = np.arange(rows.start, rows.stop - cfg.window)
    offsets = torch.arange(cfg.window)
    history = []
    for epoch in range(epochs):
        model.train()
        order = rng.permutation(starts)
        losses = []
        for b in range(0, len(order), cfg.batch_size):
            idx = torch.from_numpy(order[b:b + cfg.batch_size])[:, None] + offsets
            pred = model(xt[idx])
            loss = _loss(pred, yt[idx].transpose(1, 2))
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        record = {"epoch": epoch, "train_loss": float(np.mean(losses))}
        if on_epoch is not None:
            record.update(on_epoch(epoch, model))
        history.append(record)
        if record.pop("stop", False):
            break
    return history


def corr_mean(pred: np.ndarray, true: np.ndarray, fingers=range(5)) -> float:
    """The notebooks' metric: mean Pearson r over fingers (all five by default) at 100 Hz."""
    return float(np.mean([np.corrcoef(pred[:, f], true[:, f])[0, 1] for f in fingers]))


def snapshot(model: nn.Module) -> dict:
    return copy.deepcopy(model.state_dict())
