"""Online (chunk-by-chunk) version of the pipeline, as it would run next to an implant.

``StreamingFeatureExtractor`` consumes raw ECoG in 40-sample chunks and carries filter states
and trailing windows between calls; it reproduces ``features.extract_features`` exactly
(tested). ``replay_online`` exports a trained decoder's single-step form to ONNX, streams a
whole recording through features + decoder, checks the result against the offline
prediction and measures the per-step latency.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
from scipy import signal

from .config import DEFAULT_FEATURES, FeatureConfig
from .export import export_onnx, latency_summary, onnx_session
from .features import Standardiser, band_filters, mask_artefact_channels, notch_filters, rereference
from .models import LSTMDecoder, LSTMStep, S4DDecoder, S4DStep, StateGatedDecoder, StateGatedStep


class StreamingFeatureExtractor:
    def __init__(self, n_channels: int, cfg: FeatureConfig = DEFAULT_FEATURES):
        self.cfg = cfg
        self.window = int(round(cfg.window_ms * cfg.fs / 1000))
        self.notches = [[b, a, np.zeros((max(len(a), len(b)) - 1, n_channels))] for b, a in notch_filters(cfg)]
        self.bands = [[sos, np.zeros((sos.shape[0], 2, n_channels))] for sos in band_filters(cfg)]
        self.buf_x = np.zeros((self.window, n_channels))
        self.buf_p = np.zeros((len(self.bands), self.window, n_channels))
        self.n_seen = 0

    def process(self, chunk: np.ndarray) -> np.ndarray:
        """Take ``hop`` new samples (hop, n_channels); return one feature vector."""
        x = rereference(chunk.astype(np.float64), self.cfg.reference)
        for notch in self.notches:
            x, notch[2] = signal.lfilter(notch[0], notch[1], x, axis=0, zi=notch[2])
        n = len(x)
        self.buf_x = np.concatenate([self.buf_x[n:], x])
        for i, band in enumerate(self.bands):
            y, band[1] = signal.sosfilt(band[0], x, axis=0, zi=band[1])
            self.buf_p[i] = np.concatenate([self.buf_p[i][n:], y * y])
        self.n_seen += n
        valid = min(self.n_seen, self.window)
        feats = [self.buf_x[-valid:].mean(axis=0)] if self.cfg.include_lmp else []
        feats += [np.log(p[-valid:].mean(axis=0) + 1e-12) for p in self.buf_p]
        return np.stack(feats, axis=-1).reshape(-1).astype(np.float32)


def single_step(model: torch.nn.Module):
    """The ONNX-exportable single-step form of a trained decoder, its initial state and I/O names."""
    if isinstance(model, LSTMDecoder):
        step = LSTMStep(model)
        h, c = step.initial_state()
        return step, (h, c), ["h", "c"], ["h_out", "c_out"]
    if isinstance(model, S4DDecoder):
        step = S4DStep(model)
        return step, (step.initial_state(),), ["state"], ["state_out"]
    if isinstance(model, StateGatedDecoder):
        step = StateGatedStep(model)
        return step, step.initial_state(), ["zg", "ze"], ["zg_out", "ze_out"]
    raise TypeError(f"no single-step form for {type(model).__name__}")


def run_online(ecog: np.ndarray, extractor: StreamingFeatureExtractor, scaler: Standardiser, session,
               state: dict[str, np.ndarray], state_map: dict[str, str]) -> tuple[np.ndarray, np.ndarray]:
    """Stream a raw recording through features + ONNX decoder; returns outputs and per-step latency (ms)."""
    hop = extractor.cfg.hop
    names = [o.name for o in session.get_outputs()]
    feeds = dict(state)
    outputs, latencies = [], []
    for start in range(0, len(ecog) - hop + 1, hop):
        t0 = time.perf_counter()
        x_std = scaler.transform(extractor.process(ecog[start:start + hop]))
        x_std, _ = mask_artefact_channels(x_std, extractor.cfg.n_per_channel, extractor.cfg.artefact_z)
        feeds["x"] = x_std[None]
        result = dict(zip(names, session.run(None, feeds)))
        latencies.append((time.perf_counter() - t0) * 1e3)
        outputs.append(result["y"][0])
        for out_name, in_name in state_map.items():
            feeds[in_name] = result[out_name]
    return np.asarray(outputs), np.asarray(latencies)


def replay_online(model: torch.nn.Module, x_offline: np.ndarray, x_scaler: Standardiser, ecog: np.ndarray,
                  cfg: FeatureConfig, onnx_path: str | Path) -> dict:
    """Export, replay the whole recording online, compare with the offline output, time each step."""
    step, state0, in_names, out_names = single_step(model.eval())
    path = export_onnx(step, (torch.zeros(1, x_offline.shape[1]), *state0), ["x", *in_names], ["y", *out_names],
                       onnx_path)
    state = {n: s.numpy() for n, s in zip(in_names, state0)}
    streamed, latencies = run_online(ecog, StreamingFeatureExtractor(ecog.shape[1], cfg), x_scaler,
                                     onnx_session(path, threads=1), state, dict(zip(out_names, in_names)))
    with torch.no_grad():
        offline = model(torch.from_numpy(x_offline)[None])[0].numpy()
    return {"onnx_file": Path(path).name, "max_abs_diff_vs_offline": float(np.abs(streamed - offline).max()),
            **latency_summary(latencies)}
