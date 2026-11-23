"""A trained causal decoder packaged as it would run next to an implant: raw ECoG in, finger flexion out.

    decoder = OnlineDecoder.fit(train_ecog, train_flex, approach="lstm", n_electrodes=32)
    decoder.reset()
    for chunk in stream:                   # 40 new samples (40 ms at 1 kHz) x n_channels, raw
        flexion = decoder.step(chunk)      # five finger values, in glove units

Each step runs the benchmark's pipeline in the order a device would: median re-reference, notches and band power
over a trailing window (``StreamingFeatureExtractor``), z-scoring with training statistics, the electrodes the decoder
reads, the online artefact rule, one step of the network (PyTorch, or an exported ONNX graph after
``use_onnx``), the causal output smoother and the conversion back to glove units. ``decode`` streams a whole
recording; it reproduces the batch prediction of the trained model (tested).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .approaches import RunResult, run_network
from .config import DEFAULT_FEATURES, DEFAULT_PROTOCOL, FeatureConfig, Protocol, TrainConfig
from .data import SubjectData
from .evaluation import causal_ema
from .export import export_onnx, onnx_session
from .features import Standardiser, mask_artefact_channels
from .pipeline import Prepared, prepare
from .streaming import StreamingFeatureExtractor, single_step


class OnlineDecoder:
    """Streaming wrapper around a trained network with a single-step form (LSTM, S4D, gated or POSSM-style)."""

    def __init__(
        self,
        model: torch.nn.Module,
        x_scaler: Standardiser,
        y_scaler: Standardiser,
        n_channels: int,
        electrodes: list[int] | None = None,
        smooth_ms: float = 0.0,
        cfg: FeatureConfig = DEFAULT_FEATURES,
    ):
        self.model = model.eval()
        self.step_module, self.state0, self.in_names, self.out_names = single_step(self.model)
        self.x_scaler, self.y_scaler = x_scaler, y_scaler
        self.n_channels, self.cfg, self.smooth_ms = n_channels, cfg, smooth_ms
        self.electrodes = None if electrodes is None else [int(e) for e in electrodes]
        npc = cfg.n_per_channel
        self.columns = None if electrodes is None else (np.asarray(electrodes)[:, None] * npc + np.arange(npc)).ravel()
        self.session = None
        self.reset()

    # ------------------------------------------------------------------ construction
    @classmethod
    def from_run(cls, run: RunResult, prep: Prepared, n_channels: int) -> OnlineDecoder:
        """Package a network trained by ``approaches.run_network`` (its final, refitted model)."""
        y_scaler = Standardiser().fit(prep.y[prep.train_rows])  # the refit's target statistics, as in run_network
        return cls(
            run.model,
            run.x_scaler,
            y_scaler,
            n_channels,
            run.selected.get("electrodes"),
            run.selected.get("smooth_ms", 0.0),
            prep.cfg,
        )

    @classmethod
    def fit(
        cls,
        train_ecog: np.ndarray,
        train_flex: np.ndarray,
        approach: str = "lstm",
        n_electrodes: int | None = None,
        seed: int = 0,
        protocol: Protocol = DEFAULT_PROTOCOL,
        train_cfg: TrainConfig | None = None,
    ) -> OnlineDecoder:
        """Train one approach on a training recording (raw ECoG at 1 kHz, flexion at 1 kHz) under the shared protocol:
        choices on the last 20 %, then a refit on everything. ``n_electrodes`` keeps only the electrodes most related
        to movement (approach 5b's rule) for networks that accept it."""
        train_ecog = np.asarray(train_ecog, dtype=np.float32)
        data = SubjectData(0, train_ecog, np.asarray(train_flex, dtype=np.float32), train_ecog[:0], None)
        prep = prepare(data, protocol=protocol)
        run = run_network(prep, approach, seed, protocol, train_cfg=train_cfg, electrodes=n_electrodes)
        return cls.from_run(run, prep, train_ecog.shape[1])

    # ------------------------------------------------------------------ streaming
    def reset(self) -> None:
        """Start a new stream: empty filter states, trailing windows, recurrent state and smoother."""
        self.features = StreamingFeatureExtractor(self.n_channels, self.cfg)
        self.state = {n: s.clone() for n, s in zip(self.in_names, self.state0)}
        self.smoothed = None

    def step(self, chunk: np.ndarray) -> np.ndarray:
        """Take ``cfg.hop`` new raw samples (hop, n_channels); return the five finger values for this moment."""
        x = self.features.process(np.asarray(chunk))
        if self.columns is not None:
            x = x[self.columns]
        x = self.x_scaler.transform(x)
        x, _ = mask_artefact_channels(x, self.cfg.n_per_channel, self.cfg.artefact_z)
        if self.session is None:
            with torch.no_grad():
                y, *new = self.step_module(torch.from_numpy(x)[None], *self.state.values())
            self.state = dict(zip(self.in_names, new))
            y = y[0].numpy()
        else:
            out = dict(
                zip(
                    [o.name for o in self.session.get_outputs()],
                    self.session.run(None, {"x": x[None], **{n: s.numpy() for n, s in self.state.items()}}),
                )
            )
            self.state = {i: torch.from_numpy(out[o]) for i, o in zip(self.in_names, self.out_names)}
            y = out["y"][0]
        if self.smooth_ms > 0:
            a = 1.0 - np.exp(-self.cfg.hop / self.smooth_ms)
            self.smoothed = y if self.smoothed is None else self.smoothed + a * (y - self.smoothed)
            y = self.smoothed
        return self.y_scaler.inverse_transform(y)

    def decode(self, ecog: np.ndarray) -> np.ndarray:
        """Stream a whole recording from a fresh state; one output per ``cfg.hop`` samples."""
        self.reset()
        hop = self.cfg.hop
        return np.stack([self.step(ecog[i : i + hop]) for i in range(0, len(ecog) - hop + 1, hop)])

    # ------------------------------------------------------------------ deployment
    def export_onnx(self, path: str | Path) -> Path:
        """Export the network's single step (features in, five outputs and the next state out) to ONNX."""
        n_in = len(self.columns) if self.columns is not None else self.n_channels * self.cfg.n_per_channel
        return export_onnx(
            self.step_module, (torch.zeros(1, n_in), *self.state0), ["x", *self.in_names], ["y", *self.out_names], path
        )

    def use_onnx(self, path: str | Path, threads: int = 1) -> OnlineDecoder:
        """Run every following step through ONNX Runtime instead of PyTorch."""
        self.session = onnx_session(self.export_onnx(path), threads=threads)
        return self


def batch_prediction(decoder: OnlineDecoder, ecog: np.ndarray) -> np.ndarray:
    """The same decoder run offline over a whole recording at once (for checks): batch features, one forward pass."""
    from .features import extract_features

    x, _ = extract_features(ecog, decoder.cfg)
    if decoder.columns is not None:
        x = x[:, decoder.columns]
    x, _ = mask_artefact_channels(decoder.x_scaler.transform(x), decoder.cfg.n_per_channel, decoder.cfg.artefact_z)
    with torch.no_grad():
        y = decoder.model(torch.from_numpy(x)[None])[0].numpy()
    return decoder.y_scaler.inverse_transform(causal_ema(y, decoder.smooth_ms, decoder.cfg.hop))
