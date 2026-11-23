"""Neural-manifold gated decoder: switching between decoding regimes according to where the
current brain state sits on a low-dimensional neural manifold.

Idea: population activity lives on a low-dimensional
manifold, and different regions of that manifold correspond to different behavioural regimes
(rest, or one finger or another moving). A gate that reads the *position on the manifold* can
therefore switch, step by step, between specialised decoders. Two manifolds are provided:

* ``PCAManifold``: a fixed linear projection onto the leading principal components of the
  training features (unsupervised).
* ``ContrastiveManifold``: a small causal encoder trained in the spirit of CEBRA-behaviour
  (Schneider, Lee and Mathis, 2023): neural states recorded during similar finger postures are
  pulled together on a hypersphere, others pushed apart (InfoNCE). Unlike CEBRA's default
  models, which look a few samples into the future, this encoder only sees the past.

The decoder itself uses the same neuromorphic-style building blocks as the other gated decoders:
leaky integrators and a softmax gate over linear experts. The manifold is fitted first and kept
fixed while the gate and experts are trained.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial import cKDTree
from torch import nn

from .gated import LeakyIntegrator, _pca


class PCAManifold(nn.Module):
    def __init__(self, components: np.ndarray):
        super().__init__()
        self.register_buffer("P", torch.as_tensor(components, dtype=torch.float32))  # (dim, n_features)

    @property
    def dim(self) -> int:
        return self.P.shape[0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.P.T

    def fit(self, x: np.ndarray, y: np.ndarray | None = None, seed: int = 0) -> "PCAManifold":
        with torch.no_grad():
            self.P.copy_(torch.as_tensor(_pca(x, self.dim)))
        return self


class ContrastiveManifold(nn.Module):
    """Causal encoder: features -> leaky integrators -> point on the unit sphere."""

    def __init__(self, n_features: int, dim: int = 8, hidden: int = 64, steps: int = 800, window: int = 64,
                 batch_size: int = 128, temperature: float = 0.1, behaviour_noise: float = 0.1):
        super().__init__()
        self.inp = nn.Linear(n_features, hidden)
        self.li = LeakyIntegrator(hidden)
        self.out = nn.Linear(hidden, dim)
        self.hparams = dict(steps=steps, window=window, batch_size=batch_size, temperature=temperature,
                            behaviour_noise=behaviour_noise)

    @property
    def dim(self) -> int:
        return self.out.out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.out(F.gelu(self.li(F.gelu(self.inp(x))))), dim=-1)

    def fit(self, x: np.ndarray, y: np.ndarray, seed: int = 0) -> "ContrastiveManifold":
        """Behaviour-contrastive training on contiguous training data ``x`` (n, f), ``y`` (n, 5).

        For each anchor step t, the positive is the step whose finger posture is closest to
        y[t] + noise (CEBRA-behaviour sampling); the other positives in the batch act as negatives.
        Each sample is embedded from the window of ``window`` steps that ends at it.
        """
        h = self.hparams
        rng = np.random.default_rng(seed)
        torch.manual_seed(seed)
        w = h["window"]
        candidates = np.arange(w - 1, len(x))
        tree = cKDTree(y[candidates])
        xt = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))
        offsets = torch.arange(-w + 1, 1)
        opt = torch.optim.AdamW(self.parameters(), lr=1e-3, weight_decay=1e-2)
        self.train()
        for _ in range(h["steps"]):
            anchors = rng.choice(candidates, size=h["batch_size"])
            target = y[anchors] + rng.normal(0, h["behaviour_noise"], size=(len(anchors), y.shape[1]))
            positives = candidates[tree.query(target)[1]]
            ends = torch.from_numpy(np.concatenate([anchors, positives]))
            emb = self(xt[ends[:, None] + offsets])[:, -1]  # embedding at the end of each window
            za, zp = emb[: len(anchors)], emb[len(anchors):]
            logits = za @ zp.T / h["temperature"]
            loss = F.cross_entropy(logits, torch.arange(len(anchors)))
            opt.zero_grad()
            loss.backward()
            opt.step()
        self.eval()
        self.final_loss_ = loss.item()
        return self


class ManifoldGatedDecoder(nn.Module):
    """Gate on the neural manifold; experts read the full feature vector through leaky integrators
    (``backbone="leaky"``, ``d_latent`` units) or are linear read-outs of a shared LSTM (``backbone="lstm"``,
    ``d_rnn`` units)."""

    def __init__(self, manifold: nn.Module, n_features: int, n_outputs: int = 5, n_experts: int = 6,
                 d_gate: int = 32, d_latent: int = 64, dropout: float = 0.2, backbone: str = "leaky",
                 d_rnn: int = 128):
        super().__init__()
        self.manifold = manifold
        for p in self.manifold.parameters():
            p.requires_grad_(False)  # the manifold is fitted first and then kept fixed
        self.gate_in = nn.Linear(manifold.dim, d_gate)
        self.gate_li = LeakyIntegrator(d_gate)
        self.gate_out = nn.Linear(d_gate, n_experts)
        self.backbone = backbone
        if backbone == "leaky":  # neuromorphic-style: leaky integrators feed linear experts
            self.expert_in = nn.Sequential(nn.Dropout(dropout), nn.Linear(n_features, d_latent))
            self.expert_li = LeakyIntegrator(d_latent)
        elif backbone == "lstm":  # a shared recurrent state (d_rnn units); the gate chooses its read-out
            d_latent = d_rnn
            self.expert_in = nn.Sequential(nn.Dropout(dropout), nn.Linear(n_features, d_latent), nn.GELU())
            self.expert_rnn = nn.LSTM(d_latent, d_latent, batch_first=True)
        else:
            raise ValueError(backbone)
        self.experts = nn.Linear(d_latent, n_experts * n_outputs)
        self.n_experts, self.n_outputs = n_experts, n_outputs

    def gates(self, x: torch.Tensor) -> torch.Tensor:
        z = self.manifold(x)
        return torch.softmax(self.gate_out(F.gelu(self.gate_li(self.gate_in(z)))), dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.backbone == "leaky":
            h = F.gelu(self.expert_li(self.expert_in(x)))
        else:
            h, _ = self.expert_rnn(self.expert_in(x))
        experts = self.experts(h).unflatten(-1, (self.n_experts, self.n_outputs))
        return (self.gates(x)[..., None] * experts).sum(-2)

    def refit_to(self, x: np.ndarray, y: np.ndarray, seed: int = 0) -> None:
        """Re-estimate the manifold on new training data (used when retraining on the whole segment)."""
        for p in self.manifold.parameters():
            p.requires_grad_(True)
        self.manifold.fit(x, y, seed=seed)
        for p in self.manifold.parameters():
            p.requires_grad_(False)


def build_manifold_gated(kind: str, x_fit: np.ndarray, y_fit: np.ndarray, dim: int = 8, seed: int = 0,
                         backbone: str = "leaky", **decoder_kwargs) -> ManifoldGatedDecoder:
    """Fit the manifold on the training rows only, then wrap it in a gated decoder
    (``decoder_kwargs`` go to ``ManifoldGatedDecoder``, e.g. ``d_rnn`` and ``dropout``)."""
    if kind == "pca":
        manifold = PCAManifold(_pca(x_fit, dim))
    elif kind == "contrastive":
        manifold = ContrastiveManifold(x_fit.shape[1], dim).fit(x_fit, y_fit, seed=seed)
    else:
        raise ValueError(kind)
    return ManifoldGatedDecoder(manifold, x_fit.shape[1], backbone=backbone, **decoder_kwargs)
