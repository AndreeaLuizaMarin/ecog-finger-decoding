"""Neuromorphic-style decoders gated by brain state (new in this project).

Leaky integrators (the membrane equation of a leaky-integrator neuron) feed a mixture of linear
experts; a gate decides, at every step, which expert to trust. ``StateGatedDecoder`` gates on the
slow, LFP-like part of the ECoG; ``TransientGatedDecoder`` also gives the gate movement onset and
offset transients (inspired by Dekleva et al., 2021) and compresses its inputs with PCA.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


# --------------------------------------------------------------------------- state-gated decoder
class LeakyIntegrator(nn.Module):
    """Bank of first-order leaky integrators, z_t = a z_{t-1} + (1 - a) u_t, one learned decay per unit.

    This is the membrane equation of a leaky-integrator neuron in discrete time, the basic
    unit of neuromorphic hardware. Training uses an equivalent causal FFT convolution.
    """

    def __init__(self, dim: int, min_steps: float = 1.0, max_steps: float = 25.0):
        super().__init__()
        tau = torch.exp(torch.linspace(math.log(min_steps), math.log(max_steps), dim))
        a = torch.exp(-1.0 / tau)
        self.logit_a = nn.Parameter(torch.log(a / (1 - a)))

    @property
    def decay(self) -> torch.Tensor:
        return torch.sigmoid(self.logit_a)

    def forward(self, u: torch.Tensor) -> torch.Tensor:  # (batch, time, dim)
        length = u.shape[1]
        a = self.decay[:, None]
        k = (1 - a) * a ** torch.arange(length, device=u.device)  # (dim, length)
        n = 2 * length
        y = torch.fft.irfft(torch.fft.rfft(u.transpose(1, 2), n=n) * torch.fft.rfft(k, n=n), n=n)[..., :length]
        return y.transpose(1, 2)

    def step(self, z: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        a = self.decay
        return a * z + (1 - a) * u


class StateGatedDecoder(nn.Module):
    """Oscillation-state-gated mixture of leaky-integrator experts.

    Two pathways, both causal:
    * a gating pathway reads the slow, oscillatory part of the ECoG (local motor potential,
      mu and beta power: the LFP-like "brain state") and outputs, at every step, soft
      weights over K experts;
    * an expert pathway reads high-gamma power through leaky integrators; each expert is
      a linear readout of the integrated activity.
    Output = sum_k gate_k * expert_k. The gate is a soft switch between decoding regimes,
    inspired by the idea that low-frequency oscillatory state signals which internal map
    or movement regime is active. With ``gate_features="all"`` the gate sees every feature
    (an ablation of that hypothesis).
    """

    def __init__(self, n_features: int, n_per_channel: int, n_outputs: int = 5, n_experts: int = 6,
                 d_gate: int = 32, d_latent: int = 64, slow=(0, 1, 2), gate_features: str = "slow",
                 dropout: float = 0.2):
        super().__init__()
        n_channels = n_features // n_per_channel
        fast = [j for j in range(n_per_channel) if j not in slow]
        gate_cols = slow if gate_features == "slow" else tuple(range(n_per_channel))
        idx = lambda cols: torch.tensor([c * n_per_channel + j for c in range(n_channels) for j in cols])
        self.register_buffer("gate_idx", idx(gate_cols))
        self.register_buffer("fast_idx", idx(fast))
        self.gate_in = nn.Sequential(nn.Dropout(dropout), nn.Linear(len(self.gate_idx), d_gate))
        self.gate_li = LeakyIntegrator(d_gate)
        self.gate_out = nn.Linear(d_gate, n_experts)
        self.expert_in = nn.Sequential(nn.Dropout(dropout), nn.Linear(len(self.fast_idx), d_latent))
        self.expert_li = LeakyIntegrator(d_latent)
        self.experts = nn.Linear(d_latent, n_experts * n_outputs)
        self.n_experts, self.n_outputs = n_experts, n_outputs

    def gates(self, x: torch.Tensor) -> torch.Tensor:
        g = self.gate_li(self.gate_in(x[..., self.gate_idx]))
        return torch.softmax(self.gate_out(F.gelu(g)), dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.expert_li(self.expert_in(x[..., self.fast_idx]))
        experts = self.experts(F.gelu(z)).unflatten(-1, (self.n_experts, self.n_outputs))
        return (self.gates(x)[..., None] * experts).sum(-2)


class StateGatedStep(nn.Module):
    """One 40 ms step of a trained StateGatedDecoder; state = (gate integrators, expert integrators)."""

    def __init__(self, model: StateGatedDecoder):
        super().__init__()
        self.model = model.eval()

    def initial_state(self):
        m = self.model
        return torch.zeros(1, m.gate_out.in_features), torch.zeros(1, m.experts.in_features)

    def forward(self, x: torch.Tensor, zg: torch.Tensor, ze: torch.Tensor):
        m = self.model
        zg = m.gate_li.step(zg, m.gate_in(x[:, m.gate_idx]))
        ze = m.expert_li.step(ze, m.expert_in(x[:, m.fast_idx]))
        gates = torch.softmax(m.gate_out(F.gelu(zg)), dim=-1)
        experts = m.experts(F.gelu(ze)).unflatten(-1, (m.n_experts, m.n_outputs))
        return (gates[..., None] * experts).sum(-2), zg, ze


class TransientGatedDecoder(nn.Module):
    """State-gated decoder whose gate also sees movement onset/offset transients.

    Inspired by Dekleva et al. (2021): transient neural responses at the start and end of an
    attempted grasp were larger and more reliably detected than the sustained response. Here
    the gate reads (i) the slow LFP-band state and (ii) the step-to-step change of high-gamma
    activity (onset/offset transients). Inputs are first compressed with PCA fitted on the
    training data, a low-dimensional constraint that limits overfitting on small data.
    """

    def __init__(self, slow_components: np.ndarray, fast_components: np.ndarray, slow_idx, fast_idx,
                 n_outputs: int = 5, n_experts: int = 6, d_gate: int = 32, d_latent: int = 64, dropout: float = 0.2):
        super().__init__()
        self.register_buffer("slow_idx", torch.as_tensor(slow_idx))
        self.register_buffer("fast_idx", torch.as_tensor(fast_idx))
        self.register_buffer("P_slow", torch.as_tensor(slow_components, dtype=torch.float32))  # (k, n_slow)
        self.register_buffer("P_fast", torch.as_tensor(fast_components, dtype=torch.float32))  # (k, n_fast)
        k_s, k_f = self.P_slow.shape[0], self.P_fast.shape[0]
        self.gate_in = nn.Sequential(nn.Dropout(dropout), nn.Linear(k_s + k_f, d_gate))
        self.gate_li = LeakyIntegrator(d_gate)
        self.gate_out = nn.Linear(d_gate, n_experts)
        self.expert_in = nn.Sequential(nn.Dropout(dropout), nn.Linear(k_f, d_latent))
        self.expert_li = LeakyIntegrator(d_latent)
        self.experts = nn.Linear(d_latent, n_experts * n_outputs)
        self.n_experts, self.n_outputs = n_experts, n_outputs

    def _components(self, x: torch.Tensor):
        slow = x[..., self.slow_idx] @ self.P_slow.T
        fast = x[..., self.fast_idx] @ self.P_fast.T
        transient = fast - F.pad(fast, (0, 0, 1, 0))[:, :-1]  # x_t - x_{t-1}, zero at the start (causal)
        return slow, fast, transient

    def gates(self, x: torch.Tensor) -> torch.Tensor:
        slow, _, transient = self._components(x)
        g = self.gate_li(self.gate_in(torch.cat([slow, transient], dim=-1)))
        return torch.softmax(self.gate_out(F.gelu(g)), dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, fast, _ = self._components(x)
        experts = self.experts(F.gelu(self.expert_li(self.expert_in(fast)))).unflatten(-1, (self.n_experts, self.n_outputs))
        return (self.gates(x)[..., None] * experts).sum(-2)

    @classmethod
    def from_training_data(cls, x_fit: np.ndarray, n_per_channel: int, k: int = 16, slow=(0, 1, 2)) -> "TransientGatedDecoder":
        """Fit the PCA projections on the training rows only and build the model."""
        n_channels = x_fit.shape[1] // n_per_channel
        fast = [j for j in range(n_per_channel) if j not in slow]
        slow_idx = [c * n_per_channel + j for c in range(n_channels) for j in slow]
        fast_idx = [c * n_per_channel + j for c in range(n_channels) for j in fast]
        return cls(_pca(x_fit[:, slow_idx], k), _pca(x_fit[:, fast_idx], k), slow_idx, fast_idx)

    def refit_to(self, x: np.ndarray, y: np.ndarray | None = None, seed: int = 0) -> None:
        """Re-estimate the PCA projections (used when retraining on the whole training segment)."""
        k = self.P_slow.shape[0]
        with torch.no_grad():
            self.P_slow.copy_(torch.as_tensor(_pca(x[:, self.slow_idx.numpy()], k)))
            self.P_fast.copy_(torch.as_tensor(_pca(x[:, self.fast_idx.numpy()], k)))


def _pca(x: np.ndarray, k: int) -> np.ndarray:
    """Leading ``k`` principal axes (rows) of ``x``."""
    _, _, vt = np.linalg.svd(x - x.mean(axis=0), full_matrices=False)
    return vt[:k].astype(np.float32)
