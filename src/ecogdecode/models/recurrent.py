"""Recurrent decoders: an LSTM and a controlled neural ODE (continuous-time latent dynamics)."""
from __future__ import annotations

import torch
from torch import nn


# --------------------------------------------------------------------------- LSTM
class LSTMDecoder(nn.Module):
    def __init__(self, n_features: int, n_outputs: int = 5, hidden: int = 128, layers: int = 1,
                 dropout: float = 0.2):
        super().__init__()
        self.inp = nn.Sequential(nn.Dropout(dropout), nn.Linear(n_features, hidden), nn.GELU())
        self.rnn = nn.LSTM(hidden, hidden, num_layers=layers, batch_first=True)
        self.out = nn.Linear(hidden, n_outputs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (batch, time, features)
        h, _ = self.rnn(self.inp(x))
        return self.out(h)


class LSTMStep(nn.Module):
    """One 40 ms step of a trained LSTMDecoder, with the recurrent state passed explicitly."""

    def __init__(self, model: LSTMDecoder):
        super().__init__()
        self.model = model.eval()

    def forward(self, x: torch.Tensor, h: torch.Tensor, c: torch.Tensor):
        # x: (1, features); h, c: (layers, 1, hidden)
        out, (h, c) = self.model.rnn(self.model.inp(x).unsqueeze(1), (h, c))
        return self.model.out(out[:, 0]), h, c

    def initial_state(self):
        rnn = self.model.rnn
        shape = (rnn.num_layers, 1, rnn.hidden_size)
        return torch.zeros(shape), torch.zeros(shape)


# --------------------------------------------------------------------------- neural ODE
class ODEDecoder(nn.Module):
    """Continuous-time latent dynamics driven by the neural features (a controlled neural ODE).

        dh/dt = -h / tau + tanh(W [h, u(t)] + b),     finger flexion = readout(h)

    u(t) is held constant over each 40 ms step (zero-order hold, as in the hardware), and
    the ODE is integrated with one classical Runge-Kutta (RK4) step per 40 ms. The learned
    time constants tau give each latent unit its own timescale.
    """

    def __init__(self, n_features: int, n_outputs: int = 5, hidden: int = 64, d_input: int = 64,
                 dropout: float = 0.2):
        super().__init__()
        self.inp = nn.Sequential(nn.Dropout(dropout), nn.Linear(n_features, d_input), nn.GELU())
        self.f = nn.Linear(hidden + d_input, hidden)
        self.log_tau = nn.Parameter(torch.log(torch.linspace(1.0, 25.0, hidden)))  # 40 ms to 1 s
        self.out = nn.Linear(hidden, n_outputs)
        self.hidden = hidden

    def vector_field(self, h: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        return -h / torch.exp(self.log_tau) + torch.tanh(self.f(torch.cat([h, u], dim=-1)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u_seq = self.inp(x)
        h = x.new_zeros(x.shape[0], self.hidden)
        outs = []
        for t in range(x.shape[1]):  # one RK4 step per 40 ms step (dt = 1 step)
            u = u_seq[:, t]
            k1 = self.vector_field(h, u)
            k2 = self.vector_field(h + 0.5 * k1, u)
            k3 = self.vector_field(h + 0.5 * k2, u)
            k4 = self.vector_field(h + k3, u)
            h = h + (k1 + 2 * k2 + 2 * k3 + k4) / 6
            outs.append(h)
        return self.out(torch.stack(outs, dim=1))
