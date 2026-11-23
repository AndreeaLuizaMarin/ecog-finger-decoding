"""State-space decoders: S4D (Gu et al., 2022) and Mamba (Gu and Dao, 2023).

S4D: a diagonal complex state matrix with S4D-Lin initialisation and zero-order-hold
discretisation, trained as an FFT convolution and run as a recurrence (``S4DStep``).
Mamba: input-dependent step size and B, C matrices (a selective scan), minimal pure-PyTorch version.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


# --------------------------------------------------------------------------- S4D
class S4DLayer(nn.Module):
    """Diagonal state-space layer: one independent complex SSM per channel."""

    def __init__(self, d_model: int, d_state: int = 64, dt_min: float = 1e-3, dt_max: float = 1e-1):
        super().__init__()
        n = d_state // 2  # complex conjugate pairs
        log_dt = torch.rand(d_model) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        self.log_dt = nn.Parameter(log_dt)
        self.log_A_real = nn.Parameter(torch.log(0.5 * torch.ones(d_model, n)))
        self.A_imag = nn.Parameter(math.pi * torch.arange(n, dtype=torch.float32).repeat(d_model, 1))
        self.C = nn.Parameter(torch.view_as_real(torch.randn(d_model, n, dtype=torch.cfloat) * math.sqrt(0.5)))
        self.D = nn.Parameter(torch.randn(d_model))

    def discretize(self):
        """Zero-order hold with B = 1: returns dtA, dA = exp(dtA), dB and C (all complex, (H, N))."""
        dt = torch.exp(self.log_dt)[:, None]
        A = -torch.exp(self.log_A_real) + 1j * self.A_imag
        dtA = dt * A
        dA = torch.exp(dtA)
        dB = (dA - 1.0) / A
        return dtA, dA, dB, torch.view_as_complex(self.C)

    def kernel(self, length: int, chunk: int = 2048) -> torch.Tensor:
        """Convolution kernel K[h, l] = 2 Re(sum_n C dB dA^l), computed in chunks to bound memory."""
        dtA, _, dB, C = self.discretize()
        cb = C * dB
        parts = []
        for start in range(0, length, chunk):
            pos = torch.arange(start, min(start + chunk, length), device=dtA.device)
            parts.append(2 * torch.einsum("hn,hnl->hl", cb, torch.exp(dtA[..., None] * pos)).real)
        return torch.cat(parts, dim=-1)

    def forward(self, u: torch.Tensor) -> torch.Tensor:  # (batch, time, d_model)
        length = u.shape[1]
        k = self.kernel(length)
        u_t = u.transpose(1, 2)
        n = 2 * length  # zero-padding makes the circular FFT convolution causal
        y = torch.fft.irfft(torch.fft.rfft(u_t, n=n) * torch.fft.rfft(k, n=n), n=n)[..., :length]
        return (y + u_t * self.D[:, None]).transpose(1, 2)


class S4DBlock(nn.Module):
    def __init__(self, d_model: int, d_state: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.ssm = S4DLayer(d_model, d_state)
        self.mix = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.drop(self.mix(F.gelu(self.ssm(self.norm(x)))))


class S4DDecoder(nn.Module):
    def __init__(self, n_features: int, n_outputs: int = 5, d_model: int = 128, d_state: int = 64,
                 n_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.inp = nn.Sequential(nn.Dropout(dropout), nn.Linear(n_features, d_model), nn.GELU())
        self.blocks = nn.ModuleList(S4DBlock(d_model, d_state, dropout) for _ in range(n_layers))
        self.norm = nn.LayerNorm(d_model)
        self.out = nn.Linear(d_model, n_outputs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.inp(x)
        for block in self.blocks:
            h = block(h)
        return self.out(self.norm(h))


class S4DStep(nn.Module):
    """Recurrent (streaming) form of a trained S4DDecoder, in real arithmetic for ONNX.

    For each layer the complex state x evolves as x <- dA * x + dB * u and the output is
    y = 2 Re(sum_n C x) + D u. Real and imaginary parts are stored separately in
    ``state`` of shape (layers, 2, d_model, n).
    """

    def __init__(self, model: S4DDecoder):
        super().__init__()
        self.model = model.eval()
        with torch.no_grad():
            for i, block in enumerate(model.blocks):
                _, dA, dB, C = block.ssm.discretize()
                for name, value in (("dA", dA), ("dB", dB), ("C", C)):
                    self.register_buffer(f"{name}_re_{i}", value.real.clone())
                    self.register_buffer(f"{name}_im_{i}", value.imag.clone())

    def initial_state(self) -> torch.Tensor:
        ssm = self.model.blocks[0].ssm
        h, n = ssm.log_A_real.shape
        return torch.zeros(len(self.model.blocks), 2, h, n)

    def forward(self, x: torch.Tensor, state: torch.Tensor):
        # x: (1, features); state: (layers, 2, d_model, n)
        h = self.model.inp(x)[0]
        new_states = []
        for i, block in enumerate(self.model.blocks):
            u = block.norm(h)
            xr, xi = state[i, 0], state[i, 1]
            dar, dai = getattr(self, f"dA_re_{i}"), getattr(self, f"dA_im_{i}")
            dbr, dbi = getattr(self, f"dB_re_{i}"), getattr(self, f"dB_im_{i}")
            cr, ci = getattr(self, f"C_re_{i}"), getattr(self, f"C_im_{i}")
            nr = dar * xr - dai * xi + dbr * u[:, None]
            ni = dar * xi + dai * xr + dbi * u[:, None]
            y = 2 * (cr * nr - ci * ni).sum(-1) + block.ssm.D * u
            h = h + block.mix(F.gelu(y))
            new_states.append(torch.stack([nr, ni]))
        return self.model.out(self.model.norm(h))[None], torch.stack(new_states)


# --------------------------------------------------------------------------- Mamba
class MambaLayer(nn.Module):
    """Selective state-space layer (Gu and Dao, 2023), minimal pure-PyTorch version.

    Unlike S4D, the step size and the B and C matrices depend on the current input, so
    the layer can decide, step by step, what to remember. A short causal depthwise
    convolution precedes the scan and a SiLU gate follows it.
    """

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2,
                 dt_min: float = 1e-3, dt_max: float = 1e-1):
        super().__init__()
        d_inner = expand * d_model
        self.dt_rank = max(1, math.ceil(d_model / 16))
        self.d_state, self.d_conv = d_state, d_conv
        self.in_proj = nn.Linear(d_model, 2 * d_inner)
        self.conv = nn.Conv1d(d_inner, d_inner, d_conv, groups=d_inner)
        self.x_proj = nn.Linear(d_inner, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, d_inner)
        dt = torch.exp(torch.rand(d_inner) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min))
        with torch.no_grad():
            self.dt_proj.bias.copy_(dt + torch.log(-torch.expm1(-dt)))  # inverse softplus
        self.A_log = nn.Parameter(torch.log(torch.arange(1, d_state + 1, dtype=torch.float32)).repeat(d_inner, 1))
        self.D = nn.Parameter(torch.ones(d_inner))
        self.out_proj = nn.Linear(d_inner, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (batch, time, d_model)
        length = x.shape[1]
        u, z = self.in_proj(x).chunk(2, dim=-1)
        u = F.pad(u.transpose(1, 2), (self.d_conv - 1, 0))  # left padding keeps the convolution causal
        u = F.silu(self.conv(u)[..., :length].transpose(1, 2))
        dt, B, C = torch.split(self.x_proj(u), [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = F.softplus(self.dt_proj(dt))  # (batch, time, d_inner)
        y = selective_scan(u, dt, -torch.exp(self.A_log), B, C) + u * self.D
        return self.out_proj(y * F.silu(z))


def selective_scan(u, dt, A, B, C, chunk: int = 16):
    """h_t = exp(dt_t A) h_{t-1} + dt_t B_t u_t,  y_t = C_t . h_t   (exact, causal).

    Within each chunk the recurrence is solved in closed form with cumulative sums of the
    log-decay; the state is carried from chunk to chunk. Chunks keep exp() in a safe range.
    Shapes: u, dt (b, L, d); A (d, n); B, C (b, L, n).
    """
    b, length, d = u.shape
    h = u.new_zeros(b, d, A.shape[1])
    ys = []
    for s in range(0, length, chunk):
        dt_c, u_c = dt[:, s:s + chunk], u[:, s:s + chunk]
        B_c, C_c = B[:, s:s + chunk], C[:, s:s + chunk]
        log_decay = torch.cumsum(dt_c[..., None] * A, dim=1)  # (b, l, d, n), <= 0
        drive = (dt_c * u_c)[..., None] * B_c[:, :, None, :]  # (b, l, d, n)
        # h_t = exp(L_t) h_0 + sum_{s<=t} exp(L_t - L_s) drive_s
        weights = torch.exp(log_decay[:, :, None] - log_decay[:, None, :])  # (b, t, s, d, n)
        mask = torch.tril(torch.ones(log_decay.shape[1], log_decay.shape[1], dtype=torch.bool, device=u.device))
        weights = weights * mask[None, :, :, None, None]
        h_seq = torch.exp(log_decay) * h[:, None] + (weights * drive[:, None]).sum(2)
        ys.append((h_seq * C_c[:, :, None, :]).sum(-1))
        h = h_seq[:, -1]
    return torch.cat(ys, dim=1)


class MambaDecoder(nn.Module):
    def __init__(self, n_features: int, n_outputs: int = 5, d_model: int = 48, d_state: int = 8,
                 n_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.inp = nn.Sequential(nn.Dropout(dropout), nn.Linear(n_features, d_model), nn.GELU())
        self.norms = nn.ModuleList(nn.LayerNorm(d_model) for _ in range(n_layers))
        self.layers = nn.ModuleList(MambaLayer(d_model, d_state=d_state) for _ in range(n_layers))
        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)
        self.out = nn.Linear(d_model, n_outputs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.inp(x)
        for norm, layer in zip(self.norms, self.layers):
            h = h + self.drop(layer(norm(h)))
        return self.out(self.norm(h))
