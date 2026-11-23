"""Causal temporal convolutional network (TCN): dilated convolutions that only look back in time."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class CausalConvBlock(nn.Module):
    def __init__(self, channels: int, kernel: int, dilation: int, dropout: float):
        super().__init__()
        self.pad = (kernel - 1) * dilation  # left padding only: output t sees inputs <= t
        self.conv = nn.Conv1d(channels, channels, kernel, dilation=dilation)
        self.norm = nn.LayerNorm(channels)  # per time step; GroupNorm would pool over time and see the future
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (batch, channels, time)
        h = self.conv(F.pad(x, (self.pad, 0)))
        h = self.norm(h.transpose(1, 2)).transpose(1, 2)
        return x + self.drop(F.gelu(h))


class TCNDecoder(nn.Module):
    """Dilations 1, 2, 4, 8, 16 with kernel 3: a receptive field of 63 steps (2.5 s)."""

    def __init__(
        self,
        n_features: int,
        n_outputs: int = 5,
        channels: int = 128,
        kernel: int = 3,
        dilations: tuple[int, ...] = (1, 2, 4, 8, 16),
        dropout: float = 0.2,
    ):
        super().__init__()
        self.inp = nn.Sequential(nn.Dropout(dropout), nn.Linear(n_features, channels), nn.GELU())
        self.blocks = nn.ModuleList(CausalConvBlock(channels, kernel, d, dropout) for d in dilations)
        self.out = nn.Linear(channels, n_outputs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (batch, time, features)
        h = self.inp(x).transpose(1, 2)
        for block in self.blocks:
            h = block(h)
        return self.out(h.transpose(1, 2))
