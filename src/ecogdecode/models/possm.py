"""A POSSM-style decoder for ECoG: electrode tokens, cross-attention to a few latents, a recurrent backbone.

POSSM (Ryoo, Krishna, Mao, Azabou, Dyer, Perich and Lajoie, NeurIPS 2025, arXiv:2506.05320) decodes spiking
activity in real time: every spike becomes a token (a learned unit embedding plus its timing), a small set of learned
latent queries cross-attends to the tokens of each short time chunk, and a recurrent state-space backbone carries
the latents forward, so inference is causal and cheap. Because units enter as tokens with their own embeddings, one
model can be trained on many sessions and adapted to a new one by learning only the new units' embeddings ("unit
identification"), which is what makes it a candidate for cross-patient pretraining.

The same structure for ECoG band power, every 40 ms:
* one token per electrode: a shared projection of its features (local motor potential and five band powers) plus a
  learned embedding of that electrode (per patient, since electrode positions differ between patients);
* ``n_latents`` learned queries cross-attend to the electrode tokens of the current step only (no mixing over time);
* a GRU carries the latents over time; a linear read-out gives the five fingers.

Everything is causal (attention is within the current step, the GRU runs forward). An absent electrode (a masked
artefact, set to the training mean) still contributes its embedding; ``key_padding_mask`` support is left out to keep
the single-step form exportable.
"""
from __future__ import annotations

import torch
from torch import nn


class ElectrodeTokenizer(nn.Module):
    """Electrode tokens -> ``n_latents`` x ``d_model`` latents per step, by cross-attention."""

    def __init__(self, n_channels: int, n_per_channel: int, d_model: int = 64, n_latents: int = 4, n_heads: int = 4,
                 dropout: float = 0.1):
        super().__init__()
        self.n_channels, self.n_per_channel = n_channels, n_per_channel
        self.value = nn.Linear(n_per_channel, d_model)  # shared across electrodes and patients
        self.electrode = nn.Parameter(0.02 * torch.randn(n_channels, d_model))  # patient-specific identity
        self.latents = nn.Parameter(0.02 * torch.randn(n_latents, d_model))
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm_tokens = nn.LayerNorm(d_model)
        self.norm_out = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (batch, time, n_channels * n_per_channel)
        b, t, _ = x.shape
        tokens = self.value(x.reshape(b * t, self.n_channels, self.n_per_channel)) + self.electrode
        tokens = self.norm_tokens(self.drop(tokens))
        queries = self.latents.expand(b * t, -1, -1)
        out, _ = self.attn(queries, tokens, tokens, need_weights=False)
        return self.norm_out(out + queries).reshape(b, t, -1)  # (batch, time, n_latents * d_model)

    def new_patient(self, n_channels: int) -> None:
        """Fresh electrode embeddings for another patient; every other weight is kept."""
        self.n_channels = n_channels
        self.electrode = nn.Parameter(0.02 * torch.randn(n_channels, self.latents.shape[1]))


class POSSMDecoder(nn.Module):
    """Electrode tokenizer -> GRU -> linear read-out."""

    def __init__(self, n_features: int, n_per_channel: int, n_outputs: int = 5, d_model: int = 64,
                 n_latents: int = 4, hidden: int = 256, dropout: float = 0.1):
        super().__init__()
        self.tokenizer = ElectrodeTokenizer(n_features // n_per_channel, n_per_channel, d_model, n_latents,
                                            dropout=dropout)
        self.rnn = nn.GRU(n_latents * d_model, hidden, batch_first=True)
        self.out = nn.Linear(hidden, n_outputs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self.rnn(self.tokenizer(x))
        return self.out(h)

    def new_patient(self, n_features: int) -> "POSSMDecoder":
        self.tokenizer.new_patient(n_features // self.tokenizer.n_per_channel)
        return self

    def identification_parameters(self) -> list[nn.Parameter]:
        """The parameters learned for a new patient during unit (here: electrode) identification."""
        return [self.tokenizer.electrode, *self.out.parameters()]


class POSSMStep(nn.Module):
    """One 40 ms step of a trained POSSMDecoder, with the GRU state passed explicitly (ONNX-exportable)."""

    def __init__(self, model: POSSMDecoder):
        super().__init__()
        self.model = model.eval()

    def forward(self, x: torch.Tensor, h: torch.Tensor):  # x: (1, features); h: (1, 1, hidden)
        z = self.model.tokenizer(x[:, None])
        out, h = self.model.rnn(z, h)
        return self.model.out(out[:, 0]), h

    def initial_state(self) -> torch.Tensor:
        return torch.zeros(1, 1, self.model.rnn.hidden_size)
