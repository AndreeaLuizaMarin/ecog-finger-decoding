"""Self-supervised contrastive pretraining across subjects, then subject-specific fine-tuning.

Idea (in the spirit of CEBRA's time-contrastive learning): without any finger labels, train
a shared recurrent encoder so that neural states a few tens of milliseconds apart map to
nearby points in an embedding, and states far apart in time map apart (InfoNCE loss).
Each subject has its own input layer (electrode count and placement differ between
patients); the recurrent core is shared and so sees the data of all subjects. The
pretrained network is then fine-tuned on each subject's labelled training data exactly
like the LSTM trained from scratch, so the comparison isolates the effect of pretraining.

Only the training segments are used for pretraining; the test segments are never seen.
"""

from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .models.possm import POSSMDecoder
from .models.recurrent import LSTMDecoder


def info_nce_time(
    emb: torch.Tensor, gen: torch.Generator, n_anchors: int = 32, max_offset: int = 5, temperature: float = 0.1
) -> torch.Tensor:
    """InfoNCE with temporal positives: the positive for (window b, time t) is (b, t + offset).

    All other positives in the batch (other windows and other times) act as negatives.
    ``emb`` is (batch, time, dim) and L2-normalised.
    """
    b, length, _ = emb.shape
    t = torch.randint(0, length - max_offset, (b, n_anchors), generator=gen)
    offset = torch.randint(1, max_offset + 1, (b, n_anchors), generator=gen)
    rows = torch.arange(b)[:, None]
    anchors = emb[rows, t].reshape(b * n_anchors, -1)
    positives = emb[rows, t + offset].reshape(b * n_anchors, -1)
    logits = anchors @ positives.T / temperature
    return F.cross_entropy(logits, torch.arange(len(logits)))


def pretrain_contrastive(
    xs: dict[int, np.ndarray],
    fit_rows: dict[int, slice],
    seed: int = 0,
    steps: int = 600,
    seq_len: int = 128,
    batch_size: int = 16,
    hidden: int = 128,
    lr: float = 1e-3,
) -> tuple[dict[int, LSTMDecoder], list[float]]:
    """Pretrain one LSTM core shared by all subjects. Returns per-subject decoders (sharing the core)."""
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    rng = np.random.default_rng(seed)
    decoders = {s: LSTMDecoder(x.shape[1], hidden=hidden) for s, x in xs.items()}
    shared = next(iter(decoders.values())).rnn
    for dec in decoders.values():
        dec.rnn = shared  # one recurrent core for every subject
    head = nn.Sequential(nn.Linear(hidden, 64), nn.GELU(), nn.Linear(64, 32))
    params = {id(p): p for d in decoders.values() for p in d.parameters()}
    params.update({id(p): p for p in head.parameters()})
    opt = torch.optim.AdamW(params.values(), lr=lr, weight_decay=1e-2)
    tensors = {s: torch.from_numpy(x) for s, x in xs.items()}
    offsets = np.arange(seq_len)

    losses = []
    for _ in range(steps):
        loss = 0.0
        for s, dec in decoders.items():
            dec.train()
            starts = rng.integers(fit_rows[s].start, fit_rows[s].stop - seq_len, size=batch_size)
            xb = tensors[s][torch.from_numpy(starts[:, None] + offsets)]
            h, _ = dec.rnn(dec.inp(xb))
            loss = loss + info_nce_time(F.normalize(head(h), dim=-1), gen)
        loss = loss / len(decoders)
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(list(params.values()), 1.0)
        opt.step()
        losses.append(loss.item())
    # give every subject its own copy of the pretrained core before fine-tuning
    return {s: copy.deepcopy(dec) for s, dec in decoders.items()}, losses


def pretrain_supervised(
    xs: dict[int, np.ndarray],
    ys: dict[int, np.ndarray],
    fit_rows: dict[int, slice],
    seed: int = 0,
    steps: int = 600,
    seq_len: int = 256,
    burn_in: int = 50,
    batch_size: int = 16,
    hidden: int = 128,
    lr: float = 1e-3,
) -> dict[int, LSTMDecoder]:
    """Multi-patient training: one LSTM core shared by all subjects, each with its own input and
    output layers, trained on every subject's labelled fit rows. Returns per-subject decoders (each
    with its own copy of the shared core) to be fine-tuned per subject."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    decoders = {s: LSTMDecoder(x.shape[1], hidden=hidden) for s, x in xs.items()}
    shared = next(iter(decoders.values())).rnn
    for dec in decoders.values():
        dec.rnn = shared
    params = {id(p): p for d in decoders.values() for p in d.parameters()}
    opt = torch.optim.AdamW(params.values(), lr=lr, weight_decay=1e-2)
    xt = {s: torch.from_numpy(x) for s, x in xs.items()}
    yt = {s: torch.from_numpy(y) for s, y in ys.items()}
    offsets = np.arange(seq_len)
    for _ in range(steps):
        loss = 0.0
        for s, dec in decoders.items():
            dec.train()
            starts = rng.integers(fit_rows[s].start, fit_rows[s].stop - seq_len, size=batch_size)
            idx = torch.from_numpy(starts[:, None] + offsets)
            loss = loss + F.mse_loss(dec(xt[s][idx])[:, burn_in:], yt[s][idx][:, burn_in:])
        loss = loss / len(decoders)
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(list(params.values()), 1.0)
        opt.step()
    return {s: copy.deepcopy(dec) for s, dec in decoders.items()}


def pretrain_possm(
    xs: dict[int, np.ndarray],
    ys: dict[int, np.ndarray],
    fit_rows: dict[int, slice],
    n_per_channel: int,
    seed: int = 0,
    steps: int = 600,
    seq_len: int = 256,
    burn_in: int = 50,
    batch_size: int = 16,
    lr: float = 1e-3,
    gain_jitter: float = 0.3,
    feature_noise: float = 0.3,
) -> dict[int, POSSMDecoder]:
    """Multi-patient training of the POSSM-style decoder: every weight is shared except the electrode embeddings,
    which are per patient (electrode count and placement differ). Returns per-subject decoders, each a copy of the
    shared model with that subject's embeddings, to be fine-tuned or adapted per subject."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    first = next(iter(xs))
    shared = POSSMDecoder(xs[first].shape[1], n_per_channel)
    d_model = shared.tokenizer.latents.shape[1]
    embeddings = {s: nn.Parameter(0.02 * torch.randn(x.shape[1] // n_per_channel, d_model)) for s, x in xs.items()}
    params = [*shared.parameters(), *embeddings.values()]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-2)
    xt = {s: torch.from_numpy(x) for s, x in xs.items()}
    yt = {s: torch.from_numpy(y) for s, y in ys.items()}
    offsets = np.arange(seq_len)
    for _ in range(steps):
        loss = 0.0
        shared.train()
        for s in xs:
            n_el = embeddings[s].shape[0]
            shared.tokenizer.n_channels = n_el
            shared.tokenizer.electrode = embeddings[s]
            starts = rng.integers(fit_rows[s].start, fit_rows[s].stop - seq_len, size=batch_size)
            idx = torch.from_numpy(starts[:, None] + offsets)
            xb = xt[s][idx]
            xb = xb + (gain_jitter * torch.randn(batch_size, 1, n_el)).repeat_interleave(n_per_channel, dim=-1)
            xb = xb + feature_noise * torch.randn_like(xb)
            loss = loss + F.mse_loss(shared(xb)[:, burn_in:], yt[s][idx][:, burn_in:])
        loss = loss / len(xs)
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
    out = {}
    for s, emb in embeddings.items():
        dec = copy.deepcopy(shared)
        dec.tokenizer.n_channels = emb.shape[0]
        dec.tokenizer.electrode = nn.Parameter(emb.detach().clone())
        out[s] = dec
    return out
