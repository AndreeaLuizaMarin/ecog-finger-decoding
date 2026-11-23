"""Training of the neural decoders: random windows, early stopping on the validation block."""

from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from ..config import TrainConfig


@torch.no_grad()
def predict_sequence(model: nn.Module, x: np.ndarray) -> np.ndarray:
    """Run a causal model over one continuous feature sequence (n_steps, n_features)."""
    model.eval()
    return model(torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))[None])[0].numpy()


def correlation_loss(pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
    """1 - mean Pearson r over outputs, with all windows of the batch pooled per output."""
    p = pred.reshape(-1, pred.shape[-1])
    t = true.reshape(-1, true.shape[-1])
    p, t = p - p.mean(0), t - t.mean(0)
    r = (p * t).sum(0) / (p.norm(dim=0) * t.norm(dim=0) + 1e-6)
    return 1.0 - r.mean()


def train_sequence_model(
    model: nn.Module,
    x: np.ndarray,
    y: np.ndarray,
    fit_rows: slice,
    val_rows: slice | None,
    score_fn,
    cfg: TrainConfig,
    epochs: int | None = None,
    n_per_channel: int | None = None,
) -> list[dict]:
    """Train on random windows drawn from ``fit_rows``.

    With ``val_rows``: early stopping. After every epoch the model runs causally over the
    validation block (preceded by ``cfg.val_context`` warm-up steps, as it would online); the best
    weights are restored at the end.
    With ``val_rows=None``: train for exactly ``epochs`` epochs and keep the final weights (used to
    retrain on the whole training segment with the epoch count chosen by early stopping).
    Options from ``cfg``: ``corr_weight`` adds (1 - Pearson r) to the MSE loss; ``channel_dropout``
    silences each electrode (all its ``n_per_channel`` features) with that probability in every
    training window, which trains the model to cope with failing electrodes; ``weight_ema`` keeps
    an exponential moving average of the weights and uses it for validation and as the final model;
    ``gain_jitter``, ``feature_noise`` and ``time_warp`` are data augmentation (see ``TrainConfig``);
    ``schedule="cosine"`` trains for exactly ``max_epochs`` with a cosine-decaying learning rate and keeps
    the final weights, so no epoch count is selected on the (short, noisy) validation block.
    """
    rng = np.random.default_rng(cfg.seed)
    torch.manual_seed(cfg.seed)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    xt, yt = torch.from_numpy(x), torch.from_numpy(y)
    max_len = int(np.ceil(cfg.seq_len * (1 + cfg.time_warp)))
    lo, hi = fit_rows.start, fit_rows.stop - max_len
    offsets = np.arange(cfg.seq_len)
    cosine = cfg.schedule == "cosine"
    n_epochs = cfg.max_epochs if (val_rows is not None or cosine) else epochs
    steps_per_epoch = -(-cfg.windows_per_epoch // cfg.batch_size)
    sched = (
        torch.optim.lr_scheduler.LambdaLR(
            opt, lambda k: 0.5 * (1 + np.cos(np.pi * min(k / (n_epochs * steps_per_epoch), 1.0)))
        )
        if cosine
        else None
    )

    ema = copy.deepcopy(model) if cfg.weight_ema > 0 else None
    evaluated = ema if ema is not None else model
    history, best_score, best_state, best_epoch = [], -np.inf, None, 0
    for epoch in range(n_epochs):
        model.train()
        starts = rng.integers(lo, hi, size=cfg.windows_per_epoch)
        losses = []
        for b in range(0, len(starts), cfg.batch_size):
            if cfg.time_warp > 0:  # a window of another length, resampled to seq_len steps
                factor = np.exp(rng.uniform(-np.log1p(cfg.time_warp), np.log1p(cfg.time_warp)))
                idx = torch.from_numpy(starts[b : b + cfg.batch_size, None] + np.arange(round(cfg.seq_len * factor)))
                xb, yb = (
                    F.interpolate(
                        t[idx].transpose(1, 2), size=cfg.seq_len, mode="linear", align_corners=True
                    ).transpose(1, 2)
                    for t in (xt, yt)
                )
                yb = yb[:, cfg.burn_in :]
            else:
                idx = torch.from_numpy(starts[b : b + cfg.batch_size, None] + offsets)
                xb, yb = xt[idx], yt[idx][:, cfg.burn_in :]
            if cfg.channel_dropout > 0 and n_per_channel:
                keep = torch.rand(xb.shape[0], 1, xb.shape[-1] // n_per_channel) >= cfg.channel_dropout
                xb = xb * keep.repeat_interleave(n_per_channel, dim=-1)
            if cfg.gain_jitter > 0 and n_per_channel:
                shift = cfg.gain_jitter * torch.randn(xb.shape[0], 1, xb.shape[-1] // n_per_channel)
                xb = xb + shift.repeat_interleave(n_per_channel, dim=-1)
            if cfg.feature_noise > 0:
                xb = xb + cfg.feature_noise * torch.randn_like(xb)
            pred = model(xb)[:, cfg.burn_in :]
            loss = F.mse_loss(pred, yb)
            if cfg.corr_weight > 0:
                loss = loss + cfg.corr_weight * correlation_loss(pred, yb)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if sched is not None:
                sched.step()
            if ema is not None:
                with torch.no_grad():
                    for p_ema, p in zip(ema.parameters(), model.parameters()):
                        p_ema.mul_(cfg.weight_ema).add_(p, alpha=1 - cfg.weight_ema)
            losses.append(loss.item())
        record = {"epoch": epoch, "train_loss": float(np.mean(losses))}
        if val_rows is not None and (not cosine or epoch == n_epochs - 1):
            ctx_start = max(val_rows.start - cfg.val_context, 0)
            val_pred = predict_sequence(evaluated, x[ctx_start : val_rows.stop])[val_rows.start - ctx_start :]
            record["val_score"] = float(score_fn(val_pred, y[val_rows]))
            if record["val_score"] > best_score:
                best_score, best_state, best_epoch = record["val_score"], copy.deepcopy(evaluated.state_dict()), epoch
            elif epoch - best_epoch >= cfg.patience and not cosine:
                history.append(record)
                break
        history.append(record)
    if val_rows is not None and not cosine:
        model.load_state_dict(best_state)
    elif ema is not None:
        model.load_state_dict(ema.state_dict())
    return history
