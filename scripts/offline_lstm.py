"""Offline versus live for the best live decoder: approach 5b given every offline liberty (offline only).

The 2008 winner was designed offline and loses about 0.09 when it has to run causally (0.45 -> 0.36). This asks the
reverse question for the best live decoder here: what does approach 5b gain when it may use the future?
It is retrained with
* zero-phase filters and feature windows centred on each step (``FeatureConfig(offline=True)``),
* a bidirectional LSTM run over the whole recording, so every output also sees the signal after it,
* a centred Gaussian output smoother, sigma chosen on validation from 0, 40, 80 and 120 ms,
* linear interpolation between the 40 ms outputs when scored at 1 kHz,
with the same electrode counts as 5b (32, 8, 32), every choice on validation, a refit on the whole training segment
and three seeds. Together with the winner's offline replica and its live redesign this gives a two-by-two table:
(method designed offline, method designed live) x (scored offline, scored live). Nothing feeds back into a live
decoder.

Usage:
    python scripts/offline_lstm.py --data-dir data
"""

from __future__ import annotations

import argparse
import copy
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d
from torch import nn

from ecogdecode.approaches import ELECTRODES_5B, TUNED_LSTM
from ecogdecode.config import FeatureConfig
from ecogdecode.data import SUBJECTS, load_subject
from ecogdecode.evaluation import score
from ecogdecode.models import predict_sequence, train_sequence_model
from ecogdecode.offline import score_1khz, upsample_linear
from ecogdecode.pipeline import keep_electrodes, prepare, standardise, top_electrodes
from ecogdecode.provenance import environment, record_test_look

SIGMAS_MS = (0.0, 40.0, 80.0, 120.0)
OUT = Path("results/offline")


class BiLSTMDecoder(nn.Module):
    """Approach 5's network with a bidirectional LSTM (256 units each way): offline only."""

    def __init__(self, n_features: int, hidden: int = 256, dropout: float = 0.4):
        super().__init__()
        self.inp = nn.Sequential(nn.Dropout(dropout), nn.Linear(n_features, hidden), nn.GELU())
        self.rnn = nn.LSTM(hidden, hidden, batch_first=True, bidirectional=True)
        self.out = nn.Linear(2 * hidden, 5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self.rnn(self.inp(x))
        return self.out(h)


def smooth(pred: np.ndarray, sigma_ms: float) -> np.ndarray:
    return gaussian_filter1d(pred, sigma_ms / 40.0, axis=0) if sigma_ms > 0 else pred


def run(data, seed: int) -> dict:
    full = prepare(data, FeatureConfig(offline=True))
    k = ELECTRODES_5B[data.subject]
    cfg = replace(TUNED_LSTM, seed=seed)
    prep = keep_electrodes(full, top_electrodes(full, full.fit_rows, k))
    npc = prep.cfg.n_per_channel
    torch.manual_seed(seed)
    sc = standardise(prep, prep.fit_rows)
    model = BiLSTMDecoder(sc.x.shape[1])
    initial = copy.deepcopy(model)
    history = train_sequence_model(model, sc.x, sc.y, prep.fit_rows, prep.val_rows, score, cfg, n_per_channel=npc)
    best = max(history, key=lambda h: h.get("val_score", -np.inf))
    val_pred = predict_sequence(model, sc.x[: prep.val_rows.stop])[prep.val_rows]
    val_by_sigma = {s: float(score(smooth(val_pred, s), sc.y[prep.val_rows])) for s in SIGMAS_MS}
    sigma = max(val_by_sigma, key=val_by_sigma.get)

    prep = keep_electrodes(full, top_electrodes(full, full.train_rows, k))
    sc = standardise(prep, prep.train_rows)
    model = initial
    train_sequence_model(
        model, sc.x, sc.y, prep.train_rows, None, score, cfg, epochs=best["epoch"] + 1, n_per_channel=npc
    )
    pred = sc.y_scaler.inverse_transform(smooth(predict_sequence(model, sc.x), sigma))
    t = prep.ends[prep.test_rows] - prep.n_train_samples
    pred_1khz = upsample_linear(pred[prep.test_rows], t, len(data.test_flex))
    total, per = score_1khz(pred_1khz, data.test_flex)
    return {
        "seed": seed,
        "epochs": best["epoch"] + 1,
        "sigma_ms": sigma,
        "val_score": val_by_sigma[sigma],
        "score": total,
        "r": per,
        "pred_25hz": pred_1khz[::40].astype(np.float32),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--threads", type=int, default=1)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    record_test_look("offline 5b for the live/offline 2x2 (offline_lstm.py)")
    report: dict = {"environment": environment(args.threads), "subjects": {}}
    for s in SUBJECTS:
        data = load_subject(args.data_dir, s)
        runs = []
        for seed in args.seeds:
            res = run(data, seed)
            if seed == 0:
                (OUT / "preds").mkdir(parents=True, exist_ok=True)
                np.save(OUT / "preds" / f"lstm_selected_offline_sub{s}.npy", res["pred_25hz"])
            res.pop("pred_25hz")
            runs.append(res)
            print(
                f"S{s} seed {seed}: offline 5b {res['score']:.3f} (sigma {res['sigma_ms']:.0f} ms, "
                f"{res['epochs']} epochs)",
                flush=True,
            )
        report["subjects"][str(s)] = {
            "score": float(np.mean([r["score"] for r in runs])),
            "sd": float(np.std([r["score"] for r in runs])),
            "runs": runs,
        }
    seed_means = [
        np.mean([report["subjects"][str(s)]["runs"][i]["score"] for s in SUBJECTS]) for i in range(len(args.seeds))
    ]
    report["score"] = float(np.mean(seed_means))
    report["sd"] = float(np.std(seed_means))
    (OUT / "lstm_offline.json").write_text(json.dumps(report, indent=2))
    print(f"offline 5b: {report['score']:.3f} ± {report['sd']:.3f}")


if __name__ == "__main__":
    main()
