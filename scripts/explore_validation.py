"""Compare preprocessing and training options on the VALIDATION block only.

The test labels are never loaded here, so nothing in this study can leak test information.
Each configuration trains one model per subject and seed on the fit rows, early-stops on the
validation block and reports the validation score (mean over seeds), without and with the causal
output smoothing chosen on that same block.

Usage:
    python scripts/explore_validation.py --data-dir data --out results/validation_study.json
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from ecogdecode.approaches import run_network, run_pretrained
from ecogdecode.config import FINE_HG_FEATURES, FeatureConfig, Protocol, TrainConfig
from ecogdecode.data import SUBJECTS, load_subject
from ecogdecode.models import LSTMDecoder
from ecogdecode.pipeline import keep_electrodes, prepare, top_electrodes

FEATURES = {"base": FeatureConfig(), "fine_hg": FINE_HG_FEATURES, "win200": replace(FeatureConfig(), window_ms=200)}
CUSTOM = {  # LSTM variants built directly (trained like approach "lstm128", the untuned 128-unit reference)
    "lstm_h64": lambda n: LSTMDecoder(n, hidden=64),
    "lstm_h256": lambda n: LSTMDecoder(n, hidden=256),
    "lstm_2layer": lambda n: LSTMDecoder(n, layers=2),
    "lstm_drop04": lambda n: LSTMDecoder(n, dropout=0.4),
}
ARCH_OPTIONS = ("hidden", "layers", "dropout")  # options that change the LSTM itself
TUNED = {"hidden": 256, "dropout": 0.4, "gain_jitter": 0.3, "feature_noise": 0.3}  # approach 5 as chosen in round 4
CONFIGS = {  # name: (approach, feature set, training options); "adaptive_norm_s" is a protocol option
    "lstm_adapt60": ("lstm128", "base", {"adaptive_norm_s": 60.0}),
    "lstm_adapt180": ("lstm128", "base", {"adaptive_norm_s": 180.0}),
    "lstm_gain03": ("lstm128", "base", {"gain_jitter": 0.3}),
    "lstm_noise03": ("lstm128", "base", {"feature_noise": 0.3}),
    "lstm_aug": ("lstm128", "base", {"gain_jitter": 0.3, "feature_noise": 0.3}),
    "lstm_cos40": ("lstm128", "base", {"schedule": "cosine", "max_epochs": 40}),
    "lstm_cos80": ("lstm128", "base", {"schedule": "cosine", "max_epochs": 80}),
    "lstm_cos160": ("lstm128", "base", {"schedule": "cosine", "max_epochs": 160}),
    "lstm_win200": ("lstm128", "win200", {}),
    "h256_drop04": ("lstm", "base", {"hidden": 256, "dropout": 0.4}),
    "h384_drop04": ("lstm", "base", {"hidden": 384, "dropout": 0.4}),
    "h256_drop04_cos80": ("lstm", "base", {"hidden": 256, "dropout": 0.4, "schedule": "cosine", "max_epochs": 80}),
    "h256_drop04_aug": ("lstm", "base", {"hidden": 256, "dropout": 0.4, "gain_jitter": 0.3, "feature_noise": 0.3}),
    "h256_drop04_adapt180": ("lstm", "base", {"hidden": 256, "dropout": 0.4, "adaptive_norm_s": 180.0}),
    "h256_drop04_win200": ("lstm", "win200", {"hidden": 256, "dropout": 0.4}),
    "h256_drop04_aug_cos40": (
        "lstm",
        "base",
        {
            "hidden": 256,
            "dropout": 0.4,
            "gain_jitter": 0.3,
            "feature_noise": 0.3,
            "schedule": "cosine",
            "max_epochs": 40,
        },
    ),
    "h128_drop04_aug": ("lstm", "base", {"hidden": 128, "dropout": 0.4, "gain_jitter": 0.3, "feature_noise": 0.3}),
    "h256_drop04_aug05": ("lstm", "base", {"hidden": 256, "dropout": 0.4, "gain_jitter": 0.5, "feature_noise": 0.5}),
    # round 5: training settings of the tuned LSTM (approach 5), one at a time
    "t_lr5e-4": ("lstm", "base", {**TUNED, "lr": 5e-4, "max_epochs": 120, "patience": 20}),
    "t_lr2e-3": ("lstm", "base", {**TUNED, "lr": 2e-3}),
    "t_wd1e-1": ("lstm", "base", {**TUNED, "weight_decay": 0.1}),
    "t_seq512": ("lstm", "base", {**TUNED, "seq_len": 512}),
    "t_wpe256": ("lstm", "base", {**TUNED, "windows_per_epoch": 256}),
    "t_drop05": ("lstm", "base", {**TUNED, "dropout": 0.5}),
    "t_finehg": ("lstm", "fine_hg", {**TUNED}),
    "t_batch32": ("lstm", "base", {**TUNED, "batch_size": 32, "windows_per_epoch": 256}),
    "t_warp02": ("lstm", "base", {**TUNED, "time_warp": 0.2}),
    "t_top32": ("lstm", "base", {**TUNED, "top_electrodes": 32}),
    "t_top16": ("lstm", "base", {**TUNED, "top_electrodes": 16}),
    "t_top8": ("lstm", "base", {**TUNED, "top_electrodes": 8}),
    "t_top24": ("lstm", "base", {**TUNED, "top_electrodes": 24}),
    "t_top48": ("lstm", "base", {**TUNED, "top_electrodes": 48}),
    "t_top4": ("lstm", "base", {**TUNED, "top_electrodes": 4}),
    "t_top8_warp": ("lstm", "base", {**TUNED, "top_electrodes": 8, "time_warp": 0.2}),
    "t_top32_warp": ("lstm", "base", {**TUNED, "top_electrodes": 32, "time_warp": 0.2}),
    "t_top16_cd": ("lstm", "base", {**TUNED, "top_electrodes": 16, "channel_dropout": 0.1}),
    "t_top32_cd": ("lstm", "base", {**TUNED, "top_electrodes": 32, "channel_dropout": 0.1}),
    "manifold_pca_lstm256_aug": ("manifold_pca", "base", {"gain_jitter": 0.3, "feature_noise": 0.3}),
    "manifold_contrastive_lstm256_aug": ("manifold_contrastive", "base", {"gain_jitter": 0.3, "feature_noise": 0.3}),
    "tcn": ("tcn", "base", {}),
    "pretrained_contrastive": ("pretrained_lstm", "base", {}),
    "pretrained_multipatient": ("multipatient_lstm", "base", {}),
    "lstm_h64": ("lstm128", "base", {}),
    "lstm_h256": ("lstm128", "base", {}),
    "lstm_2layer": ("lstm128", "base", {}),
    "lstm_drop04": ("lstm128", "base", {}),
    "lstm_ema": ("lstm128", "base", {"weight_ema": 0.99}),
    "manifold_pca_mse": ("manifold_pca_leaky", "base", {}),
    "manifold_contrastive_mse": ("manifold_contrastive_leaky", "base", {}),
    "manifold_pca_lstm": ("manifold_pca_lstm", "base", {}),
    "manifold_contrastive_lstm": ("manifold_contrastive_lstm", "base", {}),
    "lstm_mse": ("lstm128", "base", {}),
    "lstm_corr": ("lstm128", "base", {"corr_weight": 1.0}),
    "lstm_dropout": ("lstm128", "base", {"channel_dropout": 0.1}),
    "lstm_corr_dropout": ("lstm128", "base", {"corr_weight": 1.0, "channel_dropout": 0.1}),
    "lstm_corr_dropout_finehg": ("lstm128", "fine_hg", {"corr_weight": 1.0, "channel_dropout": 0.1}),
    "manifold_pca": ("manifold_pca_leaky", "base", {"corr_weight": 1.0, "channel_dropout": 0.1}),
    "manifold_contrastive": ("manifold_contrastive_leaky", "base", {"corr_weight": 1.0, "channel_dropout": 0.1}),
}


DESCRIPTIONS = {
    "lstm_mse": "LSTM, 128 units (reference)",
    "lstm_h64": "LSTM, 64 units",
    "lstm_h256": "LSTM, 256 units",
    "lstm_2layer": "LSTM, two layers",
    "lstm_drop04": "LSTM, input dropout 0.4",
    "lstm_corr": "LSTM, + correlation loss",
    "lstm_dropout": "LSTM, + electrode dropout",
    "lstm_ema": "LSTM, + weight averaging (EMA)",
    "lstm_adapt60": "LSTM, + online re-normalisation (60 s)",
    "lstm_adapt180": "LSTM, + online re-normalisation (180 s)",
    "lstm_gain03": "LSTM, + electrode gain jitter",
    "lstm_noise03": "LSTM, + feature noise",
    "lstm_aug": "LSTM, + gain jitter and feature noise",
    "lstm_cos40": "LSTM, cosine schedule, 40 epochs, no early stopping",
    "lstm_cos80": "LSTM, cosine schedule, 80 epochs, no early stopping",
    "lstm_cos160": "LSTM, cosine schedule, 160 epochs, no early stopping",
    "lstm_win200": "LSTM, 200 ms feature window",
    "h256_drop04": "LSTM, 256 units, dropout 0.4",
    "h384_drop04": "LSTM, 384 units, dropout 0.4",
    "h256_drop04_cos80": "LSTM 256/0.4, cosine schedule, 80 epochs, no early stopping",
    "h256_drop04_aug": "LSTM 256/0.4, + gain jitter and feature noise",
    "h256_drop04_adapt180": "LSTM 256/0.4, + online re-normalisation (180 s)",
    "h256_drop04_win200": "LSTM 256/0.4, 200 ms feature window",
    "tcn": "Causal TCN",
    "pretrained_contrastive": "Contrastive cross-patient pretraining + LSTM",
    "pretrained_multipatient": "Supervised multi-patient pretraining + LSTM",
    "manifold_pca_mse": "Manifold-gated, PCA manifold, leaky-integrator experts",
    "manifold_pca_lstm": "Manifold-gated, PCA manifold, 128-unit LSTM",
    "manifold_contrastive_lstm": "Manifold-gated, CEBRA-style manifold, 128-unit LSTM",
    "h128_drop04_aug": "LSTM 128/0.4, + gain jitter and feature noise (no gate; control for approaches 12-13)",
    "t_lr5e-4": "Approach 5, learning rate 5e-4 (up to 120 epochs)",
    "t_lr2e-3": "Approach 5, learning rate 2e-3",
    "t_wd1e-1": "Approach 5, weight decay 0.1",
    "t_seq512": "Approach 5, 20 s training windows",
    "t_wpe256": "Approach 5, twice the steps per epoch",
    "t_drop05": "Approach 5, dropout 0.5",
    "t_finehg": "Approach 5, finer high-gamma bands",
    "t_batch32": "Approach 5, batch 32",
    "t_warp02": "Approach 5, + time-warp augmentation (±20 % speed)",
    "t_top32": "Approach 5, 32 most movement-related electrodes only",
    "t_top16": "Approach 5, 16 most movement-related electrodes only",
    "t_top8": "Approach 5, 8 most movement-related electrodes only",
    "t_top24": "Approach 5, 24 most movement-related electrodes only",
    "t_top48": "Approach 5, 48 most movement-related electrodes only",
    "t_top4": "Approach 5, 4 most movement-related electrodes only",
    "t_top8_warp": "Approach 5, 8 electrodes, + time warp",
    "t_top32_warp": "Approach 5, 32 electrodes, + time warp",
    "t_top16_cd": "Approach 5, 16 electrodes, + electrode dropout 0.1",
    "t_top32_cd": "Approach 5, 32 electrodes, + electrode dropout 0.1",
    "h256_drop04_aug05": "LSTM 256/0.4, + stronger gain jitter and feature noise (0.5)",
    "h256_drop04_aug_cos40": "LSTM 256/0.4 + augmentation, cosine schedule, 40 epochs, no early stopping",
    "manifold_pca_lstm256_aug": "Manifold-gated, PCA manifold, 128-unit LSTM, dropout 0.4 + augmentation",
    "manifold_contrastive_lstm256_aug": "Manifold-gated, CEBRA-style manifold, 128-unit LSTM, "
    "dropout 0.4 + augmentation",
}


def table(paths: list[Path]) -> str:
    """Markdown table of every 3-seed configuration in the given study files, best first."""
    rows = {}
    for p in paths:
        for name, r in json.loads(p.read_text()).items():
            if len(r.get("seeds", [])) >= 3:
                rows[name] = r
    lines = ["| Configuration | Validation r | S1 | S2 | S3 |", "|---|---|---|---|---|"]
    for name, r in sorted(rows.items(), key=lambda kv: -kv[1]["mean_val"]):
        per = " | ".join(f"{r['subjects'][s]['val']:.2f}" for s in sorted(r["subjects"], key=int))
        lines.append(f"| {DESCRIPTIONS.get(name, name)} | {r['mean_val']:.3f} | {per} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out", default="results/validation_study.json")
    ap.add_argument("--configs", nargs="+", default=list(CONFIGS), choices=list(CONFIGS))
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--table", nargs="+", type=Path, help="only print a markdown table of these study files")
    args = ap.parse_args()
    if args.table:
        print(table(args.table))
        return
    torch.set_num_threads(args.threads)
    protocol = Protocol(refit=False)

    datas = {}
    for s in SUBJECTS:
        d = load_subject(args.data_dir, s, require_test_labels=False)
        d.test_flex = None  # the test labels are not used anywhere in this study
        datas[s] = d
    preps = {}
    out_path = Path(args.out)
    results = json.loads(out_path.read_text()) if out_path.exists() else {}
    for name in args.configs:
        key, feats, opts = CONFIGS[name]
        t0, rows = time.time(), {}
        for s, d in datas.items():
            if (s, feats) not in preps:
                preps[(s, feats)] = prepare(d, FEATURES[feats], protocol)
        proto = replace(protocol, adaptive_norm_s=opts.get("adaptive_norm_s", 0.0))
        arch = {k: opts[k] for k in ARCH_OPTIONS if k in opts}
        study_only = ("adaptive_norm_s", "top_electrodes")
        cfg = replace(TrainConfig(), **{k: v for k, v in opts.items() if k not in study_only and k not in arch})
        per_subject = {s: [] for s in datas}
        for seed in args.seeds:
            if key in ("pretrained_lstm", "multipatient_lstm"):
                out = run_pretrained(
                    {s: preps[(s, feats)] for s in datas},
                    seed,
                    proto,
                    train_cfg=cfg,
                    supervised=key == "multipatient_lstm",
                )
                for s, r in out.items():
                    per_subject[s].append(r)
            else:
                for s in datas:
                    prep = preps[(s, feats)]
                    if "top_electrodes" in opts:  # chosen on the fit rows only
                        prep = keep_electrodes(prep, top_electrodes(prep, prep.fit_rows, opts["top_electrodes"]))
                    model = (
                        CUSTOM[name](prep.x.shape[1])
                        if name in CUSTOM
                        else LSTMDecoder(prep.x.shape[1], **arch)
                        if arch
                        else None
                    )
                    per_subject[s].append(run_network(prep, key, seed=seed, protocol=proto, model=model, train_cfg=cfg))
        for s, runs in per_subject.items():
            rows[str(s)] = {
                "val_unsmoothed": float(np.mean([r.selected["val_score_unsmoothed"] for r in runs])),
                "val": float(np.mean([r.val_score for r in runs])),
                "val_per_seed": [r.val_score for r in runs],
                "smooth_ms": float(np.median([r.selected["smooth_ms"] for r in runs])),
                "epochs": [r.selected["epochs"] for r in runs],
            }
        results[name] = {
            "seeds": args.seeds,
            "approach": key,
            "features": feats,
            "options": opts,
            "subjects": rows,
            "mean_val_unsmoothed": float(np.mean([r["val_unsmoothed"] for r in rows.values()])),
            "mean_val": float(np.mean([r["val"] for r in rows.values()])),
            "seconds": round(time.time() - t0, 1),
        }
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(results, indent=2))
        r = results[name]
        per = "  ".join(f"S{s} {v['val']:.3f} ({v['smooth_ms']:.0f} ms)" for s, v in rows.items())
        print(
            f"{name:<28} val {r['mean_val_unsmoothed']:.3f} -> smoothed {r['mean_val']:.3f}   {per}   {r['seconds']} s",
            flush=True,
        )


if __name__ == "__main__":
    main()
