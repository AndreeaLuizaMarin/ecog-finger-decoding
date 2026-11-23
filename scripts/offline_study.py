"""Offline versus live: what does a decoder gain from what a live system cannot have?

Parts, each saved to results/offline/:

1. replicas  - the two best competition entries as published (offline: zero-phase filters, windows that reach
               ahead, future time shifts), next to the same winner made causal, and to the live versions in the
               benchmark (approaches 1 and 2).
2. wiener    - one model, three conditions: causal features and past lags (live); zero-phase, centred
               features and past lags; the same plus future lags. Lags and penalty chosen on validation.
3. selection - choosing the stopping epoch on the test set instead of on validation (as a checkpoint
               callback that monitors the test set does), for the best live decoder (approach 5b).
4. smoothing - a centred (non-causal) Gaussian smoother on the live decoder's output, as used for
               offline results, versus the causal smoother chosen on validation.
5. fingerflex - FingerFlex (Lomtev et al., 2023) as published, with the epoch chosen on the test set (as
               published) or on validation; one file per subject and seed, then merged.

Usage:
    python scripts/offline_study.py --data-dir data --parts liang flamary wiener selection smoothing
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d

from ecogdecode.approaches import ELECTRODES_5B, TUNED_LSTM
from ecogdecode.config import DEFAULT_FEATURES
from ecogdecode.data import SUBJECTS, load_subject
from ecogdecode.evaluation import score
from ecogdecode.models import LSTMDecoder, train_sequence_model
from ecogdecode.models.linear import lag_matrix
from ecogdecode.offline import (
    run_fingerflex_offline,
    run_flamary_offline,
    run_liang_offline,
    score_1khz,
    upsample_hold,
    upsample_linear,
)
from ecogdecode.pipeline import keep_electrodes, prepare, standardise, top_electrodes, with_test_labels
from ecogdecode.provenance import record_test_look

OUT = Path("results/offline")


def save(name: str, obj: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}.json").write_text(json.dumps(obj, indent=2))


def save_pred(name: str, s: int, pred_1khz: np.ndarray) -> None:
    """Predictions at 25 Hz (every 40th sample of the 1 kHz output) for figures and animations."""
    (OUT / "preds").mkdir(parents=True, exist_ok=True)
    np.save(OUT / "preds" / f"{name}_sub{s}.npy", pred_1khz[::40].astype(np.float32))


# --------------------------------------------------------------------------- 1. replicas
def _merge(name: str, key: str, s: int, row: dict) -> None:
    path = OUT / f"{name}.json"
    report = json.loads(path.read_text()) if path.exists() else {}
    report.setdefault(str(s), {})[key] = row
    subjects = [str(x) for x in SUBJECTS if str(x) in report]
    keys = set.intersection(*(set(report[x]) for x in subjects))
    report["mean"] = {k: float(np.mean([report[x][k]["score"] for x in subjects])) for k in sorted(keys)}
    save(name, report)


def part_liang(data_dir: str) -> None:
    for s in SUBJECTS:
        d = load_subject(data_dir, s)
        for name, kwargs in {
            "liang_offline": {},
            "liang_offline_intercept": {"intercept": True},
            "liang_causal": {"causal": True},
        }.items():
            out = run_liang_offline(d.train_ecog, d.train_flex, d.test_ecog, **kwargs)
            r, per = score_1khz(out["pred_1khz"], d.test_flex)
            _merge("replicas", name, s, {"score": r, "r_per_finger": per, "pairs": out["pairs"]})
            save_pred(name, s, out["pred_1khz"])
            print(f"S{s} {name:<24} r = {r:.3f}", flush=True)


def part_flamary(data_dir: str) -> None:
    for s in SUBJECTS:
        d = load_subject(data_dir, s)
        out = run_flamary_offline(
            d.train_ecog, d.train_flex, d.test_ecog, log=lambda m, s=s: print(f"S{s} {m}", flush=True)
        )
        r, per = score_1khz(out["pred_1khz"], d.test_flex)
        _merge("replicas", "flamary_offline", s, {"score": r, "r_per_finger": per, "chosen": out["chosen"]})
        save_pred("flamary_offline", s, out["pred_1khz"])
        print(f"S{s} flamary_offline          r = {r:.3f}", flush=True)


def part_diagnostics(data_dir: str) -> None:
    """Post hoc: subject 3's electrode 49 carries a burst of artefact about 130-150 s into the test segment
    (normal throughout training). Rerun both replicas without it to measure how much of their subject-3 score
    depends on it. Not part of any method; reported as a diagnostic."""
    d = load_subject(data_dir, 3)
    rows = {}
    for name, fn in (
        (
            "liang_offline_without_e49",
            lambda: run_liang_offline(d.train_ecog, d.train_flex, d.test_ecog, exclude=(49,)),
        ),
        (
            "flamary_offline_without_e49",
            lambda: run_flamary_offline(
                d.train_ecog, d.train_flex, d.test_ecog, log=lambda m: print(f"S3 {m}"), exclude=(49,)
            ),
        ),
    ):
        r, per = score_1khz(fn()["pred_1khz"], d.test_flex)
        rows[name] = {"score": r, "r_per_finger": per}
        print(f"S3 {name:<30} r = {r:.3f}", flush=True)
    save("diagnostics_sub3", rows)


# --------------------------------------------------------------------------- 2. wiener
def lead_lag_matrix(x: np.ndarray, past: int, future: int) -> np.ndarray:
    """x(t), x(t-1), ..., x(t-past+1) and x(t+1), ..., x(t+future)."""
    blocks = [lag_matrix(x, past)]
    if future:
        lead = np.zeros((len(x), x.shape[1] * future), dtype=x.dtype)
        for k in range(1, future + 1):
            lead[: len(x) - k, (k - 1) * x.shape[1] : k * x.shape[1]] = x[k:]
        blocks.append(lead)
    return np.hstack(blocks)


def ridge_path(design, y, fit, val, alphas):
    a, yy = design[fit].astype(np.float64), y[fit].astype(np.float64)
    xm, ym = a.mean(0), yy.mean(0)
    a -= xm
    evals, evecs = np.linalg.eigh(a.T @ a)
    proj = evecs.T @ (a.T @ (yy - ym))
    out = []
    for alpha in alphas:
        coef = evecs @ (proj / (evals + alpha)[:, None])
        out.append((alpha, coef, xm, ym, score((design[val] - xm) @ coef + ym, y[val])))
    return out


def part_wiener(data_dir: str) -> None:
    alphas = np.logspace(2, 6.5, 10)
    report = {}
    for s in SUBJECTS:
        d = load_subject(data_dir, s)
        rows = {}
        for name, cfg, past, future in (
            ("live: causal features, past lags", DEFAULT_FEATURES, 10, 0),
            ("zero-phase centred features, past lags", replace(DEFAULT_FEATURES, offline=True), 10, 0),
            ("zero-phase centred features, past and future lags", replace(DEFAULT_FEATURES, offline=True), 10, 10),
        ):
            prep = prepare(d, cfg)
            sc = standardise(prep, prep.fit_rows)
            design = lead_lag_matrix(sc.x, past, future)
            alpha = max(ridge_path(design, sc.y, prep.fit_rows, prep.val_rows, alphas), key=lambda t: t[4])[0]
            sc = standardise(prep, prep.train_rows)
            design = lead_lag_matrix(sc.x, past, future)
            _, coef, xm, ym, _ = ridge_path(design, sc.y, prep.train_rows, prep.train_rows, [alpha])[0]
            pred = sc.y_scaler.inverse_transform((design - xm) @ coef + ym)[prep.test_rows]
            t = prep.ends[prep.test_rows] - prep.n_train_samples
            up = upsample_hold if not cfg.offline else upsample_linear
            r, per = score_1khz(up(pred, t, len(d.test_ecog)), d.test_flex)
            rows[name] = {"score": r, "r_per_finger": per, "alpha": float(alpha), "past": past, "future": future}
            print(f"S{s} {name:<52} r = {r:.3f}", flush=True)
        report[str(s)] = rows
    report["mean"] = {k: float(np.mean([report[str(s)][k]["score"] for s in SUBJECTS])) for k in report["1"]}
    save("wiener", report)


# --------------------------------------------------------------------------- 3. selection
def part_selection(data_dir: str, seeds=(0, 1, 2), max_epochs: int = 60) -> None:
    runs = json.loads(Path("results/runs/lstm_selected.json").read_text())
    report = {}
    for s in SUBJECTS:
        d = load_subject(data_dir, s)
        full = with_test_labels(prepare(d), d.test_flex)  # the test block is monitored after every epoch
        prep = keep_electrodes(full, top_electrodes(full, full.train_rows, ELECTRODES_5B[s]))
        sc = standardise(prep, prep.train_rows)
        rows = []
        for seed in seeds:
            torch.manual_seed(seed)
            model = LSTMDecoder(sc.x.shape[1], hidden=256, dropout=0.4)
            cfg = replace(TUNED_LSTM, seed=seed, max_epochs=max_epochs, patience=10**6)
            history = train_sequence_model(
                model, sc.x, sc.y, prep.train_rows, prep.test_rows, score, cfg, n_per_channel=prep.cfg.n_per_channel
            )
            curve = [h["val_score"] for h in history]  # here: score on the TEST block after every epoch
            chosen = runs["subjects"][str(s)]["runs"][seeds.index(seed)]["selected"]["epochs"]
            rows.append(
                {
                    "seed": seed,
                    "test_curve": curve,
                    "epochs_chosen_on_validation": chosen,
                    "at_validation_choice": curve[chosen - 1],
                    "best_on_test": max(curve),
                    "best_epoch_on_test": int(np.argmax(curve)) + 1,
                }
            )
            print(
                f"S{s} seed {seed}: validation-chosen epoch {chosen} -> {curve[chosen - 1]:.3f}; "
                f"best test epoch {int(np.argmax(curve)) + 1} -> {max(curve):.3f}",
                flush=True,
            )
        report[str(s)] = rows
    report["mean"] = {
        k: float(np.mean([r[k] for s in SUBJECTS for r in report[str(s)]]))
        for k in ("at_validation_choice", "best_on_test")
    }
    save("selection", report)


# --------------------------------------------------------------------------- 4. smoothing
def part_smoothing(data_dir: str) -> None:
    report = {}
    for s in SUBJECTS:
        d = load_subject(data_dir, s)
        prep = prepare(d)
        pred = np.load(f"results/preds/lstm_selected_sub{s}.npy")[1:]  # seed-0 model, test steps
        t = prep.ends[prep.test_rows] - prep.n_train_samples
        rows = {
            "live (as scored)": score_1khz(upsample_hold(pred, t, len(d.test_ecog)), d.test_flex)[0],
            "linear interpolation only (uses the next output)": score_1khz(
                upsample_linear(pred, t, len(d.test_ecog)), d.test_flex
            )[0],
        }
        for sigma_ms in (40, 60, 120):
            smooth = gaussian_filter1d(pred, sigma=sigma_ms / 40, axis=0)  # centred: uses future outputs
            rows[f"centred Gaussian, sigma {sigma_ms} ms, interpolated"] = score_1khz(
                upsample_linear(smooth, t, len(d.test_ecog)), d.test_flex
            )[0]
        report[str(s)] = rows
        print(f"S{s}", {k: round(v, 3) for k, v in rows.items()}, flush=True)
    report["mean"] = {k: float(np.mean([report[str(s)][k] for s in SUBJECTS])) for k in report["1"]}
    save("smoothing", report)


def part_fingerflex(data_dir: str, subjects=SUBJECTS, seeds=(0,)) -> None:
    """FingerFlex as published (offline), epoch chosen on the test set (as published) or on validation. One file
    per subject and seed in results/offline/fingerflex/, so runs can share the machine; then merged."""
    folder = OUT / "fingerflex"
    folder.mkdir(parents=True, exist_ok=True)
    for s in subjects:
        d = load_subject(data_dir, s)
        for seed in seeds:
            print(f"S{s} seed {seed}", flush=True)
            res = run_fingerflex_offline(d.train_ecog, d.train_flex, d.test_ecog, d.test_flex, seed=seed)
            for kind in ("published", "validation_epoch"):
                if seed == 0:
                    save_pred(f"fingerflex_{kind}", s, res[kind]["pred_1khz"])
                res[kind].pop("pred_1khz")
            (folder / f"s{s}_seed{seed}.json").write_text(json.dumps(res, indent=2))
            print(
                f"S{s} seed {seed}: published {res['published']['score']:.3f} (epoch {res['published']['epoch']}), "
                f"epoch chosen on validation {res['validation_epoch']['score']:.3f} "
                f"(epoch {res['validation_epoch']['epoch']})",
                flush=True,
            )
    merge_fingerflex()


def merge_fingerflex() -> None:
    """results/offline/fingerflex.json: per subject and overall, mean and SD over seeds of both variants."""
    files = sorted((OUT / "fingerflex").glob("s*_seed*.json"))
    runs: dict[str, dict[int, dict]] = {}
    for f in files:
        s, seed = f.stem.split("_seed")
        runs.setdefault(s[1:], {})[int(seed)] = json.loads(f.read_text())
    seeds = sorted(set.intersection(*(set(r) for r in runs.values())))
    report: dict = {"seeds": seeds}
    for kind in ("published", "validation_epoch"):
        per_subject = {s: [runs[s][seed][kind] for seed in seeds] for s in sorted(runs, key=int)}
        means = {
            s: {m: float(np.mean([r[m] for r in rs])) for m in ("score", "notebook_metric")}
            for s, rs in per_subject.items()
        }
        seed_scores = [np.mean([per_subject[s][i]["score"] for s in per_subject]) for i in range(len(seeds))]
        report[kind] = {
            "subjects": means,
            "epochs": {s: [r["epoch"] for r in rs] for s, rs in per_subject.items()},
            "score": float(np.mean([m["score"] for m in means.values()])),
            "notebook_metric": float(np.mean([m["notebook_metric"] for m in means.values()])),
            "sd": float(np.std(seed_scores)),
        }
    save("fingerflex", report)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--parts", nargs="+", default=["liang", "flamary", "wiener", "selection", "smoothing"])
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--subjects", type=int, nargs="+", default=list(SUBJECTS), help="fingerflex part only")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0], help="fingerflex part only")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    record_test_look("offline study (offline_study.py)", {"parts": args.parts})
    for part in args.parts:
        if part == "fingerflex":
            part_fingerflex(args.data_dir, args.subjects, tuple(args.seeds))
            continue
        {
            "liang": part_liang,
            "flamary": part_flamary,
            "diagnostics": part_diagnostics,
            "wiener": part_wiener,
            "selection": part_selection,
            "smoothing": part_smoothing,
        }[part](args.data_dir)


if __name__ == "__main__":
    main()
