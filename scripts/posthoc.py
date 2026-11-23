"""Post-hoc analyses of the saved test predictions (the seed-0 model of each approach). Nothing feeds back.

1. Uncertainty from the test data. A paired block bootstrap resamples 4 s blocks of the test segment (about one cue
   and its rest; 50 blocks per patient), the same blocks for every approach, and recomputes the official score
   exactly (Pearson r at 1 kHz from per-block sums). It gives a 95 % interval for each score and for the difference
   between two approaches. The SD over training seeds in the main table measures training randomness instead.
2. Which finger. The correlation of every decoded finger with every recorded finger (5 x 5 per patient);
   off-diagonal values are cross-talk between fingers.
3. Lag. The shift between -400 and +400 ms at which decoded and recorded flexion correlate best; positive values
   mean the decoder lags behind the glove.
4. Movement or trajectory. r of the movement envelope (both traces averaged over 1 s), which a decoder gets right
   when it knows which finger moves when, against r within movements only (samples where that finger is flexed),
   which needs the shape of each flexion.
5. Artefact masking. How often the online rule silenced an electrode in the test segment.

Usage:
    python scripts/posthoc.py --data-dir data
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import uniform_filter1d

from ecogdecode.approaches import APPROACHES
from ecogdecode.data import SCORED_FINGERS, SUBJECTS, load_subject
from ecogdecode.evaluation import pearson_r, to_sample_rate
from ecogdecode.features import Standardiser, mask_artefact_channels
from ecogdecode.pipeline import prepare
from ecogdecode.provenance import environment, record_test_look

PREDS = Path("results/preds")
BLOCK = 4000  # samples (4 s) per bootstrap block
N_BOOT = 2000
LAGS_MS = np.arange(-400, 401, 10)
PAIRS = [("lstm_selected", "lstm"), ("lstm_selected", "fingerflex"), ("lstm_selected", "possm"),
         ("lstm", "mamba"), ("lstm", "tcn"), ("lstm", "s4d"), ("lstm", "wiener"), ("lstm", "possm"),
         ("possm_pretrained", "possm"), ("wiener", "liang_bougrain"), ("wiener", "kalman")]


def held_1khz(key: str, s: int, n: int) -> np.ndarray | None:
    path = PREDS / f"{key}_sub{s}.npy"
    if not path.exists():
        return None
    times = np.load(PREDS / f"times_sub{s}.npy")
    return to_sample_rate(np.load(path).astype(np.float64), times, n)


def block_sums(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    """Per block and finger: n, sum x, sum y, sum x^2, sum y^2, sum xy -> (n_blocks, 6, 5)."""
    nb = len(true) // BLOCK
    p = pred[: nb * BLOCK].reshape(nb, BLOCK, -1)
    t = true[: nb * BLOCK].reshape(nb, BLOCK, -1).astype(np.float64)
    return np.stack([np.full(p.shape[::2], BLOCK, float), p.sum(1), t.sum(1), (p * p).sum(1), (t * t).sum(1),
                     (p * t).sum(1)], axis=1)


def r_from_sums(s: np.ndarray) -> np.ndarray:
    """Pearson r per finger from summed statistics (..., 6, 5)."""
    n, sx, sy, sxx, syy, sxy = (s[..., i, :] for i in range(6))
    return (n * sxy - sx * sy) / np.sqrt((n * sxx - sx ** 2) * (n * syy - sy ** 2))


def lag_profile(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    """Mean r over scored fingers for decoded(t) against recorded(t - lag), for each lag in LAGS_MS."""
    out = []
    for lag in LAGS_MS:
        if lag >= 0:
            p, t = pred[lag:], true[: len(true) - lag]
        else:
            p, t = pred[: lag], true[-lag:]
        out.append(float(pearson_r(p, t)[list(SCORED_FINGERS)].mean()))
    return np.asarray(out)


def movement_split(pred: np.ndarray, true: np.ndarray) -> tuple[float, float]:
    """(envelope r, within-movement r), each the mean over scored fingers."""
    env = pearson_r(uniform_filter1d(pred, 1000, axis=0), uniform_filter1d(true, 1000, axis=0))
    within = []
    for f in SCORED_FINGERS:
        lo, hi = np.percentile(true[:, f], [20, 99])
        moving = true[:, f] > lo + 0.25 * (hi - lo)
        within.append(float(pearson_r(pred[moving, f], true[moving, f])[0]) if moving.sum() > 1000 else np.nan)
    return float(env[list(SCORED_FINGERS)].mean()), float(np.nanmean(within))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out", default="results/posthoc.json")
    args = ap.parse_args()
    record_test_look("post-hoc analyses of saved seed-0 predictions (posthoc.py)")
    rng = np.random.default_rng(0)
    keys = [k for k in APPROACHES if all((PREDS / f"{k}_sub{s}.npy").exists() for s in SUBJECTS)]
    report: dict = {"environment": environment(), "approaches": keys, "subjects": {}}
    sums: dict[str, list[np.ndarray]] = {k: [] for k in keys}
    counts = []
    for s in SUBJECTS:
        data = load_subject(args.data_dir, s)
        true = data.test_flex.astype(np.float64)
        n = len(true)
        preds = {k: held_1khz(k, s, n) for k in keys}
        nb = n // BLOCK
        counts.append(rng.multinomial(nb, np.full(nb, 1 / nb), size=N_BOOT).astype(float))  # (N_BOOT, nb)
        entry: dict = {"n_blocks": nb, "approaches": {}}
        for k, pred in preds.items():
            sums[k].append(block_sums(pred, true))
            cross = np.array([[float(pearson_r(pred[:, i], true[:, j])[0]) for j in range(5)] for i in range(5)])
            env, within = movement_split(pred, true)
            profile = lag_profile(pred, true)
            entry["approaches"][k] = {"score": float(r_from_sums(sums[k][-1].sum(0))[list(SCORED_FINGERS)].mean()),
                                      "cross_finger_r": cross.round(4).tolist(),
                                      "best_lag_ms": int(LAGS_MS[np.argmax(profile)]),
                                      "r_at_best_lag": float(profile.max()), "envelope_r": env,
                                      "within_movement_r": within}
        prep = prepare(data)
        x_std = Standardiser().fit(prep.x[prep.train_rows]).transform(prep.x)  # the final models' statistics
        masked = mask_artefact_channels(x_std, prep.cfg.n_per_channel, prep.cfg.artefact_z)[1][prep.test_rows]
        entry["masking"] = {"fraction_of_electrode_steps": float(masked.mean()),
                            "fraction_of_steps_with_any": float(masked.any(axis=1).mean()),
                            "electrodes_ever_masked": int(masked.any(axis=0).sum())}
        report["subjects"][str(s)] = entry
        print(f"S{s}: masking {entry['masking']}", flush=True)

    # paired block bootstrap of the three-patient mean score
    boot = {}
    for k in keys:
        per_subject = [r_from_sums(np.einsum("bn,nsf->bsf", c, sm))[:, list(SCORED_FINGERS)].mean(1)
                       for c, sm in zip(counts, sums[k])]
        boot[k] = np.mean(per_subject, axis=0)  # (N_BOOT,)
    point = {k: float(np.mean([report["subjects"][str(s)]["approaches"][k]["score"] for s in SUBJECTS])) for k in keys}
    report["bootstrap"] = {k: {"score": point[k], "ci95": np.percentile(boot[k], [2.5, 97.5]).round(4).tolist()}
                           for k in keys}
    report["paired_differences"] = []
    for a, b in PAIRS:
        if a in boot and b in boot:
            d = boot[a] - boot[b]
            report["paired_differences"].append({"a": a, "b": b, "difference": point[a] - point[b],
                                                 "ci95": np.percentile(d, [2.5, 97.5]).round(4).tolist(),
                                                 "p_not_greater": float((d <= 0).mean())})
    for row in report["paired_differences"]:
        print(f"{row['a']:>16} - {row['b']:<16} {row['difference']:+.3f}  95% CI [{row['ci95'][0]:+.3f}, "
              f"{row['ci95'][1]:+.3f}]")
    Path(args.out).write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
