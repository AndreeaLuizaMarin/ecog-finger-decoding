"""Recompute test scores from the saved predictions and compare them with the numbers in the result files.

For every benchmark approach: the seed-0 model's saved test predictions (results/preds/) are scored with the
competition metric and compared with the seed-0 score stored in results/runs/<key>.json. For every offline replica:
the saved 25 Hz predictions (results/offline/preds/) are interpolated to 1 kHz, scored, and compared with
results/offline/replicas.json; the same for the offline FingerFlex replica and the offline LSTM (round 8).
Needs the competition data (with the test labels).

Usage:
    python scripts/verify_scores.py --data-dir data
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ecogdecode.data import SUBJECTS, load_subject
from ecogdecode.evaluation import score_test
from ecogdecode.offline import score_1khz, upsample_linear
from ecogdecode.pipeline import prepare


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--results", default="results")
    args = ap.parse_args()
    res = Path(args.results)
    runs = {p.stem: json.loads(p.read_text()) for p in sorted((res / "runs").glob("*.json"))}
    replicas = json.loads((res / "offline" / "replicas.json").read_text())
    worst = 0.0
    for s in SUBJECTS:
        d = load_subject(args.data_dir, s)
        prep = prepare(d)
        for key, run in runs.items():
            path = res / "preds" / f"{key}_sub{s}.npy"
            if not path.exists():
                continue
            mine = score_test(np.load(path), prep.test_times, d.test_flex)["score"]
            stored = run["subjects"][str(s)]["runs"][0]["score"]
            worst = max(worst, abs(mine - stored))
            print(f"S{s} {key:<22} recomputed {mine:.4f}   stored {stored:.4f}")
        for key in replicas[str(s)]:
            path = res / "offline" / "preds" / f"{key}_sub{s}.npy"
            if not path.exists():
                continue
            pred = np.load(path).astype(np.float64)
            mine = score_1khz(upsample_linear(pred, np.arange(len(pred)) * 40, len(d.test_ecog)), d.test_flex)[0]
            stored = replicas[str(s)][key]["score"]
            note = ("" if key.startswith("liang_offline") else
                    "  (output was at 250 Hz; saved at 25 Hz, so small differences are expected)" if key.startswith("flamary")
                    else "  (stored score used a hold, not interpolation)")
            if not note:
                worst = max(worst, abs(mine - stored))
            print(f"S{s} {key:<22} recomputed {mine:.4f}   stored {stored:.4f}{note}")
        for name, stored in _round8_offline(res, s):
            path = res / "offline" / "preds" / f"{name}_sub{s}.npy"
            if not path.exists():
                continue
            pred = np.load(path).astype(np.float64)
            mine = score_1khz(upsample_linear(pred, np.arange(len(pred)) * 40, len(d.test_ecog)), d.test_flex)[0]
            print(f"S{s} {name:<22} recomputed {mine:.4f}   stored {stored:.4f}  (saved at 25 Hz from a finer output, "
                  "so small differences are expected)")
    print(f"largest difference: {worst:.2e}")


def _round8_offline(res: Path, s: int) -> list[tuple[str, float]]:
    """Seed-0 scores of the offline FingerFlex replica (both epoch rules) and of the offline LSTM (approach 5b)."""
    out = []
    ff = res / "offline" / "fingerflex" / f"s{s}_seed0.json"
    if ff.exists():
        run = json.loads(ff.read_text())
        out += [(f"fingerflex_{kind}", run[kind]["score"]) for kind in ("published", "validation_epoch")]
    lstm = res / "offline" / "lstm_offline.json"
    if lstm.exists():
        runs = json.loads(lstm.read_text())["subjects"][str(s)]["runs"]
        out += [("lstm_selected_offline", next(r["score"] for r in runs if r["seed"] == 0))]
    return out


if __name__ == "__main__":
    main()
