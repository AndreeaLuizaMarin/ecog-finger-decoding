"""Choose the feature window and decoder history on the VALIDATION block only.

The official test labels are never read here (``require_test_labels=False``), so this
selection cannot leak test information. The lagged-ridge decoder is used as a fast proxy.

Usage:
    python scripts/select_features.py --data-dir data
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ecogdecode.data import SUBJECTS, load_subject
from ecogdecode.config import FeatureConfig
from ecogdecode.evaluation import score
from ecogdecode.models import select_ridge
from ecogdecode.pipeline import prepare, standardise


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--windows", type=int, nargs="+", default=[100, 200, 300])
    ap.add_argument("--lags", type=int, nargs="+", default=[5, 10, 15, 25])
    ap.add_argument("--out", default="results/feature_selection.json")
    args = ap.parse_args()

    table = {}
    for subject in SUBJECTS:
        data = load_subject(args.data_dir, subject, require_test_labels=False)
        data.test_flex = None  # make sure no test labels are present
        for window in args.windows:
            prep = prepare(data, FeatureConfig(window_ms=window))
            sc = standardise(prep, prep.fit_rows)
            best, grid, _ = select_ridge(sc.x, sc.y, prep.fit_rows, prep.val_rows, tuple(args.lags),
                                      np.logspace(1, 6, 11), score)
            for lag in args.lags:
                best_for_lag = max(g["val_score"] for g in grid if g["n_lags"] == lag)
                table.setdefault(f"{window}ms_{lag}lags", {})[subject] = best_for_lag
            print(f"subject {subject}, window {window} ms: best val r = {best['val_score']:.3f} "
                  f"({best['n_lags']} lags)")

    summary = {k: {"per_subject": v, "mean_val_r": float(np.mean(list(v.values())))} for k, v in table.items()}
    best = max(summary, key=lambda k: summary[k]["mean_val_r"])
    print("\nmean validation r:")
    for k, v in sorted(summary.items(), key=lambda kv: -kv[1]["mean_val_r"]):
        print(f"  {k:<14} {v['mean_val_r']:.3f}")
    print(f"selected: {best}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"selected": best, "results": summary}, indent=2))


if __name__ == "__main__":
    main()
