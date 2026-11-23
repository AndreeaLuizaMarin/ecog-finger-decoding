"""Why does the Kalman filter decode ECoG poorly, and does a better-specified one help? (validation only)

Compares, on the validation block of each patient (test labels are never loaded):
  * the classical Kalman filter (Wu et al., 2006): observations = PCA of the current features;
  * a lagged Kalman filter: observations = PCA of the current and recent features;
  * the Wiener filter (lagged ridge regression) alone;
  * a discriminative Kalman filter (Burkhart et al., 2020) on top of that Wiener filter.

Usage:
    python scripts/kalman_study.py --data-dir data
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ecogdecode.data import SUBJECTS, load_subject
from ecogdecode.evaluation import score
from ecogdecode.models import DiscriminativeKalmanDecoder, KalmanDecoder, LaggedKalmanDecoder, LaggedRidge
from ecogdecode.models.linear import select_ridge
from ecogdecode.pipeline import prepare, standardise


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out", default="results/kalman_study.json")
    ap.add_argument("--subjects", nargs="+", type=int, default=list(SUBJECTS))
    args = ap.parse_args()
    out = Path(args.out)
    report = json.loads(out.read_text()) if out.exists() else {}  # results of earlier subjects are kept
    for s in args.subjects:
        data = load_subject(args.data_dir, s)
        data.test_flex = None  # validation only
        prep = prepare(data)
        sc = standardise(prep, prep.fit_rows)
        fit, val = prep.fit_rows, prep.val_rows
        yv = sc.y[val]
        res = {}
        for k in (8, 16, 32, 64):
            res[f"kalman_pca{k}"] = score(KalmanDecoder(k).fit(sc.x[fit], sc.y[fit]).predict(sc.x[: val.stop])[val], yv)
            for lags in (3, 5):
                kf = LaggedKalmanDecoder(k, lags).fit(sc.x[fit], sc.y[fit])
                res[f"kalman_pca{k}_lags{lags}"] = score(kf.predict(sc.x[: val.stop])[val], yv)
        best, _, wiener_val = select_ridge(sc.x, sc.y, fit, val, (10, 15, 20, 25), np.logspace(2, 6.5, 10), score)
        res["wiener"] = best["val_score"]
        # DKF: the regressor's residual covariance comes from a chronological split inside the fit rows
        inner = slice(fit.start, fit.start + int(0.75 * (fit.stop - fit.start)))
        held = slice(inner.stop, fit.stop)
        r_inner = LaggedRidge(best["n_lags"], best["alpha"]).fit(sc.x, sc.y, inner)
        q = np.cov((sc.y[held] - r_inner.predict(sc.x[: held.stop])[held]).T)
        reg = LaggedRidge(best["n_lags"], best["alpha"]).fit(sc.x, sc.y, fit)
        dkf = DiscriminativeKalmanDecoder(reg, q).fit(sc.y[fit])
        res["dkf_on_wiener"] = score(dkf.predict(sc.x[: val.stop])[val], yv)
        report[str(s)] = {k: float(v) for k, v in res.items()}
        print(s, {k: round(v, 3) for k, v in res.items()}, flush=True)
        out.write_text(json.dumps(report, indent=2))
    done = [str(s) for s in SUBJECTS if str(s) in report]
    if len(done) == len(SUBJECTS):
        keys = report[done[0]].keys()
        report["mean"] = {k: float(np.mean([report[s][k] for s in done])) for k in keys}
        print("mean", {k: round(v, 3) for k, v in sorted(report["mean"].items(), key=lambda kv: -kv[1])})
        out.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
