"""Post hoc: how fast does each decoder degrade as electrodes fail, and does electrode dropout in training help?

For each patient, the seed-0 models of approach 5 (LSTM, all electrodes), 5b (LSTM, selected electrodes) and 5b
trained with electrode dropout (each electrode silenced with probability 0.1 in every training window; the validation
study scored it 0.59-0.60 against 0.62-0.66 without) are trained as in the benchmark. Electrodes are then
silenced for the whole test segment (features set to the training mean, which is what the online artefact rule does
to a failed electrode):
* best first: the k electrodes most related to movement in the training data (k = 0 ... 8);
* at random: k electrodes drawn from all of the patient's electrodes (k = 1, 2, 4, 8, 16; 20 draws each). A decoder
  that reads only selected electrodes is unaffected when a failed electrode is not one of them.
Nothing here feeds back into any decoder.

Usage:
    python scripts/robustness_curve.py --data-dir data
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from ecogdecode.approaches import TUNED_LSTM, run_network
from ecogdecode.config import DEFAULT_PROTOCOL
from ecogdecode.data import SUBJECTS, load_subject
from ecogdecode.evaluation import causal_ema
from ecogdecode.models.training import predict_sequence
from ecogdecode.pipeline import evaluate, prepare, top_electrodes
from ecogdecode.provenance import environment, record_test_look

VARIANTS = {
    "lstm": ("lstm", TUNED_LSTM),
    "lstm_selected": ("lstm_selected", TUNED_LSTM),
    "lstm_selected_dropout": ("lstm_selected", replace(TUNED_LSTM, channel_dropout=0.1)),
}
BEST_FIRST = range(0, 9)
RANDOM_K = (1, 2, 4, 8, 16)
DRAWS = 20


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--out", default="results/robustness_curve.json")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    record_test_look("electrode-loss curve (robustness_curve.py)")
    rng = np.random.default_rng(0)
    report: dict = {"environment": environment(args.threads), "subjects": {}}
    for s in SUBJECTS:
        data = load_subject(args.data_dir, s)
        prep = prepare(data)
        n_el = prep.x.shape[1] // prep.cfg.n_per_channel
        ranking = sorted(range(n_el), key=lambda e: -_relatedness(prep, e))  # most related to movement first
        assert set(ranking[:8]) == set(top_electrodes(prep, prep.train_rows, 8).tolist())
        draws = {k: [rng.choice(n_el, size=k, replace=False).tolist() for _ in range(DRAWS)] for k in RANDOM_K}
        entry: dict = {"ranking": ranking[:16], "n_electrodes": n_el}
        for name, (key, cfg) in VARIANTS.items():
            res = run_network(prep, key, seed=0, protocol=DEFAULT_PROTOCOL, train_cfg=cfg)
            used = res.selected.get("electrodes", list(range(n_el)))
            best = [silenced_score(res, prep, data.test_flex, used, ranking[:k]) for k in BEST_FIRST]
            random = {
                str(k): [silenced_score(res, prep, data.test_flex, used, d) for d in ds] for k, ds in draws.items()
            }
            entry[name] = {
                "electrodes_used": len(used),
                "best_first": best,
                "random_mean": {k: float(np.mean(v)) for k, v in random.items()},
                "random_sd": {k: float(np.std(v)) for k, v in random.items()},
            }
            print(
                f"S{s} {name:<22} intact {best[0]:.3f}  best-first k=1 {best[1]:.3f}  k=4 {best[4]:.3f}  "
                f"k=8 {best[8]:.3f}  random k=4 {entry[name]['random_mean']['4']:.3f}",
                flush=True,
            )
        report["subjects"][str(s)] = entry
    Path(args.out).write_text(json.dumps(report, indent=2))


def silenced_score(res, prep, test_flex: np.ndarray, used: list[int], lost: list[int]) -> float:
    """Test score of a trained decoder with the ``lost`` electrodes silenced for the whole test segment."""
    npc = prep.cfg.n_per_channel
    x = res.x.copy()
    for e in lost:
        if e in used:
            col = used.index(e) * npc
            x[prep.test_rows, col : col + npc] = 0.0  # standardised features: 0 = training mean
    pred = causal_ema(predict_sequence(res.model, x), res.selected["smooth_ms"])
    return evaluate(prep, pred, test_flex)["score"]


def _relatedness(prep, electrode: int) -> float:
    """Largest absolute correlation between any feature of the electrode and any finger in the training rows
    (the ranking used by approach 5b, pipeline.top_electrodes)."""
    npc = prep.cfg.n_per_channel
    x = prep.x[prep.train_rows, electrode * npc : (electrode + 1) * npc]
    y = np.nan_to_num(prep.y[prep.train_rows])
    xz = (x - x.mean(0)) / (x.std(0) + 1e-9)
    yz = (y - y.mean(0)) / (y.std(0) + 1e-9)
    return float(np.abs(xz.T @ yz / len(x)).max())


if __name__ == "__main__":
    main()
