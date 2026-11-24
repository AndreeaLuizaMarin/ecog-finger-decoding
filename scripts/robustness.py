"""Post-hoc robustness check: what happens to a trained decoder when its most important electrode fails?

For each subject, the seed-0 model of each approach is trained as in the benchmark, then scored on the test set
twice: intact, and with the electrode most related to movement in the training data silenced for the whole test
segment (its features set to the training mean, which is what the online artefact masking does to a failed
electrode). Nothing here feeds back into any choice; it is an analysis of the final models.

Usage:
    python scripts/robustness.py --data-dir data --approaches lstm lstm_selected
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from ecogdecode.approaches import run_network
from ecogdecode.config import DEFAULT_PROTOCOL
from ecogdecode.data import SUBJECTS, load_subject
from ecogdecode.evaluation import causal_ema
from ecogdecode.models.training import predict_sequence
from ecogdecode.pipeline import evaluate, prepare, top_electrodes
from ecogdecode.provenance import record_test_look


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--approaches", nargs="+", default=["lstm", "lstm_selected"])
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--out", default="results/robustness.json")
    args = ap.parse_args()
    record_test_look("one electrode lost (robustness.py)")
    torch.set_num_threads(args.threads)

    report = {}
    for s in SUBJECTS:
        data = load_subject(args.data_dir, s)
        prep = prepare(data)
        npc = prep.cfg.n_per_channel
        lost = int(top_electrodes(prep, prep.train_rows, 1)[0])
        for key in args.approaches:
            res = run_network(prep, key, seed=0, protocol=DEFAULT_PROTOCOL)
            electrodes = res.selected.get("electrodes", list(range(prep.x.shape[1] // npc)))
            col = electrodes.index(lost) * npc
            x = res.x.copy()
            x[prep.test_rows, col : col + npc] = 0.0  # standardised features: 0 = training mean
            tau = res.selected["smooth_ms"]
            intact = evaluate(prep, causal_ema(predict_sequence(res.model, res.x), tau), data.test_flex)["score"]
            failed = evaluate(prep, causal_ema(predict_sequence(res.model, x), tau), data.test_flex)["score"]
            report.setdefault(key, {})[str(s)] = {
                "lost_electrode": lost,
                "n_electrodes": len(electrodes),
                "intact": intact,
                "one_electrode_lost": failed,
            }
            print(
                f"S{s} {key:<14} {len(electrodes):>2} electrodes   intact {intact:.3f}   "
                f"electrode {lost} lost {failed:.3f}"
            )
    for rows in report.values():
        rows["mean"] = {k: float(np.mean([r[k] for r in rows.values()])) for k in ("intact", "one_electrode_lost")}
    Path(args.out).write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
