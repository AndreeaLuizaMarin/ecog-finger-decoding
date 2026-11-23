"""How much calibration data does a new patient need? Test r against minutes of training data (post hoc).

For each patient and each amount D of training data (1, 2, 4 and all 6.5 minutes), only the last D minutes of the
training segment are used, split as everywhere (first 80 % fit, last 20 % validation for every choice, then a refit
on all D minutes), and the model is scored on the test segment. Taking the last D minutes keeps the gap to the test
segment the same for every D, so only the amount of data changes.

Decoders:
* wiener          approach 3 (closed form);
* lstm            approach 5;
* lstm_selected   approach 5b (electrodes ranked on the D minutes; the number kept per patient as in 5b);
* lstm_others     an LSTM whose recurrent core was pretrained with labels on the *other two* patients' whole training
                  segments (as approach 11b, which pretrains on all three), then fine-tuned on the D minutes with a
                  fresh input layer: a new patient joining a pretrained model.

Nothing here feeds back into any decoder; the test labels are read once per run and the look is logged.

Usage (one process per subject and seed, so runs can share the machine):
    python scripts/calibration.py --data-dir data --subject 1 --seed 0
    python scripts/calibration.py --summarise
"""

from __future__ import annotations

import argparse
import copy
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from ecogdecode.approaches import CLOSED_FORM, run_network
from ecogdecode.config import DEFAULT_PROTOCOL, TrainConfig
from ecogdecode.data import SUBJECTS, load_subject
from ecogdecode.models import LSTMDecoder
from ecogdecode.pipeline import Prepared, evaluate, prepare, standardise
from ecogdecode.pretrain import pretrain_supervised
from ecogdecode.provenance import environment, record_test_look

OUT = Path("results/calibration")
MINUTES = (1.0, 2.0, 4.0, None)  # None = the whole training segment
STEPS_PER_MINUTE = 60 * 25


def last_minutes(prep: Prepared, minutes: float | None) -> Prepared:
    """The last ``minutes`` of the training segment, split 80 / 20 into fit and validation rows."""
    end = prep.train_rows.stop
    start = (
        prep.train_rows.start
        if minutes is None
        else max(end - int(round(minutes * STEPS_PER_MINUTE)), prep.train_rows.start)
    )
    split = end - int(round(0.2 * (end - start)))
    return replace(prep, fit_rows=slice(start, split), val_rows=slice(split, end), train_rows=slice(start, end))


def pretrained_on_others(preps: dict[int, Prepared], target: int, seed: int) -> torch.nn.Module:
    """An LSTM whose recurrent core was pretrained with labels on every other patient's whole training segment
    (``pretrain.pretrain_supervised``, as approach 11b), with a fresh input layer sized for the target patient."""
    others = {s: p for s, p in preps.items() if s != target}
    scaled = {s: standardise(p, p.train_rows) for s, p in others.items()}
    decoders = pretrain_supervised(
        {s: v.x for s, v in scaled.items()},
        {s: v.y for s, v in scaled.items()},
        {s: p.train_rows for s, p in others.items()},
        seed=seed,
    )
    torch.manual_seed(seed)
    model = LSTMDecoder(preps[target].x.shape[1])  # 128 units, as in pretraining
    model.rnn = copy.deepcopy(next(iter(decoders.values())).rnn)
    return model


def run(data_dir: str, subject: int, seed: int, threads: int) -> None:
    torch.set_num_threads(threads)
    preps, data = {}, None
    for s in SUBJECTS:  # keep the features only; the raw recordings are not needed after this (memory)
        d = load_subject(data_dir, s)
        preps[s] = prepare(d)
        if s == subject:
            data = replace(d, train_ecog=d.train_ecog[:0], test_ecog=d.test_ecog[:0])
    record_test_look("calibration curve (calibration.py)", {"subject": subject, "seed": seed})
    prep = preps[subject]
    base = pretrained_on_others(preps, subject, seed)
    rows = []
    for minutes in MINUTES:
        sub = last_minutes(prep, minutes)
        label = "all" if minutes is None else minutes
        results = {}
        if seed == 0:  # deterministic
            results["wiener"] = CLOSED_FORM["wiener"](sub, data, DEFAULT_PROTOCOL)
        for key in ("lstm", "lstm_selected"):
            results[key] = run_network(sub, key, seed, DEFAULT_PROTOCOL)
        results["lstm_others"] = run_network(
            sub, "lstm", seed, DEFAULT_PROTOCOL, model=copy.deepcopy(base), train_cfg=TrainConfig()
        )
        for key, res in results.items():
            test = evaluate(prep, res.pred, data.test_flex)
            rows.append(
                {
                    "approach": key,
                    "minutes": label,
                    "train_steps": sub.train_rows.stop - sub.train_rows.start,
                    "subject": subject,
                    "seed": seed,
                    "val_score": res.val_score,
                    "score": test["score"],
                    "r": test["r"],
                    "epochs": res.selected.get("epochs"),
                }
            )
            print(
                f"S{subject} seed {seed} {label!s:>4} min  {key:<14} val {res.val_score:.3f}  test {test['score']:.3f}",
                flush=True,
            )
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"s{subject}_seed{seed}.json").write_text(
        json.dumps({"environment": environment(threads), "rows": rows}, indent=2)
    )


def summarise() -> dict:
    """Per approach and amount of data: the mean over subjects of the single-model test scores (per seed), their
    mean and SD over seeds, and each subject's mean over seeds."""
    rows = [r for f in sorted(OUT.glob("s*_seed*.json")) for r in json.loads(f.read_text())["rows"]]
    out: dict = {}
    for key in sorted({r["approach"] for r in rows}):
        out[key] = {}
        for minutes in ("1.0", "2.0", "4.0", "all"):
            sel = [r for r in rows if r["approach"] == key and str(r["minutes"]) == minutes]
            if not sel:
                continue
            subjects = sorted({r["subject"] for r in sel})
            seeds = sorted(
                {
                    r["seed"]
                    for r in sel
                    if all(any(q["subject"] == s and q["seed"] == r["seed"] for q in sel) for s in subjects)
                }
            )
            per_seed = [np.mean([r["score"] for r in sel if r["seed"] == seed]) for seed in seeds]
            per_subject = {str(s): float(np.mean([r["score"] for r in sel if r["subject"] == s])) for s in subjects}
            out[key][minutes] = {
                "score": float(np.mean(per_seed)),
                "sd": float(np.std(per_seed)),
                "subjects": per_subject,
                "seeds": seeds,
            }
    (OUT.parent / "calibration.json").write_text(json.dumps(out, indent=2))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--subject", type=int)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--summarise", action="store_true")
    args = ap.parse_args()
    if args.summarise:
        for key, by_minutes in summarise().items():
            print(key, {m: round(v["score"], 3) for m, v in by_minutes.items()})
    else:
        run(args.data_dir, args.subject, args.seed, args.threads)


if __name__ == "__main__":
    main()
