"""Run decoding approaches under the shared protocol and score them on the blind test set.

Usage:
    python scripts/run_benchmark.py --data-dir data                      # every approach
    python scripts/run_benchmark.py --data-dir data --approaches lstm wiener --streaming
    python scripts/summarise.py                                          # results table and metrics.json

Every reported number is the score of ONE model: networks are trained with several seeds, each seed
is scored on its own, and the mean ± SD over seeds is reported (outputs are never averaged).
For each approach this writes results/runs/<key>.json (validation and test scores per subject
and seed, chosen settings) and results/preds/<key>_subX.npy (test predictions of the seed-0 model,
fixed in advance, starting with the last training step so the causal hold at 1 kHz is exact).
Approaches can run in parallel processes.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from ecogdecode.approaches import APPROACHES, run
from ecogdecode.config import FeatureConfig, Protocol
from ecogdecode.data import SUBJECTS, load_subject
from ecogdecode.pipeline import evaluate, prepare
from ecogdecode.provenance import environment, record_test_look
from ecogdecode.streaming import replay_online

GATED = ("gated_lfp", "gated_all", "gated_transient")


def to_json(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(type(obj))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out", default="results")
    ap.add_argument("--approaches", nargs="+", default=list(APPROACHES), choices=list(APPROACHES))
    ap.add_argument("--subjects", type=int, nargs="+", default=list(SUBJECTS))
    ap.add_argument("--seeds", type=int, nargs="+", default=list(Protocol().seeds))
    ap.add_argument("--no-refit", action="store_true", help="keep the early-stopped models (no final retraining)")
    ap.add_argument("--streaming", action="store_true", help="export seed 0 to ONNX and replay the recording live")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument(
        "--tag",
        default="",
        help="write runs/parts/<key>__<tag>.json, one part of a run split over processes "
        "(scripts/merge_runs.py joins the parts)",
    )
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    protocol = Protocol(seeds=tuple(args.seeds), refit=not args.no_refit)
    out = Path(args.out)
    for sub in ("runs", "preds", "onnx", "gates"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    datas = {s: load_subject(args.data_dir, s) for s in args.subjects}
    preps = {s: prepare(d, FeatureConfig(), protocol) for s, d in datas.items()}
    record_test_look(
        "blind test (run_benchmark.py)", {"approaches": args.approaches, "subjects": args.subjects, "seeds": args.seeds}
    )
    for s, p in preps.items():
        np.save(
            out / "preds" / f"truth_sub{s}.npy",
            datas[s].test_flex[p.ends[p.test_rows] - p.n_train_samples].astype(np.float32),
        )
        np.save(out / "preds" / f"times_sub{s}.npy", p.test_times)

    for key in args.approaches:
        approach, t0 = APPROACHES[key], time.time()
        results = run(key, preps, datas, protocol)
        report = {
            "key": key,
            "number": approach.number,
            "label": approach.label,
            "family": approach.family,
            "protocol": asdict(protocol),
            "train_config": asdict(approach.train_cfg),
            "environment": environment(args.threads),
            "subjects": {},
        }
        for s, runs in results.items():
            prep, test_flex = preps[s], datas[s].test_flex
            per_run = [
                {"val_score": r.val_score, "selected": r.selected, **evaluate(prep, r.pred, test_flex)} for r in runs
            ]
            entry = {
                "val_score": float(np.mean([r.val_score for r in runs])),
                "score": float(np.mean([p["score"] for p in per_run])),  # mean over single models
                "sd": float(np.std([p["score"] for p in per_run])),
                "r": np.mean([p["r"] for p in per_run], axis=0).tolist(),
                "runs": per_run,
            }
            seed0 = [r for r in runs if r.selected.get("seed", 0) == 0]  # figures use the seed-0 model
            if seed0:
                np.save(out / "preds" / f"{key}_sub{s}.npy", prep.test_predictions(seed0[0].pred).astype(np.float32))
            if key in GATED:
                with torch.no_grad():
                    gates = runs[0].model.eval().gates(torch.from_numpy(runs[0].x)[None])[0].numpy()
                np.save(out / "gates" / f"{key}_sub{s}.npy", gates[prep.test_rows])
            if args.streaming and approach.streamable:
                entry["streaming"] = replay_online(
                    runs[0].model,
                    runs[0].x,
                    runs[0].x_scaler,
                    datas[s].ecog,
                    prep.cfg,
                    out / "onnx" / f"{key}_sub{s}.onnx",
                )
            report["subjects"][str(s)] = entry
        report["val_score"] = float(np.mean([e["val_score"] for e in report["subjects"].values()]))
        report["score"] = float(np.mean([e["score"] for e in report["subjects"].values()]))
        seed_means = [np.mean([e["runs"][i]["score"] for e in report["subjects"].values()]) for i in range(len(runs))]
        report["sd"] = float(np.std(seed_means))  # spread over seeds of the subject-averaged single-model score
        report["seconds"] = round(time.time() - t0, 1)
        target = out / "runs" / "parts" / f"{key}__{args.tag}.json" if args.tag else out / "runs" / f"{key}.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(report, indent=2, default=to_json))
        per_subject = "  ".join(f"S{s} {e['score']:.3f}" for s, e in report["subjects"].items())
        print(
            f"{key:<18} val {report['val_score']:.3f}   test {report['score']:.3f} ± {report['sd']:.3f}   "
            f"({per_subject})   {report['seconds']} s",
            flush=True,
        )


if __name__ == "__main__":
    main()
