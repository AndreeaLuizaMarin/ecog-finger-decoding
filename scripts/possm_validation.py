"""Validation-only check of the POSSM-style decoder's electrode embeddings (round 8; test labels never loaded).

The first version of approach 15 started its electrode embeddings at scale 0.02 and scored 0.13 on validation (0.10
on test): next to the projected features the embeddings were too small for the attention to tell electrodes apart.
This compares embedding scales and latent counts on the validation block only, seed 0, every patient.

Usage:
    python scripts/possm_validation.py --data-dir data --variants scale1 scale1_latents8
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
from ecogdecode.models import POSSMDecoder
from ecogdecode.pipeline import prepare

VARIANTS = {
    "scale0.02": {"embedding_scale": 0.02},
    "scale1": {"embedding_scale": 1.0},
    "scale1_latents8": {"embedding_scale": 1.0, "n_latents": 8},
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=list(VARIANTS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--out", default="results/validation_study_r8_possm.json")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    preps = {}
    for s in SUBJECTS:
        d = load_subject(args.data_dir, s, require_test_labels=False)
        d.test_flex = None  # validation only: the test labels are not used anywhere here
        preps[s] = prepare(d)
    out = Path(args.out)
    report = json.loads(out.read_text()) if out.exists() else {}
    for name in args.variants:
        scores = {}
        for s, prep in preps.items():
            torch.manual_seed(args.seed)
            model = POSSMDecoder(prep.x.shape[1], prep.cfg.n_per_channel, **VARIANTS[name])
            res = run_network(prep, "possm", args.seed, DEFAULT_PROTOCOL, model=model)
            scores[str(s)] = res.val_score
            print(f"{name:<18} S{s} validation {res.val_score:.3f} ({res.selected['epochs']} epochs)", flush=True)
        report[name] = {
            "options": VARIANTS[name],
            "seed": args.seed,
            "per_subject": scores,
            "validation_r": float(np.mean(list(scores.values()))),
        }
        out.write_text(json.dumps(report, indent=2))
        print(f"{name:<18} mean validation {report[name]['validation_r']:.3f}", flush=True)


if __name__ == "__main__":
    main()
