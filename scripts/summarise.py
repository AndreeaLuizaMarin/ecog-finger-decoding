"""Collect the benchmark runs into the results table (results/summary.md) and results/metrics.json.

Every number is the score of a single model. For networks it is the mean ± SD over the training
seeds, each seed's model scored on its own; outputs are never averaged across seeds or approaches.

Usage:
    python scripts/summarise.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ecogdecode.approaches import APPROACHES

FAMILIES = {
    "competition": "BCI Competition IV entries, re-implemented to run live",
    "classical": "Classical decoders",
    "deep": "Deep sequence models",
    "new": "New in this project",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    args = ap.parse_args()
    res = Path(args.results)

    runs = {
        k: json.loads((res / "runs" / f"{k}.json").read_text())
        for k in APPROACHES
        if (res / "runs" / f"{k}.json").exists()
    }
    subjects = sorted(next(iter(runs.values()))["subjects"], key=int)

    head = (
        "| # | Approach | Validation r | "
        + " | ".join(f"S{s}" for s in subjects)
        + " | **Test r** (one model; mean ± SD over seeds) | Live step |"
    )
    lines = [head, "|" + "---|" * (head.count("|") - 1)]
    for family, title in FAMILIES.items():
        keys = [k for k in runs if APPROACHES[k].family == family]
        if not keys:
            continue
        lines.append(f"| | *{title}* |" + " |" * (head.count("|") - 3))
        for k in keys:
            r = runs[k]
            subj = [f"{r['subjects'][s]['score']:.2f}" for s in subjects]
            deterministic = APPROACHES[k].trainer == "closed_form"
            test = f"**{r['score']:.2f}**" + ("" if deterministic else f" ± {r['sd']:.2f}")
            lat = [r["subjects"][s]["streaming"]["median_ms"] for s in subjects if "streaming" in r["subjects"][s]]
            lat_txt = f"{np.median(lat):.2f} ms" if lat else "–"
            lines.append(
                f"| {APPROACHES[k].number} | {APPROACHES[k].label} | {r['val_score']:.2f} | "
                + " | ".join(subj)
                + f" | {test} | {lat_txt} |"
            )
    table = "\n".join(lines) + "\n"
    (res / "summary.md").write_text(table, encoding="utf-8")

    metrics = {
        "approaches": {
            k: {
                "number": APPROACHES[k].number,
                "label": APPROACHES[k].label,
                "family": APPROACHES[k].family,
                "val_score": r["val_score"],
                "score": r["score"],
                "sd": r.get("sd", 0.0),
                "per_subject": {s: r["subjects"][s]["score"] for s in subjects},
                "per_finger_r_seed0": {s: r["subjects"][s]["runs"][0]["r"] for s in subjects},
            }
            for k, r in runs.items()
        },
        "ranking_by_validation": sorted(runs, key=lambda k: -runs[k]["val_score"]),
    }
    (res / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(table)


if __name__ == "__main__":
    main()
