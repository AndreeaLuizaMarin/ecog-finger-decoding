"""Join the parts of a benchmark run that was split over processes into one results/runs/<key>.json.

    python scripts/run_benchmark.py --data-dir data --approaches fingerflex --subjects 1 --seeds 0 --tag s1_seed0
    ...                                                              (one process per subject and seed)
    python scripts/merge_runs.py fingerflex

Each part is results/runs/parts/<key>__<tag>.json. The merged file has the same layout as an unsplit run: per subject
the single-model runs sorted by seed, their mean and SD, and the SD over seeds of the subject-averaged score.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def merge(reports: list[dict]) -> dict:
    merged = {k: v for k, v in reports[0].items() if k not in ("subjects", "val_score", "score", "sd", "seconds")}
    merged["seconds"] = round(sum(r.get("seconds", 0.0) for r in reports), 1)
    runs: dict[str, dict[int, dict]] = {}
    for report in reports:
        for s, entry in report["subjects"].items():
            for run in entry["runs"]:
                runs.setdefault(s, {})[int(run["selected"].get("seed", 0))] = run
    seeds = sorted(set.intersection(*(set(r) for r in runs.values())))
    subjects = {}
    for s in sorted(runs, key=int):
        chosen = [runs[s][seed] for seed in seeds]
        scores = [r["score"] for r in chosen]
        subjects[s] = {
            "val_score": float(np.mean([r["val_score"] for r in chosen])),
            "score": float(np.mean(scores)),
            "sd": float(np.std(scores)),
            "r": np.mean([r["r"] for r in chosen], axis=0).tolist(),
            "runs": chosen,
        }
    seed_means = [np.mean([subjects[s]["runs"][i]["score"] for s in subjects]) for i in range(len(seeds))]
    merged["protocol"] = {**merged.get("protocol", {}), "seeds": seeds}
    merged.update(
        subjects=subjects,
        val_score=float(np.mean([e["val_score"] for e in subjects.values()])),
        score=float(np.mean([e["score"] for e in subjects.values()])),
        sd=float(np.std(seed_means)),
    )
    return merged


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("keys", nargs="+")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    runs_dir = Path(args.out) / "runs"
    for key in args.keys:
        parts = sorted((runs_dir / "parts").glob(f"{key}__*.json"))
        if not parts:
            raise SystemExit(f"no parts for {key} in {runs_dir / 'parts'}")
        merged = merge([json.loads(p.read_text()) for p in parts])
        merged["parts"] = [p.name for p in parts]
        (runs_dir / f"{key}.json").write_text(json.dumps(merged, indent=2))
        per_subject = "  ".join(f"S{s} {e['score']:.3f}" for s, e in merged["subjects"].items())
        print(
            f"{key}: test {merged['score']:.3f} ± {merged['sd']:.3f} over seeds {merged['protocol']['seeds']} "
            f"({per_subject})"
        )


if __name__ == "__main__":
    main()
