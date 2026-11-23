"""Print every results table of METHODS.md from the saved JSON files, so each number can be traced and re-derived.

Usage:
    python scripts/report_tables.py > results/report_tables.md
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
from pathlib import Path

R = Path("results")
SUBJECTS = ("1", "2", "3")


def load(path: str):
    p = R / path
    return json.loads(p.read_text()) if p.exists() else None


def benchmark() -> None:
    m = load("metrics.json")
    if not m:
        return
    print("| # | Approach | Validation r | S1 | S2 | S3 | Test r | SD |")
    print("|---|---|---|---|---|---|---|---|")
    rows = sorted(m["approaches"].items(), key=lambda kv: -kv[1]["score"])
    for _, a in rows:
        subj = " | ".join(f"{a['per_subject'][s]:.3f}" for s in SUBJECTS)
        print(f"| {a['number']} | {a['label']} | {a['val_score']:.3f} | {subj} | {a['score']:.3f} | {a['sd']:.3f} |")
    print()


def validation_studies() -> None:
    rows = {}
    for p in sorted(R.glob("validation_study*.json")):
        for name, r in json.loads(p.read_text()).items():
            if len(r.get("seeds", [])) >= 3:
                rows[name] = (p.name, r)
    if not rows:
        return
    print("| Config key | File | Validation r | S1 | S2 | S3 | Options |")
    print("|---|---|---|---|---|---|---|")
    for name, (fname, r) in sorted(rows.items(), key=lambda kv: -kv[1][1]["mean_val"]):
        per = " | ".join(f"{r['subjects'][s]['val']:.3f}" for s in SUBJECTS)
        print(f"| {name} | {fname} | {r['mean_val']:.3f} | {per} | {r.get('options', {})} |")
    print()


def kalman() -> None:
    k = load("kalman_study.json")
    if not k:
        return
    keys = sorted(k["mean"], key=lambda key: -k["mean"][key])
    print("| Variant | S1 | S2 | S3 | Mean |")
    print("|---|---|---|---|---|")
    for key in keys:
        print(f"| {key} | " + " | ".join(f"{k[s][key]:.3f}" for s in SUBJECTS) + f" | {k['mean'][key]:.3f} |")
    print()


def replicas() -> None:
    rep = load("offline/replicas.json")
    if not rep:
        return
    print("| Variant | S1 | S2 | S3 | Mean |")
    print("|---|---|---|---|---|")
    for key in rep["mean"]:
        print(f"| {key} | " + " | ".join(f"{rep[s][key]['score']:.3f}" for s in SUBJECTS)
              + f" | {rep['mean'][key]:.3f} |")


def _conditions(name: str) -> None:
    d = load(f"offline/{name}.json")
    if not d:
        return
    print("| Condition | S1 | S2 | S3 | Mean |")
    print("|---|---|---|---|---|")
    for key in d["mean"]:
        vals = [d[s][key]["score"] if isinstance(d[s][key], dict) else d[s][key] for s in SUBJECTS]
        print(f"| {key} | " + " | ".join(f"{v:.3f}" for v in vals) + f" | {d['mean'][key]:.3f} |")


def wiener() -> None:
    _conditions("wiener")


def smoothing() -> None:
    _conditions("smoothing")


def selection() -> None:
    sel = load("offline/selection.json")
    if not sel:
        return
    print("| Subject | Seed | Epochs chosen on validation | Test r there | Best test epoch | Test r there |")
    print("|---|---|---|---|---|---|")
    for s in SUBJECTS:
        for r in sel[s]:
            print(f"| {s} | {r['seed']} | {r['epochs_chosen_on_validation']} | {r['at_validation_choice']:.3f} | "
                  f"{r['best_epoch_on_test']} | {r['best_on_test']:.3f} |")
    print(f"| mean | | | {sel['mean']['at_validation_choice']:.3f} | | {sel['mean']['best_on_test']:.3f} |")


def diagnostics() -> None:
    d = load("offline/diagnostics_sub3.json")
    if not d:
        return
    rep = load("offline/replicas.json")
    print("| Replica | With every electrode (as published) | Without electrode 49 |")
    print("|---|---|---|")
    for key, base in (("liang_offline_without_e49", "liang_offline"), ("flamary_offline_without_e49", "flamary_offline")):
        if key in d:
            print(f"| {base} | {rep['3'][base]['score']:.3f} | {d[key]['score']:.3f} |")
    print()


def robustness() -> None:
    d = load("robustness.json")
    if not d:
        return
    print("| Approach | Subject | Electrodes used | Lost electrode | Intact | Lost |")
    print("|---|---|---|---|---|---|")
    for key, rows in d.items():
        for s in SUBJECTS:
            r = rows[s]
            print(f"| {key} | {s} | {r['n_electrodes']} | {r['lost_electrode']} | {r['intact']:.3f} | "
                  f"{r['one_electrode_lost']:.3f} |")
        print(f"| {key} | mean | | | {rows['mean']['intact']:.3f} | {rows['mean']['one_electrode_lost']:.3f} |")
    print()


SECTIONS = {"BENCHMARK": benchmark, "VALIDATION": validation_studies, "ROBUSTNESS": robustness, "KALMAN": kalman,
            "REPLICAS": replicas, "DIAGNOSTICS": diagnostics, "WIENER": wiener, "SELECTION": selection,
            "SMOOTHING": smoothing}


def render(fn) -> str:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        fn()
    return buf.getvalue().strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fill", help="replace {{SECTION}} placeholders in this markdown file with the tables")
    args = ap.parse_args()
    if args.fill:
        path = Path(args.fill)
        text = path.read_text()
        for name, fn in SECTIONS.items():
            text = text.replace("{{" + name + "}}", render(fn))
        path.write_text(text)
        return
    for name, fn in SECTIONS.items():
        print(f"## {name}\n\n{render(fn)}\n")
    print(f"(generated from the JSON files in {R}/; numbers rounded to 3 decimals)")


if __name__ == "__main__":
    main()
