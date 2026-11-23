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
        print(
            f"| {key} | " + " | ".join(f"{rep[s][key]['score']:.3f}" for s in SUBJECTS) + f" | {rep['mean'][key]:.3f} |"
        )


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
            print(
                f"| {s} | {r['seed']} | {r['epochs_chosen_on_validation']} | {r['at_validation_choice']:.3f} | "
                f"{r['best_epoch_on_test']} | {r['best_on_test']:.3f} |"
            )
    print(f"| mean | | | {sel['mean']['at_validation_choice']:.3f} | | {sel['mean']['best_on_test']:.3f} |")


def diagnostics() -> None:
    d = load("offline/diagnostics_sub3.json")
    if not d:
        return
    rep = load("offline/replicas.json")
    print("| Replica | With every electrode (as published) | Without electrode 49 |")
    print("|---|---|---|")
    for key, base in (
        ("liang_offline_without_e49", "liang_offline"),
        ("flamary_offline_without_e49", "flamary_offline"),
    ):
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
            print(
                f"| {key} | {s} | {r['n_electrodes']} | {r['lost_electrode']} | {r['intact']:.3f} | "
                f"{r['one_electrode_lost']:.3f} |"
            )
        print(f"| {key} | mean | | | {rows['mean']['intact']:.3f} | {rows['mean']['one_electrode_lost']:.3f} |")
    print()


def fingerflex() -> None:
    f = load("offline/fingerflex.json")
    if not f:
        return
    print("| FingerFlex, offline | S1 | S2 | S3 | Mean ± SD over seeds | Notebook metric | Epochs kept (per subject) |")
    print("|---|---|---|---|---|---|---|")
    for kind, label in (
        ("published", "as published: epoch with the best test correlation"),
        ("validation_epoch", "epoch chosen on validation"),
    ):
        k = f[kind]
        subj = " | ".join(f"{k['subjects'][s]['score']:.3f}" for s in SUBJECTS)
        epochs = "; ".join(f"S{s}: {', '.join(map(str, k['epochs'][s]))}" for s in SUBJECTS)
        print(f"| {label} | {subj} | {k['score']:.3f} ± {k['sd']:.3f} | {k['notebook_metric']:.3f} | {epochs} |")
    print()
    print(
        f"Seeds: {', '.join(map(str, f['seeds']))}. Competition metric (1 kHz, ring finger excluded); the notebook "
        "metric is the authors' own (100 Hz, all five fingers, Gaussian sigma 1 sample)."
    )
    print()


def two_by_two() -> None:
    rep, m, off = load("offline/replicas.json"), load("metrics.json"), load("offline/lstm_offline.json")
    if not (rep and m and off):
        return
    winner_offline = rep["mean"]["liang_offline"]
    winner_live = m["approaches"]["liang_bougrain"]["score"]
    lstm_live = m["approaches"]["lstm_selected"]["score"]
    print("| Method | Designed for | Scored offline | Scored live | Cost of running live |")
    print("|---|---|---|---|---|")
    print(
        f"| Liang and Bougrain, 2008 winner | offline | {winner_offline:.3f} (replica) | {winner_live:.3f} "
        f"(approach 1) | {winner_live - winner_offline:+.3f} |"
    )
    print(
        f"| LSTM on selected electrodes | live | {off['score']:.3f} ± {off['sd']:.3f} (bidirectional, zero-phase) | "
        f"{lstm_live:.3f} (approach 5b) | {lstm_live - off['score']:+.3f} |"
    )
    print()


def bootstrap() -> None:
    p = load("posthoc.json")
    if not p:
        return
    print(
        "| Approach | Test r, seed 0 | 95 % interval (test blocks resampled) | Lag at peak (ms) | Envelope r | "
        "Within-movement r |"
    )
    print("|---|---|---|---|---|---|")
    for key in sorted(p["bootstrap"], key=lambda k: -p["bootstrap"][k]["score"]):
        b = p["bootstrap"][key]
        subs = [p["subjects"][s]["approaches"][key] for s in SUBJECTS]
        lag = sum(a["best_lag_ms"] for a in subs) / 3
        env = sum(a["envelope_r"] for a in subs) / 3
        within = sum(a["within_movement_r"] for a in subs) / 3
        print(
            f"| {key} | {b['score']:.3f} | {b['ci95'][0]:.3f} to {b['ci95'][1]:.3f} | {lag:.0f} | {env:.3f} | "
            f"{within:.3f} |"
        )
    print()


def paired() -> None:
    p = load("posthoc.json")
    if not p:
        return
    print("| A | B | A minus B | 95 % interval | Share of resamples with A <= B |")
    print("|---|---|---|---|---|")
    for row in p["paired_differences"]:
        print(
            f"| {row['a']} | {row['b']} | {row['difference']:+.3f} | {row['ci95'][0]:+.3f} to {row['ci95'][1]:+.3f} "
            f"| {row['p_not_greater']:.3f} |"
        )
    print()


def masking() -> None:
    p = load("posthoc.json")
    if not p:
        return
    print("| Subject | Steps with any electrode masked | Electrode-steps masked | Electrodes ever masked |")
    print("|---|---|---|---|")
    for s in SUBJECTS:
        mk = p["subjects"][s]["masking"]
        print(
            f"| {s} | {100 * mk['fraction_of_steps_with_any']:.2f} % | {100 * mk['fraction_of_electrode_steps']:.3f} % "
            f"| {mk['electrodes_ever_masked']} |"
        )
    print()


def calibration() -> None:
    c = load("calibration.json")
    if not c:
        return
    minutes = ("1.0", "2.0", "4.0", "all")
    print("| Approach | 1 min | 2 min | 4 min | 6.5 min (all) |")
    print("|---|---|---|---|---|")
    for key, rows in c.items():
        cells = []
        for mnt in minutes:
            r = rows.get(mnt)
            cells.append(
                "" if r is None else f"{r['score']:.3f}" + ("" if len(r["seeds"]) < 2 else f" ± {r['sd']:.3f}")
            )
        print(f"| {key} | " + " | ".join(cells) + " |")
    print()


def electrode_loss() -> None:
    d = load("robustness_curve.json")
    if not d:
        return
    variants = [k for k in d["subjects"]["1"] if isinstance(d["subjects"]["1"][k], dict)]
    print("| Decoder | Intact | 1 lost (best first) | 2 | 4 | 8 | 4 lost at random | 16 lost at random |")
    print("|---|---|---|---|---|---|---|---|")
    for v in variants:
        best = [sum(d["subjects"][s][v]["best_first"][k] for s in SUBJECTS) / 3 for k in (0, 1, 2, 4, 8)]
        rnd = [sum(d["subjects"][s][v]["random_mean"][k] for s in SUBJECTS) / 3 for k in ("4", "16")]
        print(f"| {v} | " + " | ".join(f"{x:.3f}" for x in best + rnd) + " |")
    print()


SECTIONS = {
    "BENCHMARK": benchmark,
    "VALIDATION": validation_studies,
    "ROBUSTNESS": robustness,
    "KALMAN": kalman,
    "REPLICAS": replicas,
    "DIAGNOSTICS": diagnostics,
    "WIENER": wiener,
    "SELECTION": selection,
    "SMOOTHING": smoothing,
    "FINGERFLEX": fingerflex,
    "TWO_BY_TWO": two_by_two,
    "BOOTSTRAP": bootstrap,
    "PAIRED": paired,
    "MASKING": masking,
    "CALIBRATION": calibration,
    "ELECTRODE_LOSS": electrode_loss,
}


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
