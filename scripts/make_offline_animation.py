"""Video: recorded movement next to a live decoder and an offline decoder, on the same blind test data.

Panels: three 3-D hands (data glove; live decoder; offline decoder, each decoded hand with the recorded pose as a
white outline) and scrolling traces of all five fingers. The live decoder only ever sees signal up to the moment it
decodes; the offline decoder may use signal after that moment: a published method run as published, or approach 5b
given every offline liberty (``scripts/offline_lstm.py``).

Nothing is smoothed for display: hands and traces show the raw output every 40 ms. Each trace is shifted to its own
resting level and scaled by the recorded range (``handviz.display_flexion``); the r values in the labels are for the
raw decoder outputs against the raw glove, for this patient, at 1 kHz (competition metric).

Usage:
    python scripts/make_offline_animation.py --subject 3                 # whole test segment (200 s)
    python scripts/make_offline_animation.py --subject 3 --still 70      # one frame as PNG, for checking
    python scripts/make_offline_animation.py --offline lstm_selected_offline --name lstm_live_vs_offline_sub3
Writes results/figures/<name>.mp4 (1280x720, real time; default name live_vs_offline_subX) and a GIF excerpt for
the README.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from ecogdecode.handviz import (  # noqa: E402
    BG,
    HAND_YLIM,
    LIVE,
    LIVE_EDGE,
    LIVE_HI,
    LIVE_TRACE,
    MUTED,
    OFF,
    OFF_EDGE,
    OFF_HI,
    OFF_TRACE,
    REC,
    REC_EDGE,
    REC_HI,
    REC_TRACE,
    SCORED,
    STEP_S,
    TEXT,
    HandArtist,
    display_flexion,
    fade_bottom,
    hand_axes,
    progress_bar,
    set_traces,
    trace_panel,
)
from ecogdecode.provenance import record_test_look  # noqa: E402
from ecogdecode.render import gif_excerpt, render_mp4  # noqa: E402

OFFLINE_LABELS = {
    "liang_offline": "2008 competition winner, as published",
    "flamary_offline": "2008 second place, as published",
    "lstm_selected_offline": "the same LSTM, given every offline liberty",
    "fingerflex_published": "FingerFlex, as published",
    "fingerflex_validation_epoch": "FingerFlex, epoch chosen on validation",
}


def offline_score(res: Path, key: str, s: int) -> float:
    """Stored test score (competition metric) of the offline decoder's seed-0 model for patient ``s``."""
    if key == "lstm_selected_offline":
        runs = json.loads((res / "offline" / "lstm_offline.json").read_text())["subjects"][str(s)]["runs"]
        return next(r["score"] for r in runs if r["seed"] == 0)
    if key.startswith("fingerflex_"):
        report = json.loads((res / "offline" / "fingerflex.json").read_text())
        return report[key.removeprefix("fingerflex_")]["subjects"][str(s)]["score"]
    return json.loads((res / "offline" / "replicas.json").read_text())[str(s)][key]["score"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--live", default="lstm_selected", help="benchmark approach (its seed-0 model)")
    ap.add_argument("--offline", default="liang_offline", help="offline replica in results/offline/preds")
    ap.add_argument("--subject", type=int, default=3)
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--seconds", type=float, default=None, help="default: to the end of the test segment")
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--gif-start", type=float, default=60.0)
    ap.add_argument("--gif-seconds", type=float, default=16.0)
    ap.add_argument("--still", type=float, default=None, help="write one frame at this time (s) as PNG and stop")
    ap.add_argument("--name", default=None, help="output file name without extension (default live_vs_offline_subX)")
    args = ap.parse_args()
    record_test_look("video, live versus offline (make_offline_animation.py)")

    res, s = Path(args.results), args.subject
    name = args.name or f"live_vs_offline_sub{s}"
    out = res / "figures"
    true = np.load(res / "preds" / f"truth_sub{s}.npy")
    live = np.load(res / "preds" / f"{args.live}_sub{s}.npy")[1:]  # the file starts at the last training step
    offline = np.load(res / "offline" / "preds" / f"{args.offline}_sub{s}.npy")
    n = min(len(true), len(live), len(offline))
    rec_n, live_n, off_n = display_flexion(true[:n], (live[:n], offline[:n]))
    total_s = n * STEP_S

    run = json.loads((res / "runs" / f"{args.live}.json").read_text())["subjects"][str(s)]["runs"][0]
    r_live = float(np.mean([run["r"][k] for k in SCORED]))
    r_off = offline_score(res, args.offline, s)
    a = int(round(args.start / STEP_S))
    b = n if args.seconds is None else min(n, a + int(round(args.seconds / STEP_S)))

    def build():
        fig = plt.figure(figsize=(12.8, 7.2), dpi=100, facecolor=BG)
        fig.text(0.035, 0.952, "The same finger movements, decoded live and offline", fontsize=19, fontweight="bold")
        fig.text(
            0.035,
            0.914,
            f"Electrocorticography (ECoG)  ·  patient {s}  ·  whole blind test segment  ·  "
            "live = past signal only  ·  offline = may use signal after each moment",
            fontsize=10.5,
            color=MUTED,
        )
        clock = fig.text(0.965, 0.952, "", ha="right", fontsize=15, color=TEXT)
        bar = progress_bar(fig, total_s)

        hands = []
        panels = (
            ("Recorded movement", "data glove", (REC, REC_EDGE, REC_HI), False),
            ("Live decoder", f"one LSTM, approach 5b  ·  r = {r_live:.2f}", (LIVE, LIVE_EDGE, LIVE_HI), True),
            (
                "Offline decoder",
                f"{OFFLINE_LABELS.get(args.offline, args.offline)}  ·  r = {r_off:.2f}",
                (OFF, OFF_EDGE, OFF_HI),
                True,
            ),
        )
        for i, (title, sub, colors, ghost) in enumerate(panels):
            cx = 0.02 + i * 0.325 + 0.155
            fig.text(cx, 0.832, title, ha="center", fontsize=12.5, color=TEXT, fontweight="bold")
            fig.text(cx, 0.806, sub, ha="center", fontsize=9, color=MUTED)
            ax = hand_axes(fig, [0.02 + i * 0.325, 0.385, 0.31, 0.41])
            hands.append(HandArtist(ax, *colors, ghost=ghost))
            fade_bottom(ax, HAND_YLIM[0], -0.95)

        legend = [
            Line2D([], [], color=REC_TRACE, lw=2, label="recorded"),
            Line2D([], [], color=LIVE_TRACE, lw=2, label="live"),
            Line2D([], [], color=OFF_TRACE, lw=2, label="offline"),
            Line2D([], [], color="white", lw=1.2, label="white outline on a decoded hand = recorded pose"),
            Line2D([], [], color="none", label="raw output every 40 ms, no smoothing"),
        ]
        _, tt, lines = trace_panel(
            fig,
            [0.085, 0.075, 0.88, 0.265],
            [(REC_TRACE, 1.5), (OFF_TRACE, 1.3), (LIVE_TRACE, 1.4)],
            legend,
            win_steps=200,
        )

        def update(i: int):
            t = a + i
            hands[0].update(rec_n[t])
            hands[1].update(live_n[t], ghost_flex=rec_n[t])
            hands[2].update(off_n[t], ghost_flex=rec_n[t])
            set_traces(lines, tt, [rec_n, off_n, live_n], t)
            clock.set_text(f"{(t + 1) * STEP_S:5.1f} s of {total_s:.0f} s")
            bar.set_width((t + 1) * STEP_S)

        return fig, update

    if args.still is not None:
        fig, update = build()
        update(int(round(args.still / STEP_S)) - a)
        path = out / f"{name}_still.png"
        fig.savefig(path, facecolor=BG)
        print(f"wrote {path}")
        return
    mp4 = render_mp4(build, b - a, out / f"{name}.mp4", jobs=args.jobs)
    gif = gif_excerpt(mp4, out / f"{name}.gif", args.gif_start - args.start, args.gif_seconds)
    print(f"wrote {mp4} ({(b - a) * STEP_S:.0f} s) and {gif}")


if __name__ == "__main__":
    main()
