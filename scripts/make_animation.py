"""Video: brain activity in, finger movement out, decoded live over the whole blind test segment.

Panels: scrolling high-gamma activity of the most movement-related electrodes (the decoder's main input), a 3-D hand
driven by the recorded data glove, the same hand driven by the live decoder (with the recorded pose as a white
outline), and scrolling recorded vs decoded traces for all five fingers.

Nothing is smoothed for display: the raster, hands and traces show the raw values every 40 ms. Each finger trace is
shifted to its own resting level and scaled by the recorded range (``handviz.display_flexion``); the scores in the
README are computed on the raw decoder output against the raw glove.

Usage:
    python scripts/make_animation.py --data-dir data --approach lstm_selected --subject 3     # whole test segment
    python scripts/make_animation.py --data-dir data --still 70                              # one frame as PNG
Writes results/figures/hand_live_subX.mp4 (1280x720, 25 fps = real time) and a GIF excerpt for the README.
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

from ecogdecode.data import load_subject  # noqa: E402
from ecogdecode.features import Standardiser  # noqa: E402
from ecogdecode.handviz import (  # noqa: E402
    BG,
    DEC,
    DEC_EDGE,
    DEC_HI,
    DEC_TRACE,
    HAND_YLIM,
    MUTED,
    PANEL,
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
from ecogdecode.pipeline import prepare  # noqa: E402
from ecogdecode.render import gif_excerpt, render_mp4  # noqa: E402


def high_gamma(data_dir: str, subject: int, n_show: int = 24):
    """High-gamma activity per electrode for the test segment, electrodes sorted by how strongly they
    follow finger movement in the TRAINING segment (display only)."""
    prep = prepare(load_subject(data_dir, subject))
    nf = prep.cfg.n_per_channel
    x = Standardiser().fit(prep.x[prep.train_rows]).transform(prep.x)
    hg = x.reshape(len(x), -1, nf)[..., 3:].mean(-1)  # mean of the three high-gamma bands
    move = np.abs(np.diff(prep.y[prep.train_rows], axis=0, prepend=prep.y[prep.train_rows][:1])).sum(1)
    tr = hg[prep.train_rows]
    corr = [np.corrcoef(tr[:, c], move)[0, 1] for c in range(hg.shape[1])]
    order = np.argsort(corr)[::-1][:n_show]
    return hg[:, order], prep.test_rows.start, hg.shape[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--results", default="results")
    ap.add_argument("--approach", default="lstm_selected", help="approach shown (its seed-0 model)")
    ap.add_argument("--subject", type=int, default=3)
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--seconds", type=float, default=None, help="default: to the end of the test segment")
    ap.add_argument("--label", default="one LSTM model (approach 5b)")
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--gif-start", type=float, default=60.0)
    ap.add_argument("--gif-seconds", type=float, default=16.0)
    ap.add_argument("--still", type=float, default=None, help="write one frame at this time (s) as PNG and stop")
    args = ap.parse_args()

    res, s = Path(args.results), args.subject
    out = res / "figures"
    true = np.load(res / "preds" / f"truth_sub{s}.npy")
    pred = np.load(res / "preds" / f"{args.approach}_sub{s}.npy")[1:]  # the file starts at the last training step
    n = min(len(true), len(pred))
    rec_n, dec_n = display_flexion(true[:n], (pred[:n],))
    hg, test_start, n_total = high_gamma(args.data_dir, s)
    total_s = n * STEP_S
    run = json.loads((res / "runs" / f"{args.approach}.json").read_text())["subjects"][str(s)]["runs"][0]
    r_dec = float(np.mean([run["r"][k] for k in SCORED]))
    a = int(round(args.start / STEP_S))
    b = n if args.seconds is None else min(n, a + int(round(args.seconds / STEP_S)))

    def build():
        fig = plt.figure(figsize=(12.8, 7.2), dpi=100, facecolor=BG)
        fig.text(0.035, 0.952, "Decoding finger movements from the human brain, live", fontsize=19, fontweight="bold")
        fig.text(
            0.035,
            0.914,
            f"Electrocorticography (ECoG)  ·  patient {s}  ·  whole blind test segment  ·  "
            f"{args.label}  ·  a new output every 40 ms from past signal only",
            fontsize=10.5,
            color=MUTED,
        )
        clock = fig.text(0.965, 0.952, "", ha="right", fontsize=15, color=TEXT)
        bar = progress_bar(fig, total_s)

        # brain activity raster: what the decoder reads
        ax_r = fig.add_axes([0.045, 0.43, 0.25, 0.36], facecolor=PANEL)
        win_r = 100  # 4 s
        img = ax_r.imshow(
            np.zeros((hg.shape[1], win_r)),
            aspect="auto",
            cmap="magma",
            vmin=-0.8,
            vmax=2.2,
            extent=(-win_r * STEP_S, 0, hg.shape[1], 0),
            interpolation="nearest",
        )
        fig.text(0.045, 0.832, "Brain activity in", fontsize=12.5, color=TEXT, fontweight="bold")
        fig.text(0.045, 0.806, "high-gamma power, standardised", fontsize=9, color=MUTED)
        ax_r.set_xlabel("seconds before now", fontsize=9)
        ax_r.set_ylabel(f"{hg.shape[1]} most movement-related\nof {n_total} electrodes", fontsize=9)
        ax_r.set_yticks([])
        for sp in ax_r.spines.values():
            sp.set_visible(False)
        ax_r.tick_params(labelsize=8.5, length=0)

        hands = []
        for x0, title, sub, colors, ghost in (
            (0.325, "Recorded movement", "data glove", (REC, REC_EDGE, REC_HI), False),
            (0.655, "Decoded from the brain", f"{args.label}  ·  r = {r_dec:.2f}", (DEC, DEC_EDGE, DEC_HI), True),
        ):
            fig.text(x0 + 0.155, 0.832, title, ha="center", fontsize=12.5, color=TEXT, fontweight="bold")
            fig.text(x0 + 0.155, 0.806, sub, ha="center", fontsize=9, color=MUTED)
            ax = hand_axes(fig, [x0, 0.385, 0.31, 0.41])
            hands.append(HandArtist(ax, *colors, ghost=ghost))
            fade_bottom(ax, HAND_YLIM[0], -0.95)

        legend = [
            Line2D([], [], color=REC_TRACE, lw=2, label="recorded"),
            Line2D([], [], color=DEC_TRACE, lw=2, label="decoded"),
            Line2D([], [], color="white", lw=1.2, label="white outline on the decoded hand = recorded pose"),
            Line2D([], [], color="none", label="raw output every 40 ms, no smoothing"),
        ]
        _, tt, lines = trace_panel(
            fig, [0.085, 0.075, 0.88, 0.265], [(REC_TRACE, 1.5), (DEC_TRACE, 1.4)], legend, win_steps=200
        )

        def update(i: int):
            t = a + i
            lo = test_start + t - win_r + 1
            window = hg[max(lo, 0) : test_start + t + 1]
            img.set_data(np.r_[np.full((win_r - len(window), hg.shape[1]), np.nan), window].T)
            hands[0].update(rec_n[t])
            hands[1].update(dec_n[t], ghost_flex=rec_n[t])
            set_traces(lines, tt, [rec_n, dec_n], t)
            clock.set_text(f"{(t + 1) * STEP_S:5.1f} s of {total_s:.0f} s")
            bar.set_width((t + 1) * STEP_S)

        return fig, update

    if args.still is not None:
        fig, update = build()
        update(int(round(args.still / STEP_S)) - a)
        path = out / f"hand_live_sub{s}_still.png"
        fig.savefig(path, facecolor=BG)
        print(f"wrote {path}")
        return
    mp4 = render_mp4(build, b - a, out / f"hand_live_sub{s}.mp4", jobs=args.jobs)
    gif = gif_excerpt(mp4, out / f"hand_live_sub{s}.gif", args.gif_start - args.start, args.gif_seconds)
    print(f"wrote {mp4} ({(b - a) * STEP_S:.0f} s) and {gif}")


if __name__ == "__main__":
    main()
