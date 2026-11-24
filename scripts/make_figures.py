"""Static figures for the README (run after summarise.py).

* approaches.png   blind-test accuracy of every approach (single models, mean ± SD over seeds)
* traces_*.png     recorded vs decoded flexion of all five fingers
* gates_*.png      what the gate of a gated decoder does over time

Usage:
    python scripts/make_figures.py --subject 3 --start 60 --seconds 16
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

from ecogdecode.data import FINGERS  # noqa: E402
from ecogdecode.provenance import record_test_look  # noqa: E402

STEP_S = 0.04
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
RECORDED, DECODED = "#52514e", "#2a78d6"
FAMILY_COLORS = {"competition": "#a8a69e", "classical": "#1baf7a", "deep": "#2a78d6", "new": "#eb6834"}
FAMILY_NAMES = {
    "competition": "Competition entries (live re-implementation)",
    "classical": "Classical",
    "deep": "Deep sequence models",
    "new": "New in this project",
}

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK_2,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "text.color": INK,
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titlesize": 10.5,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
    }
)


def causal_ema(x: np.ndarray, tau_ms: float) -> np.ndarray:
    """Causal exponential smoothing, as a live display would apply (display only; tau 0 = none)."""
    if tau_ms <= 0:
        return x
    a = 1 - np.exp(-STEP_S * 1000 / tau_ms)
    out = np.empty_like(x)
    out[0] = x[0]
    for t in range(1, len(x)):
        out[t] = out[t - 1] + a * (x[t] - out[t - 1])
    return out


def load_pred(res: Path, name: str, subject: str) -> tuple[np.ndarray, np.ndarray]:
    true = np.load(res / "preds" / f"truth_sub{subject}.npy")
    pred = np.load(res / "preds" / f"{name}_sub{subject}.npy")
    return true, pred  # pred[0] is the last training step, pred[1:] aligns with true


def approaches_figure(metrics: dict, out: Path) -> None:
    rows = [
        (
            m["number"] + "  " + m["label"].split(" (")[0],
            m["score"],
            m["sd"],
            list(m["per_subject"].values()),
            FAMILY_COLORS[m["family"]],
        )
        for m in metrics["approaches"].values()
    ]
    rows.sort(key=lambda r: r[1])
    fig, ax = plt.subplots(figsize=(8.8, 0.42 * len(rows) + 2.2))
    for i, (_label, value, sd, subj, color) in enumerate(rows):
        ax.barh(i, value, height=0.62, color=color, zorder=2)
        if sd > 0:
            ax.errorbar(value, i, xerr=sd, color=INK, lw=1.2, capsize=2.5, zorder=4)
        ax.scatter(subj, [i] * len(subj), s=16, color=INK, edgecolor=SURFACE, linewidth=1.0, zorder=3)
        ax.text(0.012, i, f"{value:.2f}", va="center", fontsize=9, color=SURFACE, fontweight="bold", zorder=4)
    ax.axvline(0.46, color=INK_2, lw=1, ls=(0, (4, 3)), zorder=1)
    ax.text(0.46 - 0.005, -0.95, "2008 winner as submitted (0.46)", fontsize=8, color=INK_2, ha="right")
    ax.set_yticks(range(len(rows)), [r[0] for r in rows], fontsize=9)
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(-1.1, len(rows) - 0.3)
    ax.set_xlim(0, max(max(r[3]) for r in rows) + 0.04)
    ax.set_xlabel("Test r of one model (mean over four fingers)")
    ax.xaxis.grid(True, color=GRID, lw=0.8, zorder=0)
    handles = [Line2D([], [], color=c, lw=8, label=FAMILY_NAMES[f]) for f, c in FAMILY_COLORS.items()]
    handles += [
        Line2D([], [], marker="o", color=INK, lw=0, markersize=5, label="Individual subjects"),
        Line2D([], [], marker="|", color=INK, lw=1.2, markersize=8, label="± SD over 3 training seeds"),
    ]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.3, -0.1), ncol=2, frameon=False, fontsize=8.5)
    fig.suptitle(
        "Live finger decoding from ECoG: every approach under one protocol",
        x=0.01,
        ha="left",
        fontsize=11.5,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(out / "approaches.png", dpi=200)
    plt.close(fig)


def traces_figure(
    res: Path,
    metrics: dict,
    name: str,
    label: str,
    subject: str,
    start: float,
    seconds: float,
    out: Path,
    smooth_ms: float = 0.0,
) -> None:
    true, pred = load_pred(res, name, subject)
    pred = causal_ema(pred, smooth_ms)[1:]
    r = metrics["approaches"][name]["per_finger_r_seed0"][subject]
    a, b = int(start / STEP_S), int((start + seconds) / STEP_S)
    t = np.arange(a, b) * STEP_S
    fig, axes = plt.subplots(5, 1, figsize=(7.4, 7.0), sharex=True, gridspec_kw={"hspace": 0.55})
    for k, ax in enumerate(axes):
        ax.plot(t, true[a:b, k], color=RECORDED, lw=1.6, label="Recorded (data glove)")
        ax.plot(
            t,
            pred[a:b, k],
            color=DECODED,
            lw=1.8,
            label="Decoded from ECoG ("
            + (f"causal {smooth_ms:.0f} ms smoothing)" if smooth_ms > 0 else "raw output every 40 ms)"),
        )
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        ax.text(
            0.0,
            1.02,
            f"{FINGERS[k].capitalize()}   r = {r[k]:.2f}" + ("  (not scored)" if k == 3 else ""),
            transform=ax.transAxes,
            fontsize=9,
            color=INK,
            va="bottom",
        )
    axes[0].legend(loc="lower right", frameon=False, ncol=2, fontsize=8.5, bbox_to_anchor=(1, 1.0))
    axes[-1].set_xlabel("Time in the blind test set (s)")
    fig.suptitle(
        f"Subject {subject}, {label}: recorded vs decoded flexion", x=0.02, ha="left", fontsize=11, fontweight="bold"
    )
    fig.subplots_adjust(left=0.04, right=0.98, top=0.88, bottom=0.08)
    fig.savefig(out / f"traces_sub{subject}_{name}.png", dpi=200)
    plt.close(fig)


def gates_figure(res: Path, name: str, subject: str, start: float, seconds: float, out: Path) -> None:
    gates = np.load(res / "gates" / f"{name}_sub{subject}.npy")
    true = np.load(res / "preds" / f"truth_sub{subject}.npy")
    a, b = int(start / STEP_S), int((start + seconds) / STEP_S)
    t = np.arange(a, b) * STEP_S
    corr = np.array([[np.corrcoef(gates[:, e], true[:, k])[0, 1] for k in range(5)] for e in range(gates.shape[1])])
    fig = plt.figure(figsize=(9.2, 4.6))
    ax1 = fig.add_axes([0.08, 0.52, 0.61, 0.33])
    ax2 = fig.add_axes([0.08, 0.11, 0.61, 0.33], sharex=ax1)
    ax3 = fig.add_axes([0.78, 0.11, 0.2, 0.74])
    ax1.imshow(
        gates[a:b].T,
        aspect="auto",
        cmap="Blues",
        vmin=0,
        vmax=1,
        interpolation="nearest",
        extent=(t[0], t[-1], gates.shape[1] - 0.5, -0.5),
    )
    ax1.set_yticks(range(gates.shape[1]), [f"E{e + 1}" for e in range(gates.shape[1])])
    ax1.set_title("Gate weight of each expert")
    plt.setp(ax1.get_xticklabels(), visible=False)
    span = true[a:b].max(axis=0) - true[a:b].min(axis=0) + 1e-9
    for k in range(5):
        ax2.plot(t, (true[a:b, k] - true[a:b, k].min()) / span[k] + k * 1.1, color=RECORDED, lw=1.2)
    ax2.set_yticks([k * 1.1 + 0.5 for k in range(5)], [f.capitalize() for f in FINGERS])
    ax2.tick_params(axis="y", length=0)
    ax2.set_xlabel("Time in the blind test set (s)")
    ax2.set_title("Recorded finger flexion")
    lim = np.nanmax(np.abs(corr))
    ax3.imshow(corr, cmap="RdBu_r", vmin=-lim, vmax=lim)
    ax3.set_xticks(range(5), [f[0].upper() for f in FINGERS])
    ax3.set_yticks(range(gates.shape[1]), [f"E{e + 1}" for e in range(gates.shape[1])])
    for e in range(corr.shape[0]):
        for k in range(5):
            ax3.text(
                k,
                e,
                f"{corr[e, k]:.2f}",
                ha="center",
                va="center",
                fontsize=7,
                color=SURFACE if abs(corr[e, k]) > 0.6 * lim else INK,
            )
    ax3.set_title("Gate vs finger (r)")
    fig.suptitle(
        f"Subject {subject}: what the low-frequency (LMP, mu, beta) gate does",
        x=0.02,
        y=0.99,
        ha="left",
        fontsize=11,
        fontweight="bold",
    )
    fig.savefig(out / f"gates_sub{subject}_{name}.png", dpi=200)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--subject", default="3")
    ap.add_argument("--start", type=float, default=60.0, help="window start within the test set (s)")
    ap.add_argument("--seconds", type=float, default=16.0)
    ap.add_argument(
        "--approach", default="lstm_selected", help="approach shown in the traces figure (its seed-0 model)"
    )
    ap.add_argument("--gated", default="gated_lfp")
    args = ap.parse_args()
    record_test_look("figures of the benchmark (make_figures.py)")
    res = Path(args.results)
    figs = res / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    metrics = json.loads((res / "metrics.json").read_text())
    approaches_figure(metrics, figs)
    m = metrics["approaches"][args.approach]
    label = f"approach {m['number']}"
    traces_figure(res, metrics, args.approach, label, args.subject, args.start, args.seconds, figs)
    if (res / "gates" / f"{args.gated}_sub{args.subject}.npy").exists():
        gates_figure(res, args.gated, args.subject, args.start, args.seconds, figs)
    print(f"figures written to {figs}")


if __name__ == "__main__":
    main()
