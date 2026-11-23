"""Figures for the post-hoc and round-8 analyses (run after posthoc.py and the studies they read).

* uncertainty.png      paired differences between approaches with 95 % block-bootstrap intervals
* fingers_sub3.png     which finger: recorded-vs-recorded (the glove's own coupling) next to decoded-vs-recorded
* onsets_sub3.png      training data: high gamma around every movement onset, one row per movement
* lag.png              correlation against time shift: live decoders trail the glove, an offline replica does not
* calibration.png      test r against minutes of calibration data (if results/calibration.json exists)
* electrode_loss.png   test r as electrodes fail, best first and at random (if results/robustness_curve.json exists)

Colours follow the README figures (reference palette: categorical slots in fixed order, a blue-red diverging map
with a neutral midpoint).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from ecogdecode.data import FINGERS, SCORED_FINGERS, SUBJECTS, load_subject  # noqa: E402
from ecogdecode.evaluation import pearson_r, to_sample_rate  # noqa: E402
from ecogdecode.offline import upsample_linear  # noqa: E402

RES = Path("results")
FIG = RES / "figures"
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]  # categorical, fixed order
DIVERGING = LinearSegmentedColormap.from_list("blue_red", ["#2a78d6", "#f0efec", "#e34948"])
LAGS_MS = np.arange(-400, 401, 10)
PRETTY = {
    "lstm_selected": "5b LSTM, selected electrodes",
    "lstm": "5 LSTM",
    "mamba": "7 Mamba",
    "tcn": "8b TCN",
    "s4d": "6 S4D",
    "wiener": "3 Wiener filter",
    "liang_bougrain": "1 winner, live",
    "kalman": "4 Kalman filter",
    "fingerflex": "14 FingerFlex, causal",
    "possm": "15 POSSM-style",
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
for family in ("Inter", "DejaVu Sans"):
    if any(family in f.name for f in matplotlib.font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = family
        break


def uncertainty(post: dict) -> None:
    rows = post["paired_differences"][::-1]
    fig, ax = plt.subplots(figsize=(9.0, 0.42 * len(rows) + 1.6))
    for i, row in enumerate(rows):
        lo, hi = row["ci95"]
        resolved = lo > 0 or hi < 0
        ax.plot([lo, hi], [i, i], color=INK if resolved else INK_2, lw=2, solid_capstyle="round", zorder=2)
        ax.scatter(
            row["difference"],
            i,
            s=46,
            color=SLOTS[0] if resolved else SURFACE,
            edgecolor=SLOTS[0] if resolved else INK_2,
            linewidth=1.6,
            zorder=3,
        )
    ax.axvline(0, color=INK_2, lw=1, zorder=1)
    ax.set_yticks(
        range(len(rows)),
        [f"{PRETTY.get(r['a'], r['a'])}  minus  {PRETTY.get(r['b'], r['b'])}" for r in rows],
        fontsize=9,
    )
    ax.set_xlabel("Difference in test r")
    ax.xaxis.grid(True, color=GRID, lw=0.8, zorder=0)
    ax.tick_params(axis="y", length=0)
    fig.suptitle(
        "Which differences survive resampling the test data?", x=0.01, ha="left", fontsize=10.5, fontweight="bold"
    )
    fig.text(
        0.01,
        0.905,
        "Seed-0 models; 95 % intervals from resampling 4 s blocks of the test segment; filled: the "
        "interval excludes zero",
        fontsize=8.5,
        color=INK_2,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(FIG / "uncertainty.png", dpi=200)
    plt.close(fig)


def fingers(post: dict, subject: str = "3", key: str = "lstm_selected") -> None:
    truth = np.load(RES / "preds" / f"truth_sub{subject}.npy")
    glove = np.array([[pearson_r(truth[:, i], truth[:, j])[0] for j in range(5)] for i in range(5)])
    decoded = np.array(post["subjects"][subject]["approaches"][key]["cross_finger_r"])
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 4.1))
    for ax, mat, title, ylabel in [
        (axes[0], glove, "The glove: recorded against recorded", "recorded finger"),
        (axes[1], decoded, "Approach 5b: decoded against recorded", "decoded finger"),
    ]:
        ax.imshow(mat, cmap=DIVERGING, vmin=-1, vmax=1)
        for i in range(5):
            for j in range(5):
                ax.text(
                    j,
                    i,
                    f"{mat[i, j]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=8.5,
                    color=SURFACE if abs(mat[i, j]) > 0.55 else INK,
                )
        ax.set_xticks(range(5), FINGERS, fontsize=8.5)
        ax.set_yticks(range(5), FINGERS, fontsize=8.5)
        ax.set_xlabel("recorded finger")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=10)
        ax.spines[:].set_visible(False)
        ax.tick_params(length=0)
    fig.suptitle(
        f"Patient {subject}: off-diagonal decoded values track the glove's own coupling (index-middle, ring-little)",
        x=0.01,
        ha="left",
        fontsize=10.5,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(FIG / f"fingers_sub{subject}.png", dpi=200)
    plt.close(fig)


def _profile(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    out = []
    for lag in LAGS_MS:
        p, t = (pred[lag:], true[: len(true) - lag]) if lag >= 0 else (pred[:lag], true[-lag:])
        out.append(pearson_r(p, t)[list(SCORED_FINGERS)].mean())
    return np.asarray(out)


def lag() -> None:
    series = {
        "5b LSTM, selected electrodes (live)": ("live", "lstm_selected"),
        "3 Wiener filter (live)": ("live", "wiener"),
        "2008 winner, offline replica": ("offline", "liang_offline"),
    }
    profiles = {name: [] for name in series}
    for s in SUBJECTS:
        true = load_subject("data", s).test_flex.astype(np.float64)
        times = np.load(RES / "preds" / f"times_sub{s}.npy")
        for name, (kind, key) in series.items():
            if kind == "live":
                pred = to_sample_rate(np.load(RES / "preds" / f"{key}_sub{s}.npy").astype(np.float64), times, len(true))
            else:  # offline replicas are saved every 40 ms (25 Hz); interpolate back to 1 kHz
                p25 = np.load(RES / "offline" / "preds" / f"{key}_sub{s}.npy").astype(np.float64)
                pred = upsample_linear(p25, np.arange(len(p25)) * 40, len(true))
            profiles[name].append(_profile(pred, true))
    fig, ax = plt.subplots(figsize=(7.6, 3.6))
    for (name, rows), color in zip(profiles.items(), SLOTS):
        mean = np.mean(rows, axis=0)
        best = LAGS_MS[np.argmax(mean)]
        ax.plot(LAGS_MS, mean, color=color, lw=2, label=f"{name}: peak at {best:+d} ms")
        ax.scatter([best], [mean.max()], s=36, color=color, edgecolor=SURFACE, linewidth=1.5, zorder=3)
    ax.axvline(0, color=INK_2, lw=1, ls=(0, (4, 3)))
    ax.set_xlabel("Shift of the decoded trace (ms); positive = the decoder trails the glove")
    ax.set_ylabel("Test r (mean of 3 patients)")
    ax.yaxis.grid(True, color=GRID, lw=0.8)
    ax.legend(frameon=False, fontsize=8.5, loc="lower center")
    ax.set_title("Live decoders trail the movement by 70-90 ms; the offline replica does not", pad=10)
    fig.tight_layout()
    fig.savefig(FIG / "lag.png", dpi=200)
    plt.close(fig)


def _onsets(flex: np.ndarray, rows: slice, before: int = 25, after: int = 50) -> np.ndarray:
    """Movement onsets in ``rows``: the flexion crosses 30 % of its range after at least 1 s below it."""
    lo, hi = np.percentile(flex[rows], [20, 99])
    above = flex > lo + 0.3 * (hi - lo)
    t = np.arange(rows.start + before, rows.stop - after)
    keep = above[t] & ~above[t - 1] & np.array([not above[i - before : i].any() for i in t])
    return t[keep]


def onsets(subject: int = 3, before: int = 25, after: int = 50) -> None:
    """Training data only: high-gamma (75-115 Hz) of the electrode most related to each finger, every movement onset
    as one row (in recording order), with the mean response and the mean flexion above."""
    from ecogdecode.pipeline import prepare

    prep = prepare(load_subject("data", subject, require_test_labels=False))
    rows, npc = prep.train_rows, prep.cfg.n_per_channel
    band = prep.cfg.names.index("75-115Hz")
    hg = prep.x[:, band::npc]
    hg = (hg - hg[rows].mean(0)) / hg[rows].std(0)
    fingers = [f for f in SCORED_FINGERS]
    t_ms = np.arange(-before, after) * 40
    fig, axes = plt.subplots(2, len(fingers), figsize=(11, 5.6), height_ratios=[1, 1.6], sharex=True)
    for col, f in zip(axes.T, fingers):
        flex = prep.y[:, f]
        r = [abs(np.corrcoef(hg[rows, e], flex[rows])[0, 1]) for e in range(hg.shape[1])]
        e = int(np.argmax(r))
        on = _onsets(flex, rows, before, after)
        trials = np.stack([hg[o - before : o + after, e] for o in on])
        moves = np.stack([flex[o - before : o + after] for o in on])
        span = moves.max() - moves.min()
        col[0].plot(t_ms, trials.mean(0), color=SLOTS[0], lw=2, label="high gamma (z)")
        col[0].plot(
            t_ms,
            (moves.mean(0) - moves.min()) / span * trials.mean(0).max(),
            color=INK_2,
            lw=1.5,
            label="flexion (scaled)",
        )
        col[0].axvline(0, color=INK_2, lw=1, ls=(0, (4, 3)))
        col[0].set_title(f"{FINGERS[f]}: electrode {e}", fontsize=10)
        col[0].yaxis.grid(True, color=GRID, lw=0.8)
        lim = np.percentile(np.abs(trials), 98)
        col[1].imshow(
            trials,
            aspect="auto",
            cmap=DIVERGING,
            vmin=-lim,
            vmax=lim,
            interpolation="nearest",
            extent=(t_ms[0] - 20, t_ms[-1] + 20, len(on), 0),
        )
        col[1].axvline(0, color=INK, lw=1)
        col[1].yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
        col[1].set_xlabel("ms from movement onset")
        col[1].spines[:].set_visible(False)
    axes[0, 0].set_ylabel("mean")
    axes[1, 0].set_ylabel("one movement per row (recording order)", fontsize=9)
    axes[0, 0].legend(frameon=False, fontsize=8, loc="upper left")
    fig.suptitle(
        f"Patient {subject}, training data: high gamma rises before the glove registers the movement",
        x=0.01,
        ha="left",
        fontsize=10.5,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(FIG / f"onsets_sub{subject}.png", dpi=200)
    plt.close(fig)


CAL_NAMES = {
    "wiener": "3 Wiener filter",
    "lstm": "5 LSTM, all electrodes",
    "lstm_selected": "5b LSTM, selected electrodes",
    "lstm_others": "LSTM pretrained on the other two patients",
}


def calibration() -> None:
    path = RES / "calibration.json"
    if not path.exists():
        return
    cal = json.loads(path.read_text())
    minutes = {"1.0": 1.0, "2.0": 2.0, "4.0": 4.0, "all": 6.5}
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    for (key, name), color in zip(CAL_NAMES.items(), SLOTS):
        rows = cal.get(key, {})
        xs = [minutes[m] for m in minutes if m in rows]
        ys = [rows[m]["score"] for m in minutes if m in rows]
        sd = [rows[m]["sd"] for m in minutes if m in rows]
        if not xs:
            continue
        ax.errorbar(xs, ys, yerr=sd, color=color, lw=2, marker="o", markersize=5, capsize=2.5, label=name)
    ax.set_xticks(list(minutes.values()), ["1", "2", "4", "6.5 (all)"])
    ax.set_xlabel("Minutes of training data from the patient (the last minutes before the test segment)")
    ax.set_ylabel("Test r (mean of 3 patients)")
    ax.yaxis.grid(True, color=GRID, lw=0.8)
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    ax.set_title("How much calibration data does a new patient need?", pad=10)
    fig.tight_layout()
    fig.savefig(FIG / "calibration.png", dpi=200)
    plt.close(fig)


LOSS_NAMES = {
    "lstm": "5 LSTM, all electrodes",
    "lstm_selected": "5b LSTM, selected electrodes",
    "lstm_selected_dropout": "5b trained with electrode dropout",
}


def electrode_loss() -> None:
    path = RES / "robustness_curve.json"
    if not path.exists():
        return
    d = json.loads(path.read_text())["subjects"]
    subjects = sorted(d, key=int)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.9), sharey=True)
    for (key, name), color in zip(LOSS_NAMES.items(), SLOTS):
        best = np.mean([d[s][key]["best_first"] for s in subjects], axis=0)
        ax1.plot(range(len(best)), best, color=color, lw=2, marker="o", markersize=4, label=name)
        ks = sorted(d[subjects[0]][key]["random_mean"], key=int)
        mean = [np.mean([d[s][key]["random_mean"][k] for s in subjects]) for k in ks]
        ax2.plot([int(k) for k in ks], mean, color=color, lw=2, marker="o", markersize=4, label=name)
    ax1.set_xlabel("Electrodes lost, most informative first")
    ax2.set_xlabel("Electrodes lost at random (mean of 20 draws)")
    ax2.set_xscale("log", base=2)
    ax2.set_xticks([1, 2, 4, 8, 16], ["1", "2", "4", "8", "16"])
    ax1.set_ylabel("Test r (seed-0 models, mean of 3 patients)")
    for ax in (ax1, ax2):
        ax.yaxis.grid(True, color=GRID, lw=0.8)
    ax1.legend(frameon=False, fontsize=8.5, loc="lower left")
    fig.suptitle(
        "Electrode failure: fewer electrodes means each one matters more",
        x=0.01,
        ha="left",
        fontsize=10.5,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(FIG / "electrode_loss.png", dpi=200)
    plt.close(fig)


def main() -> None:
    post = json.loads((RES / "posthoc.json").read_text())
    uncertainty(post)
    fingers(post)
    lag()
    onsets()
    calibration()
    electrode_loss()


if __name__ == "__main__":
    main()
