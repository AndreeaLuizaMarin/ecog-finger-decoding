"""Offline replicas of the two best BCI Competition IV (data set 4) entries, as published.

These are *not* live decoders. They follow the papers as closely as the papers allow, including the steps that
use samples after the predicted moment, so that the offline-versus-live comparison (scripts/offline_study.py)
measures what a live system gives up. Every point a paper leaves open is listed in the class docstring together
with the choice made here.

* Liang and Bougrain (2012), Front. Neurosci. 6:91 - competition winner, r = 0.46.
* Flamary and Rakotomamonjy (2012), Front. Neurosci. 6:29 - second place, r = 0.42.
"""

from __future__ import annotations

import numpy as np
from scipy import interpolate, signal

from .evaluation import pearson_r
from .models.linear import lag_matrix


def r1d(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson r of two 1-D series (0 for a constant series)."""
    return float(pearson_r(a, b)[0])


GLOVE_STEP = 40  # samples: the data glove ran at 25 Hz, the ECoG at 1 kHz


def upsample_linear(values: np.ndarray, times: np.ndarray, n: int) -> np.ndarray:
    """Linear interpolation of predictions given at sample indices ``times`` onto 0..n-1 (uses the next value)."""
    grid = np.arange(n)
    return np.stack([np.interp(grid, times, values[:, k]) for k in range(values.shape[1])], axis=1)


def upsample_hold(values: np.ndarray, times: np.ndarray, n: int) -> np.ndarray:
    """Zero-order hold: each sample takes the latest prediction at or before it (causal)."""
    idx = np.searchsorted(times, np.arange(n), side="right") - 1
    return values[np.clip(idx, 0, len(values) - 1)]


def score_1khz(pred: np.ndarray, truth: np.ndarray, fingers=(0, 1, 2, 4)) -> tuple[float, list[float]]:
    """Competition metric: Pearson r per finger at 1 kHz, mean over thumb, index, middle, little."""
    r = [float(r1d(pred[:, k], truth[:, k])) for k in range(truth.shape[1])]
    return float(np.mean([r[k] for k in fingers])), r


# --------------------------------------------------------------------------- 1st place
class LiangBougrainOffline:
    """Competition winner, as published: band-specific amplitude modulation, 25-step memory, forward selection.

    From the paper: three bands (1-60, 60-100, 100-200 Hz) from FIR band-pass filters; amplitude modulation
    x(t_n) = sum of v(t)^2 over t_n .. t_n + 40 ms (Eq. 1, a window that starts at t_n); the k = 25 most recent AM
    values form the memory; one linear model per finger, d = W^T x_E with no bias term (Eq. 2), solved with the
    pseudo-inverse (Eq. 3); per finger, greedy forward selection of electrode-band pairs on a 3/5 - 2/5
    chronological split of the training data, adding the pair that most increases validation correlation,
    stopping when it no longer increases or at 10 pairs.

    Left open by the paper, chosen here:
    * filter design - the paper used equiripple FIR filters of unstated length and direction. Equiripple
      design is numerically unstable for a 1 Hz edge at 1 kHz (scipy.signal.remez diverges), so windowed
      linear-phase FIR filters are used (3301 taps for 1-60 Hz, 1001 taps otherwise), applied with their
      group delay removed, i.e. zero-phase. ``causal=True`` applies them forward only instead.
    * the recording is filtered as one continuous signal (training then test).
    * memory - 25 values: x(t_n), x(t_n-1), ..., x(t_n-24).
    * the final model is refitted on the whole training segment with the selected pairs.
    * 25 Hz predictions are brought to 1 kHz by linear interpolation (offline) or a hold (``causal=True``).
    """

    BANDS = ((1.0, 60.0), (60.0, 100.0), (100.0, 200.0))
    TAPS = (3301, 1001, 1001)

    def __init__(
        self, n_lags: int = 25, max_pairs: int = 10, split: float = 0.6, causal: bool = False, intercept: bool = False
    ):
        self.n_lags, self.max_pairs, self.split = n_lags, max_pairs, split
        self.causal, self.intercept = causal, intercept

    def amplitude_modulation(self, ecog: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """AM per 40 ms window for every electrode and band, ordered band-major within electrode
        (feature index = electrode * 3 + band). Returns (am, t_n) with t_n the sample of each step."""
        n, n_ch = ecog.shape
        x = ecog.astype(np.float64)
        n_steps = n // GLOVE_STEP
        am = np.empty((n_steps, n_ch, len(self.BANDS)))
        for b, (band, taps) in enumerate(zip(self.BANDS, self.TAPS)):
            h = signal.firwin(taps, band, pass_zero=False, fs=1000.0)
            if self.causal:
                v = signal.lfilter(h, 1.0, x, axis=0)
            else:
                v = signal.fftconvolve(x, h[:, None], mode="same", axes=0)  # linear phase, delay removed
            sq = v * v
            if self.causal:  # window ending at t_n: [t_n - 39, t_n]
                csum = np.concatenate([np.zeros((1, n_ch)), np.cumsum(sq, axis=0)])
                ends = np.arange(n_steps) * GLOVE_STEP
                starts = np.maximum(ends - GLOVE_STEP + 1, 0)
                am[:, :, b] = csum[ends + 1] - csum[starts]
            else:  # Eq. 1: window starting at t_n: [t_n, t_n + 39]
                am[:, :, b] = sq[: n_steps * GLOVE_STEP].reshape(n_steps, GLOVE_STEP, n_ch).sum(axis=1)
        return am.reshape(n_steps, -1), np.arange(n_steps) * GLOVE_STEP

    def _cols(self, pair: int, n_pairs: int) -> list[int]:
        return [lag * n_pairs + pair for lag in range(self.n_lags)]

    def _design(self, am: np.ndarray, pairs: list[int]) -> np.ndarray:
        cols = [lag_matrix(am[:, [p]], self.n_lags) for p in pairs]
        if self.intercept:
            cols.append(np.ones((len(am), 1)))
        return np.hstack(cols)

    def select(self, am: np.ndarray, y: np.ndarray, rows: slice) -> LiangBougrainOffline:
        """Greedy forward selection per finger on a chronological 3/5 - 2/5 split of ``rows``."""
        n, n_pairs = rows.stop - rows.start, am.shape[1]
        cut = int(self.split * n)
        design = lag_matrix(am, self.n_lags)[rows]
        if self.intercept:
            design = np.hstack([design, np.ones((n, 1))])
        extra = [design.shape[1] - 1] if self.intercept else []
        fit, val = design[:cut], design[cut:]
        gram = fit.T @ fit
        y = y[rows]
        self.pairs_, self.selection_curves_ = [], []
        for f in range(y.shape[1]):
            rhs = fit.T @ y[:cut, f]
            chosen, best, curve = [], -np.inf, []
            while len(chosen) < self.max_pairs:
                base = [c for p in chosen for c in self._cols(p, n_pairs)]
                trial = {}
                for p in range(n_pairs):
                    if p in chosen:
                        continue
                    idx = base + self._cols(p, n_pairs) + extra
                    w = np.linalg.lstsq(gram[np.ix_(idx, idx)], rhs[idx], rcond=None)[0]  # pinv(X'X) X'd
                    trial[p] = r1d(val[:, idx] @ w, y[cut:, f])
                p_best = max(trial, key=trial.get)
                if not trial[p_best] > best:
                    break
                chosen.append(p_best)
                best = trial[p_best]
                curve.append(float(best))
            self.pairs_.append(chosen)
            self.selection_curves_.append(curve)
        return self

    def fit(self, am: np.ndarray, y: np.ndarray, rows: slice) -> LiangBougrainOffline:
        self.coef_ = []
        for f, pairs in enumerate(self.pairs_):
            a = self._design(am, pairs)[rows]
            self.coef_.append(np.linalg.pinv(a.T @ a) @ (a.T @ y[rows, f]))  # Eq. 3 with the pseudo-inverse
        return self

    def predict(self, am: np.ndarray) -> np.ndarray:
        return np.stack([self._design(am, pairs) @ w for pairs, w in zip(self.pairs_, self.coef_)], axis=1)


def run_liang_offline(
    train_ecog, train_flex, test_ecog, causal: bool = False, intercept: bool = False, exclude: tuple[int, ...] = ()
) -> dict:
    """Train on the competition training data, predict the test data at 1 kHz."""
    ecog = np.vstack([train_ecog, test_ecog])
    if exclude:
        ecog = np.delete(ecog, list(exclude), axis=1)
    model = LiangBougrainOffline(causal=causal, intercept=intercept)
    am, t = model.amplitude_modulation(ecog)
    am = am / np.sqrt(np.mean(am[: len(train_flex) // GLOVE_STEP] ** 2, axis=0))  # unit RMS; predictions unchanged
    n_train = len(train_flex) // GLOVE_STEP
    y = np.zeros((len(am), train_flex.shape[1]))
    y[:n_train] = train_flex[t[:n_train]]  # glove sample at t_n
    rows = slice(0, n_train)
    model.select(am, y, rows).fit(am, y, rows)
    pred = model.predict(am)
    test_t = t[n_train:] - len(train_flex)
    up = upsample_hold if causal else upsample_linear
    return {
        "pred_1khz": up(pred[n_train:], test_t, len(test_ecog)),
        "pairs": model.pairs_,
        "selection_curves": model.selection_curves_,
        "n_channels": ecog.shape[1],
    }


# --------------------------------------------------------------------------- 2nd place
def _ar_first_two(x: np.ndarray, order: int = 10) -> np.ndarray:
    """First two coefficients of a least-squares AR(order) model of each column of ``x`` (window, channels)."""
    out = np.empty((2, x.shape[1]))
    for c in range(x.shape[1]):
        s = x[:, c] - x[:, c].mean()
        lagged = np.stack([s[order - k - 1 : len(s) - k - 1] for k in range(order)], axis=1)
        coef = np.linalg.lstsq(lagged, s[order:], rcond=None)[0]
        out[:, c] = coef[:2]
    return out


def ar_features(ecog: np.ndarray, window: int = 300, order: int = 10, step: int = 4) -> np.ndarray:
    """First two AR coefficients of non-overlapping ``window``-sample windows, per channel, joined by a cubic
    spline through the window centres and read every ``step`` samples. Returns (n // step, channels, 2)."""
    n, n_ch = ecog.shape
    n_win = n // window
    coefs = np.stack([_ar_first_two(ecog[i * window : (i + 1) * window], order) for i in range(n_win)])
    centres = np.arange(n_win) * window + (window - 1) / 2
    grid = np.arange(0, n, step)
    spline = interpolate.CubicSpline(centres, coefs, axis=0, extrapolate=True)  # uses the next window too
    return spline(np.clip(grid, centres[0], centres[-1])).transpose(0, 2, 1)


def shifted(x: np.ndarray, shift: int) -> np.ndarray:
    """x(t + shift) for shift > 0 (future), x(t - |shift|) for shift < 0, edges held."""
    if shift == 0:
        return x
    idx = np.clip(np.arange(len(x)) + shift, 0, len(x) - 1)
    return x[idx]


def group_lasso_least_squares(x: np.ndarray, y: np.ndarray, lam: float, n_iter: int = 300) -> np.ndarray:
    """min ||Y - XC||_F^2 + lam * sum_i ||C_i,:||_2 (rows of C shared across outputs; the last column of X is
    an unpenalised intercept), by accelerated proximal gradient. Stands in for the block-coordinate solver
    cited by Flamary and Rakotomamonjy; same objective. ``lam`` is relative to the smallest value that zeroes
    every row (lam_max)."""
    xty, xtx = x.T @ y, x.T @ x
    step = 1.0 / (2 * np.linalg.eigvalsh(xtx)[-1])
    c = np.zeros((x.shape[1], y.shape[1]))
    z, t = c.copy(), 1.0
    lam = lam * np.max(np.linalg.norm(2 * xty[:-1], axis=1))
    for _ in range(n_iter):
        grad = 2 * (xtx @ z - xty)
        v = z - step * grad
        norms = np.linalg.norm(v, axis=1, keepdims=True)
        c_new = v * np.maximum(0, 1 - step * lam / np.maximum(norms, 1e-12))
        c_new[-1] = v[-1]  # the last row (intercept) is not penalised
        t_new = (1 + np.sqrt(1 + 4 * t * t)) / 2
        z = c_new + (t - 1) / t_new * (c_new - c)
        c, t = c_new, t_new
    return c


def flamary_states(
    flex_250: np.ndarray, rows: slice, threshold: float = 0.2, fill_gaps: bool = False, max_gap: int = 200
) -> np.ndarray:
    """0 = rest, k = 1..5 = finger k moving: the most flexed finger above ``threshold`` of its training range.

    ``fill_gaps``: rest gaps shorter than ``max_gap`` samples (0.8 s at 250 Hz) between two stretches of the same
    finger are relabelled as that finger, so a cue with several flexions becomes one movement period - closer to
    segmenting movement periods by hand, as the paper did."""
    med = np.median(flex_250[rows], axis=0)
    span = np.percentile(flex_250[rows], 95, axis=0) - med + 1e-9
    z = (flex_250 - med) / span
    state = np.argmax(z, axis=1) + 1
    state[z.max(axis=1) < threshold] = 0
    if fill_gaps:
        moving = np.flatnonzero(state > 0)
        for i, j in zip(moving[:-1], moving[1:]):
            if 1 < j - i <= max_gap and state[i] == state[j]:
                state[i + 1 : j] = state[i]
    return state


def run_flamary_offline(train_ecog, train_flex, test_ecog, log=print, exclude: tuple[int, ...] = ()) -> dict:
    """Second place, as published: a linear state classifier switches between per-state ridge models.

    From the paper: ECoG delayed by 37 ms; everything at 250 Hz (downsampled by 4); training split 75 % /
    25 % for tuning; state features = first two coefficients of AR(10) models on non-overlapping 300-sample
    windows, spline-interpolated, at t, t + t_s and t - t_s; channel ranking from per-finger linear regressions
    (+1 moving / -1 not) on the first AR coefficient, top K channels kept; six linear classifiers (rest and
    each finger) with a mixed l1-l2 (row-sparse) penalty, winner takes all; flexion models = one multi-output
    ridge regression per state, trained on that state's samples, inputs = Savitzky-Golay low-passed ECoG
    (3rd order, 0.4 s) at t, t - tau and t + tau, keeping the M features with the largest coefficients.

    Left open by the paper, chosen here (and tuned on the 25 % validation part where the paper tuned them):
    * the spline is an interpolating cubic spline through the window centres;
    * states come from the glove: the most flexed finger above 20 % of its training range, else rest, either at
      each instant or filled into movement periods (gaps under 0.8 s closed); the paper segmented by hand;
    * the 300-sample AR windows are read either at 1 kHz (0.3 s) or at 250 Hz (1.2 s);
    * "37 ms delay to the ECoG": x'(t) = x(t - 37 ms);
    * grids: t_s in {100, 200, 300} ms, K in {10, 20, 40, all}, lambda_s in {0.001, 0.01, 0.05, 0.2} x lambda_max;
      tau in {40, 100, 200} ms, lambda_k in {10, 100, 1000, 10000}, M in {all, 1/2, 1/4} of the features; the
      flexion models are tuned by correlation with the true states (the paper's Table 2 setting), then the
      labels, AR window and classifier settings by the validation correlation of the whole decoder (the paper
      chose K "to maximize validation cross-correlation");
    * final models are refitted on the whole training segment;
    * 250 Hz predictions are brought to 1 kHz by linear interpolation.
    """

    fs, step = 250, 4
    ecog = np.vstack([train_ecog, test_ecog]).astype(np.float64)
    if exclude:  # post-hoc diagnostics only; the replica itself uses every electrode, as the paper did
        ecog = np.delete(ecog, list(exclude), axis=1)
    ecog = shifted(ecog, -37)  # x'(t) = x(t - 37 ms)
    flex = train_flex[::step].astype(np.float64)
    n_tr = len(flex)
    cut = int(0.75 * n_tr)
    fit, val, train = slice(0, cut), slice(cut, n_tr), slice(0, n_tr)
    y_mean, y_std = flex[train].mean(0), flex[train].std(0)
    yz = (flex - y_mean) / y_std

    def corr(pred, true):
        return float(np.mean([r1d(pred[:, k], true[:, k]) for k in (0, 1, 2, 4)]))

    # ---- flexion models: Savitzky-Golay low-pass (3rd order, 0.4 s, centred) of every channel, at 250 Hz
    sg = signal.savgol_filter(ecog, window_length=401, polyorder=3, axis=0)[::step]
    sg = (sg - sg[:n_tr].mean(0)) / (sg[:n_tr].std(0) + 1e-12)

    def flex_design(tau_steps):
        return np.hstack([sg, shifted(sg, -tau_steps), shifted(sg, tau_steps), np.ones((len(sg), 1))])

    def ridge(a, y, lam):
        reg = lam * np.eye(a.shape[1])
        reg[-1, -1] = 0.0
        return np.linalg.solve(a.T @ a + reg, a.T @ y)

    def fit_state_models(xd, states, rows, lam, keep):
        models = []
        for k in range(6):
            idx = np.arange(rows.start, rows.stop)[states[rows] == k]
            h = ridge(xd[idx], yz[idx], lam)
            if keep < 1.0:  # keep the features with the largest summed |coefficient|, refit
                m = max(1, int(keep * (xd.shape[1] - 1)))
                cols = np.append(np.argsort(-np.abs(h[:-1]).sum(axis=1))[:m], xd.shape[1] - 1)
                h_small = ridge(xd[idx][:, cols], yz[idx], lam)
                h = np.zeros_like(h)
                h[cols] = h_small
            models.append(h)
        return models

    def switch(xd, models, state_seq):
        out = np.empty((len(xd), 5))
        for k in range(6):
            sel = state_seq == k
            out[sel] = xd[sel] @ models[k]
        return out

    # ---- state classifier features: first two AR(10) coefficients, two readings of "300-sample windows"
    ar_sets = {}
    for win_label, ecog_in, win, st in (
        ("300 samples at 1 kHz", ecog, 300, step),
        ("300 samples at 250 Hz", ecog[::step], 300, 1),
    ):
        ar = ar_features(ecog_in, window=win, step=st)[: len(sg)]
        ar_sets[win_label] = (ar - ar[:n_tr].mean(0)) / (ar[:n_tr].std(0) + 1e-12)

    def state_design(ar, ts_steps, chans):
        base = ar[:, chans, :].reshape(len(ar), -1)
        return np.hstack([base, shifted(base, ts_steps), shifted(base, -ts_steps), np.ones((len(base), 1))])

    best = None
    for label_kind in ("instants", "movement periods"):
        states = flamary_states(flex, train, fill_gaps=label_kind == "movement periods")
        # flexion models tuned with the true states (the paper's Table 2 setting)
        best_reg = None
        for tau_ms in (40, 100, 200):
            xd = flex_design(tau_ms * fs // 1000)
            for lam in (10.0, 100.0, 1000.0, 10000.0):
                for keep in (1.0, 0.5, 0.25):
                    models = fit_state_models(xd, states, fit, lam, keep)
                    r = corr(switch(xd[val], models, states[val]), yz[val])
                    if best_reg is None or r > best_reg["r_true_states"]:
                        best_reg = {"tau_ms": tau_ms, "lambda_k": lam, "keep": keep, "r_true_states": r}
        xf = flex_design(best_reg["tau_ms"] * fs // 1000)
        models_fit = fit_state_models(xf, states, fit, best_reg["lambda_k"], best_reg["keep"])
        targets = np.where(states[:, None] == np.arange(6)[None, :], 1.0, -1.0)  # columns: rest, f1..f5
        for win_label, ar in ar_sets.items():
            # channel ranking: per-finger regression of +-1 on the first AR coefficient, score = sum_k |c_k|
            a1 = np.hstack([ar[:n_tr, :, 0], np.ones((n_tr, 1))])
            coef = np.linalg.lstsq(a1[fit], targets[fit, 1:], rcond=None)[0][:-1]
            ranking = np.argsort(-np.abs(coef).sum(axis=1))
            for ts_ms in (100, 200, 300):
                for k_ch in (10, 20, 40, ar.shape[1]):
                    xd = state_design(ar, ts_ms * fs // 1000, ranking[:k_ch])
                    for lam in (0.001, 0.01, 0.05, 0.2):
                        c = group_lasso_least_squares(xd[fit], targets[fit], lam)
                        est = np.argmax(xd[val] @ c, axis=1)
                        r = corr(switch(xf[val], models_fit, est), yz[val])  # K etc. chosen on this, as in the paper
                        if best is None or r > best["val_r"]:
                            best = {
                                "labels": label_kind,
                                "ar_window": win_label,
                                "ts_ms": ts_ms,
                                "k": int(k_ch),
                                "lambda_s": lam,
                                "val_r": r,
                                "state_acc": float(np.mean(est == states[val])),
                                **best_reg,
                                "ranking": ranking,
                                "states": states,
                            }
    log("  chosen: " + str({k: v for k, v in best.items() if k not in ("ranking", "states")}))

    # ---- final: refit both blocks on the whole training segment, decode the test segment
    states = best["states"]
    targets = np.where(states[:, None] == np.arange(6)[None, :], 1.0, -1.0)
    xs = state_design(ar_sets[best["ar_window"]], best["ts_ms"] * fs // 1000, best["ranking"][: best["k"]])
    c = group_lasso_least_squares(xs[:n_tr], targets, best["lambda_s"])
    xf = flex_design(best["tau_ms"] * fs // 1000)
    models = fit_state_models(xf, states, train, best["lambda_k"], best["keep"])
    est_states = np.argmax(xs @ c, axis=1)
    pred = switch(xf, models, est_states) * y_std + y_mean
    test_t = np.arange(len(pred) - n_tr) * step
    chosen = {k: v for k, v in best.items() if k not in ("ranking", "states")}
    return {
        "pred_1khz": upsample_linear(pred[n_tr:], test_t, len(test_ecog)),
        "states_test_250hz": est_states[n_tr:],
        "chosen": chosen,
    }


# --------------------------------------------------------------------------- FingerFlex (Lomtev et al., 2023)
def run_fingerflex_offline(
    train_ecog: np.ndarray,
    train_flex: np.ndarray,
    test_ecog: np.ndarray,
    test_flex: np.ndarray,
    seed: int = 0,
    cfg=None,
    log=print,
) -> dict:
    """FingerFlex as published (``fingerflex.py``, offline form), trained twice from the same initial weights.

    Run A, the published procedure: all training data, the test recording decoded after every epoch, and the epoch
    whose test correlation (centred Gaussian, sigma 6 samples, all five fingers) is best is kept.
    Run B, this repository's protocol: the first 80 % of the training recording, the epoch chosen with the same
    criterion on the last 20 %. Run A trained for that many epochs is the refit on all training data, so its test
    output at Run B's epoch is the offline score with the epoch chosen without the test labels.
    Each is scored twice: the notebooks' metric (100 Hz, all five fingers, centred Gaussian sigma 1 sample) and the
    competition metric (1 kHz, ring finger excluded, outputs linearly interpolated between 10 ms steps).
    """
    import copy

    import torch
    from scipy.ndimage import gaussian_filter1d

    from . import fingerflex as ff

    cfg = cfg or ff.FingerFlexConfig(seed=seed)
    s = ff.SHIFT_OFFLINE
    x_tr_raw = ff.recording_features(train_ecog, offline=True)
    x_te_raw = ff.recording_features(test_ecog, offline=True)
    y_tr, y_te = ff.glove_100hz(train_flex), ff.glove_100hz(test_flex)
    n = min(len(x_tr_raw), len(y_tr))
    split = int(0.8 * n)
    yt = y_te[s : len(x_te_raw)]  # fingers 200 ms after each test input

    def scaled(rows: slice):
        sc = ff.RobustScaler().fit(x_tr_raw[rows])
        lo, hi = y_tr[rows].min(0), y_tr[rows].max(0)
        return (
            sc.transform(x_tr_raw[: n - s]),
            sc.transform(x_te_raw[: len(x_te_raw) - s]),
            (y_tr[s:n] - lo) / (hi - lo),
        )

    def smooth(out: np.ndarray, sigma: float) -> np.ndarray:
        return gaussian_filter1d(out, sigma, axis=0)

    torch.manual_seed(seed)
    xa, _, ya = scaled(slice(0, split))
    model = ff.FingerFlexNet(xa.shape[1], causal=False, dropout=cfg.dropout)
    initial = copy.deepcopy(model)
    val = slice(split - s, n - s)  # inputs whose fingers are the last 20 % of the training recording
    best_b = {"val": -np.inf, "epoch": 0}

    def on_val(epoch: int, m) -> dict:
        r = ff.corr_mean(smooth(ff.predict(m, xa)[val], 6), ya[val])
        if r > best_b["val"]:
            best_b.update(val=r, epoch=epoch)
        return {"val": r, "stop": epoch - best_b["epoch"] >= cfg.patience}

    hist_b = ff.train(model, xa, ya.astype(np.float32), slice(0, split - s), cfg, cfg.max_epochs, on_val)
    log(f"  run B (epoch chosen on validation): {[round(h['val'], 3) for h in hist_b]}")

    xa, xt, ya = scaled(slice(0, n))
    model, outs = initial, []

    def on_test(epoch: int, m) -> dict:
        outs.append(ff.predict(m, xt))
        return {"test_monitor": ff.corr_mean(smooth(outs[-1], 6), yt)}

    hist_a = ff.train(model, xa, ya.astype(np.float32), slice(0, n - s), cfg, cfg.max_epochs, on_test)
    log(f"  run A (test monitored): {[round(h['test_monitor'], 3) for h in hist_a]}")

    def scores(out: np.ndarray) -> dict:
        out = smooth(out, 1)
        t = (np.arange(len(out)) + s) * ff.DECIM  # 1 kHz sample of the finger each output estimates
        pred_1khz = upsample_linear(out, t, len(test_flex))
        comp, per = score_1khz(pred_1khz, test_flex)
        return {"notebook_metric": ff.corr_mean(out, yt), "score": comp, "r": per, "pred_1khz": pred_1khz}

    e_test = int(np.argmax([h["test_monitor"] for h in hist_a]))
    e_val = int(np.argmax([h["val"] for h in hist_b]))
    return {
        "published": {"epoch": e_test + 1, **scores(outs[e_test])},
        "validation_epoch": {"epoch": e_val + 1, **scores(outs[e_val])},
        "history_b_val": [h["val"] for h in hist_b],
        "history_a_test": [h["test_monitor"] for h in hist_a],
    }
