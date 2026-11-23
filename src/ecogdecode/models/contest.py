"""The two best entries of BCI Competition IV, data set 4, re-implemented to run live.

Both original methods were designed offline: some of their filters and feature shifts used
samples after the predicted moment. The versions here keep each method's structure and
replace every non-causal step with its causal counterpart; the deviations are listed in the
class docstrings.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression

from ..evaluation import score
from .linear import lag_matrix


# --------------------------------------------------------------------------- 1st place
class LiangBougrainDecoder:
    """Winner: band-specific amplitude modulation + short-term memory + forward selection.

    Liang and Bougrain (2012): amplitude modulation (AM) = summed squared signal in 40 ms windows,
    in 1-60, 60-100 and 100-200 Hz, for every electrode; the last k = 25 AM values (1 s) form the
    short-term memory; one linear model per finger, solved with the pseudo-inverse (Wiener
    solution); greedy forward selection of electrode-band pairs on a 3/5 - 2/5 split of the
    training data, adding the pair that most increases validation correlation and stopping when
    it no longer increases or after 10 pairs.

    Deviations: causal Butterworth band-pass filters (the paper used equiripple FIR filters
    without stating the filtering direction); median re-referencing and artefact masking, as for
    every approach here; z-scored AM (an affine change that leaves least-squares predictions
    unchanged but keeps the masking rule uniform).
    """

    def __init__(self, n_lags: int = 25, max_pairs: int = 10, selection_split: float = 0.6):
        self.n_lags, self.max_pairs, self.selection_split = n_lags, max_pairs, selection_split

    def _design(self, am: np.ndarray, pairs: list[int]) -> np.ndarray:
        cols = [lag_matrix(am[:, [p]], self.n_lags) for p in pairs]
        return np.hstack(cols + [np.ones((len(am), 1), dtype=am.dtype)])

    @staticmethod
    def _solve(a: np.ndarray, y: np.ndarray) -> np.ndarray:
        return np.linalg.lstsq(a.astype(np.float64), y.astype(np.float64), rcond=None)[0]  # pseudo-inverse

    def select(self, am: np.ndarray, y: np.ndarray, rows: slice) -> LiangBougrainDecoder:
        """Greedy forward selection, per finger, inside ``rows`` (chronological 3/5 - 2/5 split).

        Every candidate model is a least-squares fit; the Gram matrix of all lagged features is
        computed once so that each candidate costs only a small solve.
        """
        n, n_pairs, k = rows.stop - rows.start, am.shape[1], self.n_lags
        cut = int(self.selection_split * n)
        lagged = lag_matrix(am, k).astype(np.float64)[rows]  # (n, lag * pairs), lag-major

        def cols_of(p):
            return [lag * n_pairs + p for lag in range(k)]

        design = np.hstack([lagged, np.ones((n, 1))])
        intercept = design.shape[1] - 1
        fit, val = design[:cut], design[cut:]
        gram = fit.T @ fit
        y = y[rows].astype(np.float64)
        self.pairs_ = []
        for f in range(y.shape[1]):
            rhs = fit.T @ y[:cut, f]
            chosen, best = [], -np.inf
            while len(chosen) < self.max_pairs:
                base = [c for p in chosen for c in cols_of(p)]
                trial = {}
                for p in range(n_pairs):
                    if p in chosen:
                        continue
                    idx = base + cols_of(p) + [intercept]
                    w = np.linalg.lstsq(gram[np.ix_(idx, idx)], rhs[idx], rcond=None)[0]
                    trial[p] = np.corrcoef(val[:, idx] @ w, y[cut:, f])[0, 1]
                p_best = max(trial, key=trial.get)
                if not trial[p_best] > best:
                    break
                chosen.append(p_best)
                best = trial[p_best]
            self.pairs_.append(chosen)
        return self

    def fit(self, am: np.ndarray, y: np.ndarray, rows: slice) -> LiangBougrainDecoder:
        self.coef_ = [self._solve(self._design(am, pairs)[rows], y[rows, k]) for k, pairs in enumerate(self.pairs_)]
        return self

    def predict(self, am: np.ndarray) -> np.ndarray:
        return np.stack([self._design(am, pairs) @ w for pairs, w in zip(self.pairs_, self.coef_)], axis=1)


# --------------------------------------------------------------------------- 2nd place
def finger_states(y: np.ndarray, rows: slice, threshold: float = 0.2) -> np.ndarray:
    """0 = rest, k = 1..5 = finger k-1 moving (the most flexed finger above its threshold).

    Flexion is normalised per finger with training statistics: (y - median) / (95th pct - median).
    """
    med = np.median(y[rows], axis=0)
    span = np.percentile(y[rows], 95, axis=0) - med + 1e-9
    z = (y - med) / span
    state = np.argmax(z, axis=1) + 1
    state[z.max(axis=1) < threshold] = 0
    return state


def _delayed(x: np.ndarray, d: int) -> np.ndarray:
    return np.vstack([np.repeat(x[:1], d, 0), x[:-d]])


class SwitchingLinearDecoder:
    """Second place: a state classifier switches between per-state linear decoders.

    Flamary and Rakotomamonjy (2012): block 1 estimates which of six states holds (rest, or
    finger k moving) with linear models and a winner-takes-all rule, with electrodes selected
    jointly across states; block 2 holds one ridge-regression model per state, trained only on
    samples of that state, mapping low-pass filtered ECoG at several time shifts to all five
    fingers; the output is the model of the predicted state.

    Deviations, all needed to run live: the state classifier reads band power at t and t - 200 ms
    (the paper used autoregressive coefficients from centred windows with shifts of +/- t_s; made
    causal, those gave chance-level state accuracy on validation); the flexion models read the
    local motor potential at t, t - 200 ms and t - 400 ms (paper: Savitzky-Golay filter, shifts t
    and t +/- tau); states come from a fixed threshold on the training flexion (paper: manual
    segmentation); multinomial logistic regression with balanced class weights and electrode
    selection by group norm (paper: a mixed-norm simultaneous sparse approximation).
    """

    def __init__(
        self, n_per_channel: int, n_keep: int | None = None, C: float = 0.01, alpha: float = 10.0, shift: int = 5
    ):
        self.n_per_channel, self.n_keep, self.C, self.alpha, self.shift = n_per_channel, n_keep, C, alpha, shift

    def _classifier(self) -> LogisticRegression:
        return LogisticRegression(C=self.C, max_iter=1000, class_weight="balanced")

    def _state_inputs(self, x: np.ndarray, cols: np.ndarray | None = None) -> np.ndarray:
        xs = np.hstack([x, _delayed(x, self.shift)])
        return xs if cols is None else xs[:, cols]

    def _flex_inputs(self, lmp: np.ndarray) -> np.ndarray:
        return np.hstack([lmp, _delayed(lmp, self.shift), _delayed(lmp, 2 * self.shift)])

    def channel_ranking(self, x: np.ndarray, states: np.ndarray, rows: slice) -> np.ndarray:
        """Electrodes ordered by the group norm of their coefficients in a classifier using all of them."""
        clf = self._classifier().fit(self._state_inputs(x)[rows], states[rows])
        n_channels = x.shape[1] // self.n_per_channel
        w = clf.coef_.reshape(len(clf.classes_), 2, n_channels, self.n_per_channel)
        return np.argsort(-np.sqrt((w**2).sum(axis=(0, 1, 3))))

    def fit(
        self,
        x: np.ndarray,
        lmp: np.ndarray,
        y: np.ndarray,
        states: np.ndarray,
        rows: slice,
        ranking: np.ndarray | None = None,
    ) -> SwitchingLinearDecoder:
        self.cols_ = None
        if self.n_keep is not None and ranking is not None:
            keep = np.sort(ranking[: self.n_keep])
            one = np.concatenate([np.arange(c * self.n_per_channel, (c + 1) * self.n_per_channel) for c in keep])
            self.cols_ = np.concatenate([one, one + x.shape[1]])
        self.clf_ = self._classifier().fit(self._state_inputs(x, self.cols_)[rows], states[rows])
        a_all = np.hstack([self._flex_inputs(lmp), np.ones((len(lmp), 1))]).astype(np.float64)
        self.models_ = {}
        for s in np.unique(states[rows]):
            idx = np.arange(rows.start, rows.stop)[states[rows] == s]
            a = a_all[idx]
            self.models_[s] = np.linalg.solve(
                a.T @ a + self.alpha * np.eye(a.shape[1]), a.T @ y[idx].astype(np.float64)
            )
        return self

    def predict(self, x: np.ndarray, lmp: np.ndarray) -> np.ndarray:
        state = self.clf_.predict(self._state_inputs(x, self.cols_))
        a = np.hstack([self._flex_inputs(lmp), np.ones((len(lmp), 1))])
        out = np.zeros((len(lmp), 5))
        for s, w in self.models_.items():
            out[state == s] = a[state == s] @ w
        return out


def select_switching(
    x: np.ndarray,
    lmp: np.ndarray,
    y: np.ndarray,
    fit_rows: slice,
    val_rows: slice,
    n_per_channel: int,
    keep_grid=(16, 32, None),
    c_grid=(0.001, 0.003, 0.01),
    alpha_grid=(1.0, 10.0, 100.0),
) -> tuple[dict, list[dict], np.ndarray]:
    """Choose electrode count, classifier strength and ridge penalty on the validation block."""
    states = finger_states(y, fit_rows)
    grid = []
    for c in c_grid:
        ranking = SwitchingLinearDecoder(n_per_channel, C=c).channel_ranking(x, states, fit_rows)
        for keep in keep_grid:
            for alpha in alpha_grid:
                model = SwitchingLinearDecoder(n_per_channel, keep, c, alpha).fit(x, lmp, y, states, fit_rows, ranking)
                pred = model.predict(x[: val_rows.stop], lmp[: val_rows.stop])[val_rows]
                grid.append({"n_keep": keep, "C": c, "alpha": alpha, "val_score": score(pred, y[val_rows])})
                if grid[-1]["val_score"] >= max(g["val_score"] for g in grid):
                    best_pred = pred
    return max(grid, key=lambda g: g["val_score"]), grid, best_pred
