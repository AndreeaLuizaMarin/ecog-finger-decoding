"""Kalman-filter decoder (Wu et al., 2006), the classical baseline for continuous BCI control.

State:        s_t = [finger positions, finger velocities]      (10-D)
Dynamics:     s_t = A s_{t-1} + w,   w ~ N(0, W)
Observation:  z_t = H s_t + q,       q ~ N(0, Q)

z_t are the neural features projected onto their leading principal components, which
keeps the observation covariance Q well conditioned. All matrices are fitted in closed
form by least squares on the training data; decoding is a forward (causal) pass.
"""

from __future__ import annotations

import numpy as np


def finger_state(positions: np.ndarray) -> np.ndarray:
    """Positions plus first-difference velocities (zero velocity at the first step)."""
    vel = np.vstack([np.zeros((1, positions.shape[1])), np.diff(positions, axis=0)])
    return np.hstack([positions, vel])


class KalmanDecoder:
    def __init__(self, n_components: int = 32):
        self.n_components = n_components

    def _project(self, x: np.ndarray) -> np.ndarray:
        return (x - self.x_mean_) @ self.components_.T

    def fit(self, x: np.ndarray, y: np.ndarray) -> KalmanDecoder:
        """Fit on a contiguous training block of features ``x`` (n, f) and positions ``y`` (n, k)."""
        x, y = x.astype(np.float64), y.astype(np.float64)
        self.n_outputs_ = y.shape[1]
        self.x_mean_ = x.mean(axis=0)
        _, _, vt = np.linalg.svd(x - self.x_mean_, full_matrices=False)
        self.components_ = vt[: self.n_components]
        z = self._project(x)

        s = finger_state(y)
        self.s_mean_ = s.mean(axis=0)
        s = s - self.s_mean_
        self.z_mean_ = z.mean(axis=0)
        z = z - self.z_mean_

        prev, nxt = s[:-1], s[1:]
        self.A_ = np.linalg.lstsq(prev, nxt, rcond=None)[0].T
        self.W_ = np.cov((nxt - prev @ self.A_.T).T)
        self.H_ = np.linalg.lstsq(s, z, rcond=None)[0].T
        self.Q_ = np.cov((z - s @ self.H_.T).T)
        self.P0_ = np.cov(s.T)
        return self

    def predict(self, x_seq: np.ndarray) -> np.ndarray:
        """Run the filter forward over a continuous feature sequence; returns finger positions."""
        z_seq = self._project(x_seq.astype(np.float64)) - self.z_mean_
        A, W, H, Q = self.A_, self.W_, self.H_, self.Q_
        s = np.zeros(A.shape[0])
        P = self.P0_.copy()
        eye = np.eye(A.shape[0])
        out = np.empty((len(z_seq), self.n_outputs_))
        for t, z in enumerate(z_seq):
            s = A @ s  # predict
            P = A @ P @ A.T + W
            S = H @ P @ H.T + Q  # update
            K = np.linalg.solve(S, H @ P).T  # = P H^T S^-1 (S and P are symmetric)
            s = s + K @ (z - H @ s)
            P = (eye - K @ H) @ P
            out[t] = s[: self.n_outputs_]
        return out + self.s_mean_[: self.n_outputs_]


class LaggedKalmanDecoder(KalmanDecoder):
    """Kalman filter whose observation is the current *and recent* feature vectors (``n_lags`` steps, then PCA).

    ECoG leads movement by roughly 50-150 ms, so the features that best describe the finger state at step t
    include earlier steps. Wu et al. (2006) found the same for intracortical spikes and used a lag; here the
    lags are stacked so the filter can weigh them.
    """

    def __init__(self, n_components: int = 32, n_lags: int = 3):
        super().__init__(n_components)
        self.n_lags = n_lags

    def _stack(self, x: np.ndarray) -> np.ndarray:
        from .linear import lag_matrix

        return lag_matrix(x, self.n_lags)

    def fit(self, x: np.ndarray, y: np.ndarray) -> LaggedKalmanDecoder:
        return super().fit(self._stack(x)[self.n_lags :], y[self.n_lags :])

    def predict(self, x_seq: np.ndarray) -> np.ndarray:
        return super().predict(self._stack(x_seq))


class DiscriminativeKalmanDecoder:
    """Discriminative Kalman filter (Burkhart, Brandman, Franco, Hochberg and Harrison, 2020), Gaussian form.

    A classical Kalman filter needs a *generative* model of the neural features given the finger state
    (z = H s + noise), which is a poor description of ECoG band power. The DKF turns this around: a
    regression f(x_t) estimates the finger positions from the features, with residual covariance Q, and
    the filter combines it with the state dynamics. Bayes' rule gives p(x|s) proportional to p(s|x) / p(s),
    so the update adds the information of the regression and removes that of the stationary prior S:

        precision  P^-1 = P_pred^-1 + H^T (Q^-1 - S^-1) H
        mean       s    = P (P_pred^-1 s_pred + H^T Q^-1 f(x))

    State: positions and velocities of the five fingers (centred); H picks the positions.
    ``regressor`` is any fitted object with ``predict(x_seq)`` returning positions (here a Wiener filter).
    Q is the covariance of the regressor's residuals on rows it was not fitted on (supplied by the caller),
    so it is not underestimated.
    """

    def __init__(self, regressor, residual_cov: np.ndarray):
        self.regressor = regressor
        self.Q_ = residual_cov

    def fit(self, y: np.ndarray) -> DiscriminativeKalmanDecoder:
        """Fit dynamics and the stationary prior on training positions ``y`` (n, k)."""
        y = y.astype(np.float64)
        k = y.shape[1]
        s = finger_state(y)
        self.s_mean_ = s.mean(axis=0)
        s = s - self.s_mean_
        prev, nxt = s[:-1], s[1:]
        self.A_ = np.linalg.lstsq(prev, nxt, rcond=None)[0].T
        self.W_ = np.cov((nxt - prev @ self.A_.T).T)
        self.P0_ = np.cov(s.T)
        self.H_ = np.hstack([np.eye(k), np.zeros((k, k))])
        S = np.cov(y.T)
        info = np.linalg.inv(self.Q_) - np.linalg.inv(S)
        evals, evecs = np.linalg.eigh((info + info.T) / 2)  # keep the information positive semi-definite
        self.obs_info_ = self.H_.T @ (evecs * np.clip(evals, 0, None)) @ evecs.T @ self.H_
        self.obs_gain_ = self.H_.T @ np.linalg.inv(self.Q_)
        self.n_outputs_ = k
        return self

    def predict(self, x_seq: np.ndarray) -> np.ndarray:
        f = self.regressor.predict(x_seq) - self.s_mean_[: self.n_outputs_]
        A, W = self.A_, self.W_
        s, P = np.zeros(A.shape[0]), self.P0_.copy()
        out = np.empty((len(f), self.n_outputs_))
        for t in range(len(f)):
            s, P = A @ s, A @ P @ A.T + W
            prior_info = np.linalg.inv(P)
            P = np.linalg.inv(prior_info + self.obs_info_)
            s = P @ (prior_info @ s + self.obs_gain_ @ f[t])
            out[t] = s[: self.n_outputs_]
        return out + self.s_mean_[: self.n_outputs_]
