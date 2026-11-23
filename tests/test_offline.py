"""Offline replicas, offline features and the Kalman variants: they run, keep their shapes, and the offline
pieces really do use the future (which is why they cannot run live)."""

from dataclasses import replace

import numpy as np

from ecogdecode.config import DEFAULT_FEATURES
from ecogdecode.features import extract_features
from ecogdecode.models import DiscriminativeKalmanDecoder, LaggedKalmanDecoder, LaggedRidge
from ecogdecode.offline import (
    LiangBougrainOffline,
    flamary_states,
    run_flamary_offline,
    run_liang_offline,
    upsample_hold,
    upsample_linear,
)


def _toy(n_train=20000, n_test=8000, channels=4, seed=0):
    rng = np.random.default_rng(seed)
    ecog = rng.standard_normal((n_train + n_test, channels)).astype(np.float32)
    flex = np.repeat(rng.standard_normal(((n_train + n_test) // 40, 5)).cumsum(0), 40, axis=0).astype(np.float32)
    return ecog[:n_train], flex[:n_train], ecog[n_train:], flex[n_train:]


def test_offline_features_use_the_future_and_causal_ones_do_not():
    x = np.random.default_rng(1).standard_normal((4000, 3)).astype(np.float32)
    changed = x.copy()
    changed[3000:, 0] += 5.0 * np.sin(np.arange(1000) * 0.6)  # one channel only (re-referencing removes common changes)
    for offline, should_change in ((False, False), (True, True)):
        cfg = replace(DEFAULT_FEATURES, offline=offline)
        a, ends = extract_features(x, cfg)
        b, _ = extract_features(changed, cfg)
        early = ends < 2900
        assert np.allclose(a[early], b[early]) != should_change


def test_liang_offline_am_window_starts_at_the_step_and_causal_one_ends_there():
    x = np.zeros((2000, 1))
    x[45] = 1.0  # a single sample at t = 45 ms
    off, t = LiangBougrainOffline(causal=False).amplitude_modulation(x)
    on, _ = LiangBougrainOffline(causal=True).amplitude_modulation(x)
    assert t[1] == 40
    # the zero-phase filtered impulse is centred on 45 ms: it lands in the window starting at 40 ms
    assert np.argmax(off[:, 1]) == 1
    # forward-only filtering delays it by the filter's group delay, so it appears later
    assert np.argmax(on[:, 1]) > 1


def test_replicas_run_on_toy_data():
    tr_x, tr_y, te_x, te_y = _toy()
    for kwargs in ({}, {"causal": True}, {"intercept": True}):
        out = run_liang_offline(tr_x, tr_y, te_x, **kwargs)
        assert out["pred_1khz"].shape == te_y.shape and np.isfinite(out["pred_1khz"]).all()
    out = run_flamary_offline(tr_x, tr_y, te_x, log=lambda m: None)
    assert out["pred_1khz"].shape == te_y.shape and np.isfinite(out["pred_1khz"]).all()


def test_upsampling_hold_is_causal_and_linear_is_not():
    values = np.array([[0.0], [1.0]])
    times = np.array([0, 40])
    hold, lin = upsample_hold(values, times, 80), upsample_linear(values, times, 80)
    assert hold[20, 0] == 0.0  # only past predictions
    assert 0 < lin[20, 0] < 1  # already moving towards the next prediction


def test_movement_periods_fill_short_gaps_only():
    y = np.zeros((1000, 5))
    y[100:120, 1] = 1.0
    y[150:170, 1] = 1.0  # same finger again after a 30-sample gap
    y[600:620, 2] = 1.0
    states = flamary_states(y, slice(0, 1000), fill_gaps=True, max_gap=50)
    assert (states[100:170] == 2).all() and (states[300:500] == 0).all()


def test_kalman_variants_run():
    rng = np.random.default_rng(2)
    y = np.cumsum(rng.standard_normal((3000, 5)) * 0.1, axis=0)
    x = np.hstack([y, y**2]) + 0.5 * rng.standard_normal((3000, 10))
    fit = slice(0, 2000)
    kf = LaggedKalmanDecoder(4, 3).fit(x[fit], y[fit])
    assert kf.predict(x).shape == y.shape
    reg = LaggedRidge(3, 1.0).fit(x, y, fit)
    q = np.cov((y[fit] - reg.predict(x)[fit]).T)
    dkf = DiscriminativeKalmanDecoder(reg, q).fit(y[fit])
    out = dkf.predict(x)
    assert out.shape == y.shape and np.isfinite(out).all()
