import numpy as np
import pytest
import torch

from ecogdecode.models import (
    KalmanDecoder,
    LaggedRidge,
    LiangBougrainDecoder,
    LSTMDecoder,
    LSTMStep,
    MambaDecoder,
    ODEDecoder,
    POSSMDecoder,
    POSSMStep,
    S4DDecoder,
    S4DLayer,
    S4DStep,
    StateGatedDecoder,
    StateGatedStep,
    TCNDecoder,
    TransientGatedDecoder,
    build_manifold_gated,
    finger_states,
    lag_matrix,
    selective_scan,
)


def test_lag_matrix():
    x = np.arange(6, dtype=float).reshape(3, 2)
    np.testing.assert_array_equal(lag_matrix(x, 2), [[0, 1, 0, 0], [2, 3, 0, 1], [4, 5, 2, 3]])


def test_ridge_recovers_linear_map():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((2000, 6))
    y = x @ rng.standard_normal((6, 2)) + 0.01 * rng.standard_normal((2000, 2))
    pred = LaggedRidge(n_lags=1, alpha=1e-3).fit(x, y, slice(0, 1500)).predict(x)
    np.testing.assert_allclose(pred[1500:], y[1500:], atol=0.05)


def test_kalman_tracks_simulated_state():
    rng = np.random.default_rng(0)
    n, k, f = 4000, 2, 20
    pos = np.cumsum(rng.standard_normal((n, k)) * 0.1, axis=0)
    pos -= np.convolve(pos[:, 0], np.ones(200) / 200, mode="same")[:, None]
    x = pos @ rng.standard_normal((k, f)) + 0.5 * rng.standard_normal((n, f))
    pred = KalmanDecoder(n_components=10).fit(x[:3000], pos[:3000]).predict(x)[3000:]
    assert min(np.corrcoef(pred[:, i], pos[3000:, i])[0, 1] for i in range(k)) > 0.9


def test_liang_bougrain_selects_the_informative_pair():
    """Forward selection must pick the one feature that drives the output first."""
    rng = np.random.default_rng(0)
    am = rng.standard_normal((3000, 12)).astype(np.float32)
    drive = np.convolve(am[:, 7], np.ones(5) / 5)[:3000]  # depends on the recent past of feature 7
    y = np.stack([drive + 0.1 * rng.standard_normal(3000) for _ in range(5)], axis=1)
    model = LiangBougrainDecoder(n_lags=5, max_pairs=3).select(am, y, slice(0, 3000))
    assert all(pairs[0] == 7 for pairs in model.pairs_)
    pred = model.fit(am, y, slice(0, 2000)).predict(am)
    assert np.corrcoef(pred[2000:, 0], y[2000:, 0])[0, 1] > 0.9


def test_finger_states():
    y = np.zeros((100, 5))
    y[10:20, 2] = 5.0
    y[50:60, 0] = 3.0
    states = finger_states(y, slice(0, 100))
    assert (states[10:20] == 3).all() and (states[50:60] == 1).all() and states[:10].sum() == 0


def _transient(n_features):
    return TransientGatedDecoder.from_training_data(np.random.default_rng(0).standard_normal((200, n_features)), 6, k=4)


def _manifold(kind, backbone="leaky"):
    rng = np.random.default_rng(0)
    return lambda n: build_manifold_gated(
        kind, rng.standard_normal((300, n)), rng.standard_normal((300, 5)), dim=4, backbone=backbone
    )


NETS = [
    LSTMDecoder,
    S4DDecoder,
    MambaDecoder,
    ODEDecoder,
    TCNDecoder,
    lambda n: StateGatedDecoder(n, 6),
    _transient,
    lambda n: POSSMDecoder(n, 6),
    _manifold("pca"),
    _manifold("pca", "lstm"),
    _manifold("contrastive"),
]


@pytest.mark.parametrize("build", NETS)
def test_networks_are_causal(build):
    torch.manual_seed(0)
    model = build(12).eval()
    x = torch.randn(1, 120, 12)
    x2 = x.clone()
    x2[:, 80:] = torch.randn(1, 40, 12)
    torch.testing.assert_close(model(x2)[:, :80], model(x)[:, :80], rtol=1e-4, atol=1e-5)


def test_selective_scan_is_exact():
    torch.manual_seed(0)
    b, length, d, n = 2, 50, 8, 4
    u, dt = torch.randn(b, length, d), torch.rand(b, length, d) * 0.1
    A, B, C = -torch.rand(d, n) * 5, torch.randn(b, length, n), torch.randn(b, length, n)
    h, ys = torch.zeros(b, d, n), []
    for t in range(length):
        h = torch.exp(dt[:, t, :, None] * A) * h + (dt[:, t, :, None] * B[:, t, None, :]) * u[:, t, :, None]
        ys.append((h * C[:, t, None, :]).sum(-1))
    torch.testing.assert_close(selective_scan(u, dt, A, B, C), torch.stack(ys, 1), rtol=1e-4, atol=1e-5)


def test_s4d_convolution_equals_recurrence():
    torch.manual_seed(0)
    layer = S4DLayer(d_model=4, d_state=8)
    u = torch.randn(1, 64, 4)
    _, dA, dB, C = layer.discretize()
    state, ys = torch.zeros(4, 4, dtype=torch.cfloat), []
    for t in range(64):
        state = dA * state + dB * u[0, t, :, None]
        ys.append(2 * (C * state).sum(-1).real + layer.D * u[0, t])
    torch.testing.assert_close(layer(u)[0], torch.stack(ys), rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("kind", ["lstm", "s4d", "gated", "possm"])
def test_streaming_step_matches_full_sequence(kind):
    torch.manual_seed(0)
    model = {
        "lstm": LSTMDecoder(12),
        "s4d": S4DDecoder(12),
        "gated": StateGatedDecoder(12, 6),
        "possm": POSSMDecoder(12, 6),
    }[kind].eval()
    step = {"lstm": LSTMStep, "s4d": S4DStep, "gated": StateGatedStep, "possm": POSSMStep}[kind](model)
    x = torch.randn(1, 50, 12)
    with torch.no_grad():
        full = model(x)[0]
        state = step.initial_state()
        state = state if isinstance(state, tuple) else (state,)
        outs = []
        for t in range(50):
            y, *state = step(x[:, t], *state)
            outs.append(y[0])
    torch.testing.assert_close(torch.stack(outs), full, rtol=1e-4, atol=1e-4)
