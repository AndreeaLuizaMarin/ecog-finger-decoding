import numpy as np

from ecogdecode.evaluation import pearson_r, score_test, scored_mean, to_sample_rate


def test_pearson_matches_numpy():
    rng = np.random.default_rng(0)
    a, b = rng.standard_normal((500, 5)), rng.standard_normal((500, 5))
    b[:, 0] += 2 * a[:, 0]
    expected = [np.corrcoef(a[:, k], b[:, k])[0, 1] for k in range(5)]
    np.testing.assert_allclose(pearson_r(a, b), expected, rtol=1e-10)


def test_constant_prediction_scores_zero():
    assert np.all(pearson_r(np.ones((100, 5)), np.random.default_rng(0).standard_normal((100, 5))) == 0)


def test_scored_mean_excludes_ring_finger():
    assert scored_mean(np.array([1.0, 1.0, 1.0, -100.0, 1.0])) == 1.0


def test_zero_order_hold_is_causal():
    held = to_sample_rate(np.arange(3, dtype=float)[:, None], np.array([0, 40, 80]), 120)
    assert held[39, 0] == 0 and held[40, 0] == 1 and held[119, 0] == 2


def test_perfect_step_prediction_scores_one():
    rng = np.random.default_rng(1)
    steps = rng.standard_normal((50, 5))
    times = np.arange(-1, 49 * 40, 40)
    truth = to_sample_rate(steps, times, 49 * 40)
    assert abs(score_test(steps, times, truth)["score"] - 1.0) < 1e-12
