"""FingerFlex re-implementation: the causal form is causal, the published form is not, and both run end to end."""

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from torch import nn

from ecogdecode import fingerflex as ff
from ecogdecode.approaches import run_fingerflex
from ecogdecode.data import SubjectData
from ecogdecode.offline import run_fingerflex_offline
from ecogdecode.pipeline import evaluate, prepare

TINY = ff.FingerFlexConfig(window=64, batch_size=64, max_epochs=2)


@pytest.fixture(scope="module")
def short_subject(synthetic_subject):
    """16 s of training and 6 s of test data: enough for a few 0.64 s windows."""
    s = synthetic_subject
    return SubjectData(0, s.train_ecog[:16_000], s.train_flex[:16_000], s.test_ecog[:6_000], s.test_flex[:6_000])


@pytest.mark.parametrize("causal", [True, False])
def test_network_is_causal_only_in_causal_form(causal):
    torch.manual_seed(0)
    net = ff.FingerFlexNet(12, causal=causal).eval()
    x = torch.randn(1, 128, 12)
    x2 = x.clone()
    x2[:, 70:] = torch.randn(1, 58, 12)
    same = torch.allclose(net(x2)[..., :70], net(x)[..., :70], atol=1e-5)
    assert same == causal


@pytest.mark.parametrize("causal", [True, False])
def test_first_block_equals_the_published_convolution(causal):
    torch.manual_seed(0)
    red = ff._Reduce(12, 4, causal, 0.0).eval()
    conv = nn.Conv1d(12, 4, 3, bias=False, padding=0 if causal else "same")
    conv.weight.data = red.proj.weight.detach().unflatten(0, (3, 4)).permute(1, 2, 0).contiguous()
    x = torch.randn(2, 64, 12)
    xc = F.pad(x.transpose(1, 2), (2, 0)) if causal else x.transpose(1, 2)
    expected = F.gelu(red.norm(conv(xc).transpose(1, 2))).transpose(1, 2)
    torch.testing.assert_close(red(x), expected, rtol=1e-5, atol=1e-5)


def test_parameter_count_matches_the_notebook():
    assert sum(p.numel() for p in ff.FingerFlexNet(64 * 40).parameters()) == 660_293


def test_causal_wavelet_power_ignores_the_future():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((3000, 2))
    x2 = x.copy()
    x2[2000:] = rng.standard_normal((1000, 2))
    causal, causal2 = ff.wavelet_power(x, offline=False), ff.wavelet_power(x2, offline=False)
    np.testing.assert_allclose(causal[:200], causal2[:200], rtol=1e-4, atol=1e-6)  # outputs up to sample 1990
    centred, centred2 = ff.wavelet_power(x, offline=True), ff.wavelet_power(x2, offline=True)
    assert not np.allclose(centred[195:200], centred2[195:200])  # centred wavelets reach past sample 2000


def test_causal_fingerflex_runs_under_the_protocol(short_subject):
    prep = prepare(short_subject)
    res = run_fingerflex(prep, short_subject, seed=0, cfg=TINY)
    assert res.pred.shape == (len(prep.ends), 5) and np.all(np.isfinite(res.pred))
    assert 1 <= res.selected["epochs"] <= TINY.max_epochs
    assert np.isfinite(evaluate(prep, res.pred, short_subject.test_flex)["score"])


def test_published_fingerflex_runs(short_subject):
    s = short_subject
    res = run_fingerflex_offline(s.train_ecog, s.train_flex, s.test_ecog, s.test_flex, cfg=TINY, log=lambda m: None)
    for kind in ("published", "validation_epoch"):
        assert np.isfinite(res[kind]["score"]) and res[kind]["pred_1khz"].shape == s.test_flex.shape
