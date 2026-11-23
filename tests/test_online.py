"""The packaged streaming decoder: chunk-by-chunk output equals the batch prediction, in PyTorch and in ONNX."""

import numpy as np
import pytest

from ecogdecode.config import TrainConfig
from ecogdecode.online import OnlineDecoder, batch_prediction

FAST = TrainConfig(seq_len=64, burn_in=16, windows_per_epoch=32, max_epochs=3, patience=3)


@pytest.fixture(scope="module")
def short(synthetic_subject):
    s = synthetic_subject
    return s.train_ecog[:16_000], s.train_flex[:16_000], s.test_ecog[:4_000]


@pytest.mark.parametrize("approach, n_electrodes", [("lstm", None), ("lstm", 3), ("possm", None)])
def test_streaming_equals_batch(short, approach, n_electrodes):
    train_ecog, train_flex, test_ecog = short
    decoder = OnlineDecoder.fit(train_ecog, train_flex, approach, n_electrodes=n_electrodes, train_cfg=FAST)
    streamed = decoder.decode(test_ecog)
    assert streamed.shape == (len(test_ecog) // 40, 5)
    np.testing.assert_allclose(streamed, batch_prediction(decoder, test_ecog), rtol=1e-3, atol=1e-3)
    if n_electrodes:
        assert len(decoder.electrodes) == n_electrodes


def test_onnx_steps_match_pytorch(short, tmp_path):
    pytest.importorskip("onnxruntime")
    train_ecog, train_flex, test_ecog = short
    decoder = OnlineDecoder.fit(train_ecog, train_flex, "lstm", n_electrodes=3, train_cfg=FAST)
    torch_out = decoder.decode(test_ecog)
    onnx_out = decoder.use_onnx(tmp_path / "decoder.onnx").decode(test_ecog)
    np.testing.assert_allclose(onnx_out, torch_out, rtol=1e-3, atol=1e-3)
