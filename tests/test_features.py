import numpy as np

from ecogdecode.config import FeatureConfig
from ecogdecode.features import Standardiser, extract_features, mask_artefact_channels, step_ends
from ecogdecode.streaming import StreamingFeatureExtractor


def test_shapes(synthetic_subject):
    cfg = FeatureConfig()
    x, ends = extract_features(synthetic_subject.train_ecog, cfg)
    assert x.shape == (len(ends), synthetic_subject.n_channels * cfg.n_per_channel)
    assert np.all(np.isfinite(x))
    np.testing.assert_array_equal(ends, step_ends(synthetic_subject.n_train, cfg.hop))


def test_features_are_causal(synthetic_subject):
    """Changing the signal after sample t must not change any feature that ends before t."""
    ecog = synthetic_subject.train_ecog[:20_000].copy()
    x_ref, ends = extract_features(ecog)
    cut = 12_345
    ecog[cut:] = np.random.default_rng(1).standard_normal(ecog[cut:].shape)
    x_mod, _ = extract_features(ecog)
    before = ends < cut
    np.testing.assert_allclose(x_mod[before], x_ref[before], rtol=1e-5, atol=1e-6)
    assert not np.allclose(x_mod[~before], x_ref[~before])


def test_streaming_extractor_matches_batch(synthetic_subject):
    ecog = synthetic_subject.train_ecog[:8_000]
    x_batch, _ = extract_features(ecog)
    stream = StreamingFeatureExtractor(ecog.shape[1])
    x_stream = np.stack([stream.process(ecog[i : i + 40]) for i in range(0, len(ecog), 40)])
    np.testing.assert_allclose(x_stream, x_batch, rtol=1e-4, atol=1e-4)


def test_artefact_channel_is_masked_and_does_not_leak():
    """A channel that suddenly becomes huge is masked and barely affects the other channels."""
    rng = np.random.default_rng(0)
    n_ch = 32
    ecog = rng.standard_normal((20_000, n_ch)).astype(np.float32)
    cfg = FeatureConfig()
    x_clean, _ = extract_features(ecog, cfg)
    ecog_bad = ecog.copy()
    ecog_bad[10_000:, 2] *= 300  # channel 2 turns artefactual halfway through
    x_bad, ends = extract_features(ecog_bad, cfg)
    scaler = Standardiser().fit(x_clean[:200])
    masked, bad = mask_artefact_channels(scaler.transform(x_bad), cfg.n_per_channel, cfg.artefact_z)
    late = ends > 10_200
    others = [c for c in range(n_ch) if c != 2]
    assert bad[late, 2].all() and bad[late][:, others].mean() < 0.01
    clean = scaler.transform(x_clean).reshape(len(ends), n_ch, -1)[late][:, others].ravel()
    kept = masked.reshape(len(ends), n_ch, -1)[late][:, others].ravel()
    assert np.corrcoef(clean, kept)[0, 1] > 0.95
