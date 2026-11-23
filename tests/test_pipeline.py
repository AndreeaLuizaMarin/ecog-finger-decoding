"""End-to-end checks on synthetic data: loading, the split, every approach, live replay."""
import dataclasses

import numpy as np
import pytest
import scipy.io

from ecogdecode.approaches import APPROACHES, CLOSED_FORM, NETWORKS, run_network, run_pretrained
from ecogdecode.config import Protocol, TrainConfig
from ecogdecode.data import load_subject
from ecogdecode.pipeline import evaluate, keep_electrodes, prepare, top_electrodes, with_test_labels

FAST = TrainConfig(seq_len=64, burn_in=16, windows_per_epoch=32, max_epochs=3, patience=3)


def test_load_subject_from_mat(tmp_path, synthetic_subject):
    s = synthetic_subject
    scipy.io.savemat(tmp_path / "sub1_comp.mat", {"train_data": s.train_ecog, "train_dg": s.train_flex,
                                                   "test_data": s.test_ecog})
    scipy.io.savemat(tmp_path / "sub1_testlabels.mat", {"test_dg": s.test_flex})
    loaded = load_subject(tmp_path, 1)
    np.testing.assert_array_equal(loaded.test_flex, s.test_flex)


def test_split_is_chronological(synthetic_subject):
    prep = prepare(synthetic_subject)
    assert prep.fit_rows.stop == prep.val_rows.start and prep.val_rows.stop == prep.test_rows.start
    assert prep.ends[prep.val_rows.stop - 1] < synthetic_subject.n_train <= prep.ends[prep.test_rows.start]
    assert prep.test_times[0] < 0 <= prep.test_times[1]


def test_registry_is_complete():
    for key, a in APPROACHES.items():
        assert key in (CLOSED_FORM if a.trainer == "closed_form" else NETWORKS) or a.trainer in ("pretrained", "fingerflex")


@pytest.fixture(scope="module")
def prepared(synthetic_subject):
    return prepare(synthetic_subject)


def test_electrode_selection_sees_only_the_given_rows(prepared):
    rows, npc = prepared.fit_rows, prepared.cfg.n_per_channel
    chosen = top_electrodes(prepared, rows, 3)
    changed = dataclasses.replace(prepared, x=prepared.x.copy())
    changed.x[rows.stop:] = np.random.default_rng(0).standard_normal(changed.x[rows.stop:].shape)
    np.testing.assert_array_equal(top_electrodes(changed, rows, 3), chosen)
    kept = keep_electrodes(prepared, chosen)
    np.testing.assert_array_equal(kept.x[:, :npc], prepared.x[:, chosen[0] * npc:(chosen[0] + 1) * npc])
    assert kept.x.shape[1] == 3 * npc


@pytest.mark.parametrize("key", list(CLOSED_FORM))
def test_closed_form_approaches_decode_synthetic_data(key, prepared, synthetic_subject):
    result = CLOSED_FORM[key](prepared, synthetic_subject, Protocol())
    assert np.isfinite(result.val_score)
    assert evaluate(prepared, result.pred, synthetic_subject.test_flex)["score"] > 0.3


@pytest.mark.parametrize("key", list(NETWORKS))
def test_networks_train_and_refit(key, prepared, synthetic_subject):
    result = run_network(prepared, key, seed=0, train_cfg=FAST)
    assert result.selected["epochs"] >= 1 and np.all(np.isfinite(result.pred))


@pytest.mark.parametrize("possm", [False, True])
def test_pretraining_runs(prepared, possm):
    other = keep_electrodes(dataclasses.replace(prepared, subject=2), np.arange(5))  # patients differ in electrodes
    out = run_pretrained({1: prepared, 2: other}, seed=0, steps=5, train_cfg=FAST, possm=possm)
    assert set(out) == {1, 2} and all(np.all(np.isfinite(r.pred)) for r in out.values())


@pytest.mark.parametrize("key", ["lstm", "s4d", "gated_lfp", "possm"])
def test_live_replay_matches_offline(key, synthetic_subject, tmp_path):
    pytest.importorskip("onnxruntime")
    from ecogdecode.data import SubjectData
    from ecogdecode.streaming import replay_online

    short = SubjectData(0, synthetic_subject.train_ecog[:12_000], synthetic_subject.train_flex[:12_000],
                        synthetic_subject.test_ecog[:4_000], synthetic_subject.test_flex[:4_000])
    prep = prepare(short)
    res = run_network(prep, key, seed=0, train_cfg=FAST)
    check = replay_online(res.model, res.x, res.x_scaler, short.ecog, prep.cfg, tmp_path / f"{key}.onnx")
    assert check["max_abs_diff_vs_offline"] < 1e-3


def test_test_labels_stay_out_of_training_arrays(synthetic_subject):
    prep = prepare(synthetic_subject)
    assert np.isnan(prep.y[prep.test_rows]).all() and not np.isnan(prep.y[prep.train_rows]).any()
    full = with_test_labels(prep, synthetic_subject.test_flex)
    at_steps = synthetic_subject.test_flex[prep.ends[prep.test_rows] - prep.n_train_samples]
    np.testing.assert_array_equal(full.y[prep.test_rows], at_steps)
    assert np.isnan(prep.y[prep.test_rows]).all()  # the original is unchanged
