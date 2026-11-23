"""Every decoding approach behind one interface, run under one protocol.

    results = run("lstm", preps, datas)      # {subject: [RunResult for each seed]}

Protocol (``config.Protocol``), identical for all approaches:
1. features, scalers and the artefact rule are fitted on the training rows only;
2. settings (penalties, history, electrodes, epochs) are chosen on the validation block;
3. the chosen model is retrained on the whole training segment (``refit``);
4. networks are trained with several seeds; each seed's model is scored on its own (no averaging).
"""

from __future__ import annotations

import copy
from collections.abc import Callable
from dataclasses import dataclass, field, replace

import numpy as np
import torch

from . import fingerflex as ff
from .config import DEFAULT_PROTOCOL, LIANG_FEATURES, Protocol, TrainConfig
from .data import SubjectData
from .evaluation import causal_ema, choose_smoothing, score
from .features import Standardiser, extract_features, mask_artefact_channels
from .models import (
    KalmanDecoder,
    LaggedRidge,
    LiangBougrainDecoder,
    LSTMDecoder,
    MambaDecoder,
    ODEDecoder,
    POSSMDecoder,
    S4DDecoder,
    StateGatedDecoder,
    SwitchingLinearDecoder,
    TCNDecoder,
    TransientGatedDecoder,
    build_manifold_gated,
    finger_states,
    predict_sequence,
    select_ridge,
    select_switching,
    train_sequence_model,
)
from .pipeline import Prepared, keep_electrodes, standardise, top_electrodes
from .pretrain import pretrain_contrastive, pretrain_possm, pretrain_supervised


@dataclass(frozen=True)
class Approach:
    key: str
    number: str
    label: str
    family: str  # "competition", "classical", "deep", "new"
    trainer: str  # "closed_form", "network" or "pretrained"
    streamable: bool = False  # has a single-step form exported to ONNX for live replay
    train_cfg: TrainConfig = field(default_factory=TrainConfig)
    electrodes: dict[int, int] | None = None  # per subject: use only this many electrodes, those most related to
    # movement in the training rows (pipeline.top_electrodes); the numbers were chosen on validation


# The LSTM recipe chosen in the validation study (scripts/explore_validation.py): data augmentation on top of the
# default training settings. Gain jitter 0.5 scored 0.005 higher on validation than 0.3, within seed noise and with
# a larger seed-to-seed spread, so the milder setting was kept.
TUNED_LSTM = TrainConfig(gain_jitter=0.3, feature_noise=0.3)
# Round 5 of the validation study: approach 5 on the k electrodes most related to movement in the training rows,
# k from {4, 8, 16, 24, 32, 48, all} chosen per patient on validation.
ELECTRODES_5B = {1: 32, 2: 8, 3: 32}

APPROACHES: dict[str, Approach] = {
    a.key: a
    for a in [
        Approach(
            "liang_bougrain",
            "1",
            "Liang & Bougrain, 2008 winner (live re-implementation)",
            "competition",
            "closed_form",
        ),
        Approach(
            "switching_linear",
            "2",
            "Flamary & Rakotomamonjy, 2nd place (live re-implementation)",
            "competition",
            "closed_form",
        ),
        Approach("wiener", "3", "Wiener filter (lagged ridge regression)", "classical", "closed_form"),
        Approach("kalman", "4", "Kalman filter", "classical", "closed_form"),
        Approach(
            "lstm",
            "5",
            "LSTM (256 units, dropout 0.4, augmentation; tuned on validation)",
            "deep",
            "network",
            streamable=True,
            train_cfg=TUNED_LSTM,
        ),
        Approach(
            "lstm_selected",
            "5b",
            "LSTM on the most movement-related electrodes (number chosen per patient)",
            "deep",
            "network",
            train_cfg=TUNED_LSTM,
            electrodes=ELECTRODES_5B,
        ),
        Approach("s4d", "6", "S4D state-space model", "deep", "network", streamable=True),
        Approach("mamba", "7", "Mamba (selective state space)", "deep", "network"),
        Approach("neural_ode", "8", "Neural ODE (continuous-time dynamics)", "deep", "network"),
        Approach("tcn", "8b", "Causal temporal convolutional network (TCN)", "deep", "network"),
        Approach("gated_lfp", "9", "LFP-gated decoder (gate: LMP, mu, beta)", "new", "network", streamable=True),
        Approach("gated_all", "9b", "LFP-gated decoder, ablation: gate sees all bands", "new", "network"),
        Approach("gated_transient", "10", "Transient-gated decoder (Dekleva-inspired)", "new", "network"),
        Approach("pretrained_lstm", "11", "Contrastive pretraining + fine-tuned LSTM", "new", "pretrained"),
        Approach(
            "multipatient_lstm", "11b", "Multi-patient supervised pretraining + fine-tuned LSTM", "new", "pretrained"
        ),
        Approach("manifold_pca", "12", "Manifold-gated decoder, PCA manifold", "new", "network", train_cfg=TUNED_LSTM),
        Approach(
            "manifold_contrastive",
            "13",
            "Manifold-gated decoder, CEBRA-style manifold",
            "new",
            "network",
            train_cfg=TUNED_LSTM,
        ),
        Approach("fingerflex", "14", "FingerFlex (Lomtev et al., 2023), made causal", "deep", "fingerflex"),
        Approach(
            "possm",
            "15",
            "POSSM-style electrode-token decoder (after Ryoo et al., 2025)",
            "new",
            "network",
            streamable=True,
            train_cfg=TUNED_LSTM,
        ),
        Approach(
            "possm_pretrained",
            "15b",
            "POSSM-style decoder, multi-patient pretraining + fine-tuning",
            "new",
            "pretrained",
            train_cfg=TUNED_LSTM,
        ),
    ]
}


@dataclass
class RunResult:
    pred: np.ndarray  # (n_steps, 5) predictions over the whole recording, in flexion units
    val_score: float  # score on the validation block before the final refit (used for ranking)
    selected: dict  # the settings chosen on validation
    model: object = None
    x: np.ndarray | None = None  # standardised features the final model was run on
    x_scaler: Standardiser | None = None


# ----------------------------------------------------------------------------- closed-form approaches
def _smoothing(val_pred: np.ndarray, val_true: np.ndarray, protocol: Protocol) -> tuple[float, float]:
    return choose_smoothing(val_pred, val_true, protocol.smoothing_ms)


def _wiener(prep: Prepared, data: SubjectData, protocol: Protocol) -> RunResult:
    sc = standardise(prep, prep.fit_rows)
    best, _, val_pred = select_ridge(
        sc.x, sc.y, prep.fit_rows, prep.val_rows, (10, 15, 20, 25), np.logspace(2, 6.5, 10), score
    )
    tau, val_score = _smoothing(val_pred, sc.y[prep.val_rows], protocol)
    sc = standardise(prep, prep.train_rows)
    model = LaggedRidge(best["n_lags"], best["alpha"]).fit(sc.x, sc.y, prep.train_rows)
    pred = sc.y_scaler.inverse_transform(causal_ema(model.predict(sc.x), tau))
    return RunResult(pred, val_score, {**best, "smooth_ms": tau, "val_score": val_score}, model, sc.x, sc.x_scaler)


def _kalman(prep: Prepared, data: SubjectData, protocol: Protocol) -> RunResult:
    sc = standardise(prep, prep.fit_rows)
    grid, preds = [], {}
    for k in (8, 16, 32, 64):
        kf = KalmanDecoder(k).fit(sc.x[prep.fit_rows], sc.y[prep.fit_rows])
        preds[k] = kf.predict(sc.x[: prep.val_rows.stop])[prep.val_rows]
        grid.append({"n_components": k, "val_score": score(preds[k], sc.y[prep.val_rows])})
    best = max(grid, key=lambda g: g["val_score"])
    tau, val_score = _smoothing(preds[best["n_components"]], sc.y[prep.val_rows], protocol)
    sc = standardise(prep, prep.train_rows)
    model = KalmanDecoder(best["n_components"]).fit(sc.x[prep.train_rows], sc.y[prep.train_rows])
    pred = sc.y_scaler.inverse_transform(causal_ema(model.predict(sc.x), tau))
    return RunResult(pred, val_score, {**best, "smooth_ms": tau, "val_score": val_score}, model, sc.x, sc.x_scaler)


def _liang_features(data: SubjectData, prep: Prepared, rows: slice) -> np.ndarray:
    """Standardised amplitude modulation (mean squared signal per 40 ms) with the artefact rule applied."""
    log_am, ends = extract_features(data.ecog, LIANG_FEATURES)
    assert np.array_equal(ends, prep.ends)
    am = np.exp(log_am)
    _, bad = mask_artefact_channels(
        Standardiser().fit(log_am[rows]).transform(log_am), LIANG_FEATURES.n_per_channel, LIANG_FEATURES.artefact_z
    )
    am = Standardiser().fit(am[rows]).transform(am).reshape(len(am), -1, LIANG_FEATURES.n_per_channel)
    am[bad] = 0.0
    return am.reshape(len(am), -1)


def _liang_bougrain(prep: Prepared, data: SubjectData, protocol: Protocol) -> RunResult:
    am = _liang_features(data, prep, prep.fit_rows)
    ys = Standardiser().fit(prep.y[prep.fit_rows])
    y = ys.transform(np.nan_to_num(prep.y))
    model = LiangBougrainDecoder().select(am, y, prep.fit_rows).fit(am, y, prep.fit_rows)
    tau, val_score = _smoothing(model.predict(am[: prep.val_rows.stop])[prep.val_rows], y[prep.val_rows], protocol)
    selected = {"pairs_per_finger": [list(map(int, p)) for p in model.pairs_], "smooth_ms": tau, "val_score": val_score}
    # final model: same selected electrode-band pairs, refitted on the whole training segment
    am = _liang_features(data, prep, prep.train_rows)
    ys = Standardiser().fit(prep.y[prep.train_rows])
    y = ys.transform(np.nan_to_num(prep.y))
    model.fit(am, y, prep.train_rows)
    return RunResult(ys.inverse_transform(causal_ema(model.predict(am), tau)), val_score, selected, model, am)


def _switching_linear(prep: Prepared, data: SubjectData, protocol: Protocol) -> RunResult:
    npc = prep.cfg.n_per_channel
    sc = standardise(prep, prep.fit_rows)
    lmp = sc.x.reshape(len(sc.x), -1, npc)[..., 0]  # local motor potential per electrode
    best, _, val_pred = select_switching(sc.x, lmp, sc.y, prep.fit_rows, prep.val_rows, npc)
    tau, val_score = _smoothing(val_pred, sc.y[prep.val_rows], protocol)
    sc = standardise(prep, prep.train_rows)
    lmp = sc.x.reshape(len(sc.x), -1, npc)[..., 0]
    states = finger_states(sc.y, prep.train_rows)
    model = SwitchingLinearDecoder(npc, best["n_keep"], best["C"], best["alpha"])
    model.fit(sc.x, lmp, sc.y, states, prep.train_rows, model.channel_ranking(sc.x, states, prep.train_rows))
    pred = sc.y_scaler.inverse_transform(causal_ema(model.predict(sc.x, lmp), tau))
    return RunResult(pred, val_score, {**best, "smooth_ms": tau, "val_score": val_score}, model, sc.x, sc.x_scaler)


CLOSED_FORM: dict[str, Callable[[Prepared, SubjectData, Protocol], RunResult]] = {
    "liang_bougrain": _liang_bougrain,
    "switching_linear": _switching_linear,
    "wiener": _wiener,
    "kalman": _kalman,
}


# ----------------------------------------------------------------------------- networks
# builders receive the training rows (features, standardised targets), the features per electrode and the seed
NETWORKS: dict[str, Callable[[np.ndarray, np.ndarray, int, int], torch.nn.Module]] = {
    "lstm": lambda x, y, npc, seed: LSTMDecoder(x.shape[1], hidden=256, dropout=0.4),
    "lstm_selected": lambda x, y, npc, seed: LSTMDecoder(x.shape[1], hidden=256, dropout=0.4),
    "lstm128": lambda x, y, npc, seed: LSTMDecoder(x.shape[1]),  # the untuned reference of the validation study
    "s4d": lambda x, y, npc, seed: S4DDecoder(x.shape[1]),
    "mamba": lambda x, y, npc, seed: MambaDecoder(x.shape[1]),
    "neural_ode": lambda x, y, npc, seed: ODEDecoder(x.shape[1]),
    "tcn": lambda x, y, npc, seed: TCNDecoder(x.shape[1]),
    "possm": lambda x, y, npc, seed: POSSMDecoder(x.shape[1], npc),
    "gated_lfp": lambda x, y, npc, seed: StateGatedDecoder(x.shape[1], npc, gate_features="slow"),
    "gated_all": lambda x, y, npc, seed: StateGatedDecoder(x.shape[1], npc, gate_features="all"),
    "gated_transient": lambda x, y, npc, seed: TransientGatedDecoder.from_training_data(x, npc),
    "manifold_pca": lambda x, y, npc, seed: build_manifold_gated("pca", x, y, seed=seed, backbone="lstm", dropout=0.4),
    "manifold_contrastive": lambda x, y, npc, seed: build_manifold_gated(
        "contrastive", x, y, seed=seed, backbone="lstm", dropout=0.4
    ),
    # variants compared in the validation study
    "manifold_pca_leaky": lambda x, y, npc, seed: build_manifold_gated("pca", x, y, seed=seed),
    "manifold_contrastive_leaky": lambda x, y, npc, seed: build_manifold_gated("contrastive", x, y, seed=seed),
    "manifold_pca_lstm": lambda x, y, npc, seed: build_manifold_gated("pca", x, y, seed=seed, backbone="lstm"),
    "manifold_contrastive_lstm": lambda x, y, npc, seed: build_manifold_gated(
        "contrastive", x, y, seed=seed, backbone="lstm"
    ),
}


def run_network(
    prep: Prepared,
    key: str,
    seed: int,
    protocol: Protocol = DEFAULT_PROTOCOL,
    model: torch.nn.Module | None = None,
    train_cfg: TrainConfig | None = None,
    electrodes: int | None = None,
) -> RunResult:
    """Early stopping on validation, choice of output smoothing on validation, then (``protocol.refit``)
    retraining from the same initial weights for the chosen number of epochs on the whole training segment.
    ``electrodes`` keeps that many electrodes (approach 5b's rule) instead of the approach's per-patient count."""
    approach = APPROACHES.get(key, Approach(key, "", "", "", ""))
    cfg = replace(train_cfg or approach.train_cfg, seed=seed)
    n_keep = electrodes or (approach.electrodes or {}).get(prep.subject)
    full = prep
    if n_keep:  # electrodes chosen on the fit rows for model selection
        prep = keep_electrodes(full, top_electrodes(full, full.fit_rows, n_keep))
    npc = prep.cfg.n_per_channel
    adapt = protocol.adaptive_norm_s * 1000 / prep.cfg.hop  # seconds -> steps
    torch.manual_seed(seed)
    sc = standardise(prep, prep.fit_rows, adapt)
    if model is None:
        model = NETWORKS[key](sc.x[prep.fit_rows], sc.y[prep.fit_rows], npc, seed)
    initial = copy.deepcopy(model)
    history = train_sequence_model(model, sc.x, sc.y, prep.fit_rows, prep.val_rows, score, cfg, n_per_channel=npc)
    best = history[-1] if cfg.schedule == "cosine" else max(history, key=lambda h: h.get("val_score", -np.inf))
    ctx = max(prep.val_rows.start - cfg.val_context, 0)
    val_pred = predict_sequence(model, sc.x[ctx : prep.val_rows.stop])
    tau, val_score = choose_smoothing(
        val_pred, sc.y[prep.val_rows], protocol.smoothing_ms, context=prep.val_rows.start - ctx
    )
    selected = {
        "epochs": best["epoch"] + 1,
        "smooth_ms": tau,
        "val_score_unsmoothed": best["val_score"],
        "val_score": val_score,
        "seed": seed,
    }
    if protocol.refit:
        if n_keep:  # re-chosen on the whole training segment, like every other refitted quantity
            electrodes = top_electrodes(full, full.train_rows, n_keep)
            prep = keep_electrodes(full, electrodes)
            selected["electrodes"] = electrodes.tolist()
        sc = standardise(prep, prep.train_rows, adapt)
        model = initial
        if hasattr(model, "refit_to"):
            model.refit_to(sc.x[prep.train_rows], sc.y[prep.train_rows], seed=seed)
        train_sequence_model(
            model, sc.x, sc.y, prep.train_rows, None, score, cfg, epochs=best["epoch"] + 1, n_per_channel=npc
        )
    pred = sc.y_scaler.inverse_transform(causal_ema(predict_sequence(model, sc.x), tau))
    return RunResult(pred, val_score, selected, model, sc.x, sc.x_scaler)


def run_pretrained(
    preps: dict[int, Prepared],
    seed: int,
    protocol: Protocol = DEFAULT_PROTOCOL,
    steps: int = 600,
    train_cfg: TrainConfig | None = None,
    supervised: bool = False,
    possm: bool = False,
) -> dict[int, RunResult]:
    """Pretraining on every subject's fit rows, then per-subject fine-tuning.

    ``supervised=False``: contrastive, without labels (approach 11). ``supervised=True``: multi-patient
    training with labels, one shared core and per-patient input/output layers (approach 11b). ``possm=True``: the
    POSSM-style decoder trained on every patient, sharing every weight but the electrode embeddings (15b)."""
    scaled = {s: standardise(p, p.fit_rows) for s, p in preps.items()}
    rows = {s: p.fit_rows for s, p in preps.items()}
    if possm:
        npc = next(iter(preps.values())).cfg.n_per_channel
        decoders = pretrain_possm(
            {s: v.x for s, v in scaled.items()}, {s: v.y for s, v in scaled.items()}, rows, npc, seed=seed, steps=steps
        )
        losses = [float("nan"), float("nan")]
    elif supervised:
        decoders = pretrain_supervised(
            {s: v.x for s, v in scaled.items()}, {s: v.y for s, v in scaled.items()}, rows, seed=seed, steps=steps
        )
        losses = [float("nan"), float("nan")]
    else:
        decoders, losses = pretrain_contrastive({s: v.x for s, v in scaled.items()}, rows, seed=seed, steps=steps)
    out = {}
    for s, prep in preps.items():
        key = "possm_pretrained" if possm else "lstm"
        cfg = train_cfg or (APPROACHES[key].train_cfg if possm else TrainConfig())
        res = run_network(prep, key, seed, protocol, model=decoders[s], train_cfg=cfg)
        res.selected["pretrain_loss_first_last"] = [losses[0], losses[-1]]
        out[s] = res
    return out


def run_fingerflex(
    prep: Prepared,
    data: SubjectData,
    seed: int,
    protocol: Protocol = DEFAULT_PROTOCOL,
    cfg: ff.FingerFlexConfig | None = None,
) -> RunResult:
    """Approach 14: FingerFlex made causal (``fingerflex.py``), under the shared protocol.

    Its own features and training (every 2.56 s window at stride 1, 20 epochs, Adam, half MSE + half cosine
    distance), the epoch with the best validation score kept, a causal output smoother chosen on validation, then a
    refit on the whole training segment from the same initial weights for that many epochs. The 100 Hz output is
    read at the end of every 40 ms step (the latest output available then) and held, like every other decoder."""
    cfg = replace(cfg or ff.FingerFlexConfig(), seed=seed)
    ends100 = prep.ends // ff.DECIM  # 100 Hz sample of each step's end: its output uses ECoG up to that sample

    def rows100(rows: slice) -> slice:
        return slice(int(ends100[rows.start]), int(ends100[rows.stop - 1]) + 1)

    y100 = ff.glove_100hz(data.train_flex)

    def scaled(rows: slice) -> tuple[np.ndarray, np.ndarray, tuple[np.ndarray, np.ndarray]]:
        """Features normalised with statistics of the ECoG up to the end of ``rows`` (training data only), robust
        scaler and finger min-max scaling fitted on ``rows``."""
        x = ff.recording_features(data.ecog, offline=False, stats_end=int(rows.stop) * ff.DECIM)
        x = ff.RobustScaler().fit(x[rows]).transform(x)
        lo, hi = y100[rows].min(0), y100[rows].max(0)
        return x, ((y100 - lo) / (hi - lo)).astype(np.float32), (lo, hi)

    def to_steps(out100: np.ndarray) -> np.ndarray:
        return out100[np.minimum(ends100, len(out100) - 1)]

    fit = rows100(prep.fit_rows)
    x, y, _ = scaled(fit)
    torch.manual_seed(seed)
    model = ff.FingerFlexNet(x.shape[1], causal=True, dropout=cfg.dropout)
    initial = copy.deepcopy(model)
    best: dict = {"val_score": -np.inf}

    def on_epoch(epoch: int, m: torch.nn.Module) -> dict:
        steps = to_steps(ff.predict(m, x))
        val = float(score(steps[prep.val_rows], prep.y[prep.val_rows]))
        if val > best["val_score"]:
            best.update(val_score=val, epoch=epoch, steps=steps)
        return {"val_score": val, "stop": epoch - best["epoch"] >= cfg.patience}

    history = ff.train(model, x, y, fit, cfg, cfg.max_epochs, on_epoch)
    ctx = max(prep.val_rows.start - TrainConfig().val_context, 0)
    tau, val_score = choose_smoothing(
        best["steps"][ctx : prep.val_rows.stop],
        prep.y[prep.val_rows],
        protocol.smoothing_ms,
        context=prep.val_rows.start - ctx,
    )
    selected = {
        "epochs": best["epoch"] + 1,
        "smooth_ms": tau,
        "val_score_unsmoothed": best["val_score"],
        "val_score": val_score,
        "seed": seed,
        "val_by_epoch": [h["val_score"] for h in history],
    }
    del x, y
    train = rows100(prep.train_rows)
    x, y, (lo, hi) = scaled(train)
    model = initial
    ff.train(model, x, y, train, cfg, best["epoch"] + 1)
    pred = causal_ema(to_steps(ff.predict(model, x)) * (hi - lo) + lo, tau)
    return RunResult(pred, val_score, selected)


# ----------------------------------------------------------------------------- entry point
def run(
    key: str, preps: dict[int, Prepared], datas: dict[int, SubjectData], protocol: Protocol = DEFAULT_PROTOCOL
) -> dict[int, list[RunResult]]:
    """Run one approach on every subject. Closed-form approaches are deterministic (one result);
    networks give one result per seed."""
    approach = APPROACHES[key]
    if approach.trainer == "closed_form":
        return {s: [CLOSED_FORM[key](p, datas[s], protocol)] for s, p in preps.items()}
    results: dict[int, list[RunResult]] = {s: [] for s in preps}
    for seed in protocol.seeds:
        if approach.trainer == "pretrained":
            per_subject = run_pretrained(
                preps, seed, protocol, supervised=key == "multipatient_lstm", possm=key == "possm_pretrained"
            )
        elif approach.trainer == "fingerflex":
            per_subject = {s: run_fingerflex(p, datas[s], seed, protocol) for s, p in preps.items()}
        else:
            per_subject = {s: run_network(p, key, seed, protocol) for s, p in preps.items()}
        for s, res in per_subject.items():
            results[s].append(res)
    return results
