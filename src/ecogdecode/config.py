"""All tunable settings in one place. Values are fixed before looking at the test set."""
from __future__ import annotations

from dataclasses import dataclass, field

FS = 1000  # sampling rate of ECoG and (super-sampled) finger flexion, Hz


@dataclass(frozen=True)
class FeatureConfig:
    """Causal ECoG features computed every ``hop`` samples over a trailing window."""

    bands: tuple[tuple[float, float], ...] = ((8, 12), (18, 24), (75, 115), (125, 159), (159, 175))
    window_ms: int = 100  # trailing window; 100 ms beat 200 and 300 ms on validation (subject 1)
    hop: int = 40  # 40 ms, the native rate of the data glove
    line_freq: float = 60.0  # recordings were made in the USA
    notch_harmonics: tuple[int, ...] = (1, 2, 3)  # 60, 120, 180 Hz
    filter_order: int = 4
    reference: str = "median"  # "median" (robust to a bad electrode) or "average"
    artefact_z: float = 6.0  # features beyond this many training SDs mark a channel as artefactual
    include_lmp: bool = True  # local motor potential (trailing mean of the re-referenced signal)
    offline: bool = False  # True: zero-phase (forward-backward) filters and windows centred on each step, i.e.
    # the features an offline analysis would use; they need samples after the step and cannot run live
    fs: int = FS
    names: tuple[str, ...] = field(init=False, default=())

    def __post_init__(self) -> None:
        names = (("lmp",) if self.include_lmp else ()) + tuple(f"{lo:g}-{hi:g}Hz" for lo, hi in self.bands)
        object.__setattr__(self, "names", names)

    @property
    def n_per_channel(self) -> int:
        return len(self.names)


DEFAULT_FEATURES = FeatureConfig()

# finer high-gamma resolution (FingerFlex used dozens of wavelet bands; here eight 20 Hz bands up to 190 Hz)
FINE_HG_FEATURES = FeatureConfig(bands=((8, 12), (18, 24), (70, 90), (90, 110), (110, 130), (130, 150), (150, 170),
                                        (170, 190)))

# Liang and Bougrain (2012): amplitude modulation in three bands, 40 ms non-overlapping windows
LIANG_FEATURES = FeatureConfig(bands=((1, 60), (60, 100), (100, 199)), window_ms=40, include_lmp=False)


@dataclass(frozen=True)
class TrainConfig:
    """Training of the neural decoders (random windows, early stopping on validation)."""

    seq_len: int = 256  # steps per training window (about 10 s)
    burn_in: int = 50  # leading steps of each window excluded from the loss
    batch_size: int = 16
    windows_per_epoch: int = 128
    max_epochs: int = 60
    patience: int = 10
    val_context: int = 750  # warm-up steps (30 s) before the validation block
    lr: float = 1e-3
    weight_decay: float = 1e-2
    corr_weight: float = 0.0  # weight of (1 - Pearson r) added to the MSE loss; r is the competition metric
    channel_dropout: float = 0.0  # probability of silencing each electrode in a training window
    weight_ema: float = 0.0  # if > 0, keep an exponential moving average of the weights (one model, smoother optimum)
    gain_jitter: float = 0.0  # SD of a random per-electrode offset added to all its (standardised, log-power)
    # features in each training window: simulates slow gain and impedance changes between sessions
    feature_noise: float = 0.0  # SD of white noise added to every feature during training
    time_warp: float = 0.0  # if > 0, each training batch is resampled in time by a random factor within
    # [1/(1 + time_warp), 1 + time_warp] (features and targets together): movements at other speeds
    schedule: str = "constant"  # "constant": early stopping on validation picks the epoch count;
    # "cosine": cosine learning-rate decay over exactly max_epochs, final weights kept, no epoch selection
    seed: int = 0


@dataclass(frozen=True)
class Protocol:
    """The evaluation protocol shared by every approach."""

    val_fraction: float = 0.2  # last 20 % of the training segment is the validation block
    warmup_steps: int = 25  # first second of features is discarded (filters settling)
    seeds: tuple[int, ...] = (0, 1, 2)
    refit: bool = True  # retrain on the whole training segment after selection on validation
    adaptive_norm_s: float = 0.0  # if > 0, features are re-normalised online with running statistics over this
    # many seconds (tracks slow drift of the recording; 0 = fixed training statistics)
    smoothing_ms: tuple[float, ...] = (0.0, 40.0, 80.0, 120.0, 160.0, 240.0, 320.0)  # causal output smoothing
    # time constants tried on validation (0 = none); the best one becomes part of the decoder


DEFAULT_PROTOCOL = Protocol()
