import numpy as np
import pytest
import torch
from scipy import signal

from ecogdecode.data import FS, SubjectData

# PyTorch's default (every logical core) makes the many small operations of these tests very slow on hybrid CPUs
torch.set_num_threads(4)


def make_synthetic_subject(n_train=40_000, n_test=20_000, n_channels=8, seed=0) -> SubjectData:
    """Fake recording in which high-gamma amplitude on some channels tracks finger flexion."""
    rng = np.random.default_rng(seed)
    n = n_train + n_test
    # smooth, non-negative flexion traces (about 1 Hz bandwidth)
    b, a = signal.butter(2, 1.0, fs=FS)
    flex = np.abs(signal.filtfilt(b, a, rng.standard_normal((n, 5)), axis=0))
    flex /= flex.std(axis=0)
    noise = rng.standard_normal((n, n_channels))
    b_hg, a_hg = signal.butter(4, (80, 150), btype="bandpass", fs=FS)
    gamma = signal.lfilter(b_hg, a_hg, rng.standard_normal((n, n_channels)), axis=0)
    weights = np.zeros((5, n_channels))
    weights[np.arange(5), np.arange(5) % n_channels] = 1.0
    ecog = 0.5 * noise + gamma * (1 + 2 * flex @ weights)
    ecog, flex = ecog.astype(np.float32), flex.astype(np.float32)
    return SubjectData(0, ecog[:n_train], flex[:n_train], ecog[n_train:], flex[n_train:])


@pytest.fixture(scope="session")
def synthetic_subject() -> SubjectData:
    return make_synthetic_subject()
