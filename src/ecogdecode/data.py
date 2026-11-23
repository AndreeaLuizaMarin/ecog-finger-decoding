"""Loading BCI Competition IV data set 4: individual finger flexion from human ECoG.

Each subject file (``subX_comp.mat``) holds a continuous recording split in time:
the first two thirds (``train_data``, ``train_dg``) and the final third (``test_data``).
The finger flexion of the final third is released separately (``true_labels.zip``).
ECoG is sampled at 1 kHz; the data-glove flexion was recorded at about 25 Hz and
super-sampled to 1 kHz by the organisers.

The data are not redistributed with this repository: download them from
https://www.bbci.de/competition/iv/ (see README).
"""
from __future__ import annotations

import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.io

from .config import FS  # noqa: F401  (re-exported)
FINGERS = ("thumb", "index", "middle", "ring", "little")
SCORED_FINGERS = (0, 1, 2, 4)  # official metric excludes the ring finger (correlated with middle/little)
SUBJECTS = (1, 2, 3)


@dataclass
class SubjectData:
    """One subject's continuous recording, split as in the competition."""

    subject: int
    train_ecog: np.ndarray  # (n_train_samples, n_channels), float32
    train_flex: np.ndarray  # (n_train_samples, 5), float32
    test_ecog: np.ndarray  # (n_test_samples, n_channels), float32
    test_flex: np.ndarray | None  # (n_test_samples, 5), float32, or None if labels are missing

    @property
    def n_channels(self) -> int:
        return self.train_ecog.shape[1]

    @property
    def n_train(self) -> int:
        return self.train_ecog.shape[0]

    @property
    def ecog(self) -> np.ndarray:
        """The full continuous recording (train segment followed by test segment)."""
        return np.concatenate([self.train_ecog, self.test_ecog], axis=0)


def _loadmat(path: Path) -> dict[str, np.ndarray]:
    """Read a MATLAB file, including v7.3 (HDF5) files."""
    try:
        return {k: v for k, v in scipy.io.loadmat(path).items() if not k.startswith("__")}
    except NotImplementedError:  # MATLAB v7.3 files are HDF5
        import h5py

        with h5py.File(path, "r") as f:
            return {k: np.array(f[k]).T for k in f.keys()}


def _locate(data_dir: Path, filename: str) -> Path:
    """Find ``filename`` under ``data_dir``, extracting it from a zip archive there if needed."""
    hits = sorted(p for p in data_dir.rglob(filename) if p.is_file())
    if hits:
        return hits[0]
    for archive in sorted(data_dir.rglob("*.zip")):
        with zipfile.ZipFile(archive) as zf:
            for member in zf.namelist():
                if Path(member).name == filename:
                    target_dir = data_dir / "extracted"
                    return Path(zf.extract(member, target_dir))
    raise FileNotFoundError(
        f"{filename} not found in {data_dir} (or in any zip archive there). "
        "Download data set 4 and its true labels from https://www.bbci.de/competition/iv/."
    )


def _pick(mat: dict[str, np.ndarray], preferred: tuple[str, ...], n_cols: int | None = None) -> np.ndarray:
    """Return the first preferred variable present, else the only 2-D array with ``n_cols`` columns."""
    for name in preferred:
        if name in mat:
            return np.asarray(mat[name])
    candidates = [v for v in mat.values() if isinstance(v, np.ndarray) and v.ndim == 2
                  and (n_cols is None or v.shape[1] == n_cols)]
    if len(candidates) == 1:
        return candidates[0]
    raise KeyError(f"Could not identify the variable among {sorted(mat)} (wanted one of {preferred}).")


def load_subject(data_dir: str | Path, subject: int, require_test_labels: bool = True) -> SubjectData:
    """Load one subject's ECoG and finger flexion from ``data_dir``."""
    data_dir = Path(data_dir)
    comp = _loadmat(_locate(data_dir, f"sub{subject}_comp.mat"))
    train_ecog = _pick(comp, ("train_data",)).astype(np.float32)
    train_flex = _pick(comp, ("train_dg",), n_cols=5).astype(np.float32)
    test_ecog = _pick(comp, ("test_data",)).astype(np.float32)

    test_flex = None
    try:
        labels = _loadmat(_locate(data_dir, f"sub{subject}_testlabels.mat"))
        test_flex = _pick(labels, ("test_dg",), n_cols=5).astype(np.float32)
    except FileNotFoundError:
        if require_test_labels:
            raise

    if train_ecog.shape[0] != train_flex.shape[0]:
        raise ValueError("train_data and train_dg have different lengths")
    if test_flex is not None and test_flex.shape[0] != test_ecog.shape[0]:
        raise ValueError("test_data and the test labels have different lengths")
    if train_ecog.shape[1] != test_ecog.shape[1]:
        raise ValueError("train and test ECoG have different channel counts")
    return SubjectData(subject, train_ecog, train_flex, test_ecog, test_flex)
