"""Display helpers used by the videos: resting-level removal, the hand geometry and video rendering."""

import shutil
import subprocess

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from ecogdecode.handviz import HandArtist, HandGeometry, display_flexion, rest_baseline  # noqa: E402


def _bumps(n: int, rng: np.random.Generator) -> np.ndarray:
    x = np.zeros(n)
    for start in rng.choice(n - 60, size=n // 250, replace=False):
        x[start : start + 50] += np.sin(np.linspace(0, np.pi, 50))
    return x


def test_drifting_rest_level_is_removed():
    rng = np.random.default_rng(0)
    n = 5000
    flex = np.stack([_bumps(n, rng) for _ in range(5)], 1)
    drift = np.linspace(-1, 1, n)[:, None] * np.array([0, 0, 1.0, 0, 0])  # middle finger's rest level wanders
    (rec,) = display_flexion(flex + drift)
    at_rest = flex[:, 2] == 0
    assert abs(np.median(rec[at_rest, 2])) < 0.05
    assert abs(np.median(rec[at_rest, 2][:500]) - np.median(rec[at_rest, 2][-500:])) < 0.05


def test_decoded_baseline_is_causal():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(800, 5))
    changed = x.copy()
    changed[500:] += 3.0
    a, b = rest_baseline(x, 20.0, causal=True), rest_baseline(changed, 20.0, causal=True)
    np.testing.assert_array_equal(a[:500], b[:500])


def test_hand_geometry_and_drawing():
    for flex in (np.zeros(5), np.ones(5), np.array([1.0, 0, 0.5, 0, 1.0])):
        geo = HandGeometry(flex)
        assert len(geo.digits) == 5
        assert geo.silhouette().is_valid and geo.silhouette().area > 1.0
    assert HandGeometry(np.ones(5)).digits[0]["depth"] > HandGeometry(np.zeros(5)).digits[0]["depth"]
    fig, ax = plt.subplots()
    ax.set_xlim(-2.4, 2.4)
    ax.set_ylim(-1.5, 3.2)
    hand = HandArtist(ax, "#cccccc", "#333333", "#ffffff", ghost=True)
    hand.update(np.full(5, 0.5), ghost_flex=np.zeros(5))
    fig.canvas.draw()
    plt.close(fig)


@pytest.mark.skipif(not (shutil.which("ffmpeg") and shutil.which("ffprobe")), reason="needs ffmpeg and ffprobe")
def test_render_without_fork_gives_every_frame(tmp_path, monkeypatch):
    """On Windows there is no "fork": frames are then drawn in this process, and none may be lost."""
    from ecogdecode import render

    def build():
        fig, ax = plt.subplots(figsize=(1.6, 0.9), dpi=100)
        (line,) = ax.plot([], [])
        ax.set_xlim(0, 1)
        ax.set_ylim(-1, 1)

        def update(i):
            x = np.linspace(0, 1, 20)
            line.set_data(x, np.sin(6 * x + i))

        return fig, update

    monkeypatch.setattr(render.mp, "get_all_start_methods", lambda: ["spawn"])
    out = render.render_mp4(build, 12, tmp_path / "clip.mp4", jobs=2)
    frames = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-count_frames",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=nb_read_frames",
            "-of",
            "csv=p=0",
            str(out),
        ],
        capture_output=True,
        text=True,
    )
    assert frames.stdout.strip() == "12"
