"""Render matplotlib animations to H.264 in parallel chunks, and cut GIF excerpts for the README.

Every frame is drawn from data alone (no state carried between frames), so a long video can be split into chunks
that separate processes draw at the same time; the chunks are then joined without re-encoding.
"""
from __future__ import annotations

import multiprocessing as mp
import subprocess
import tempfile
from collections.abc import Callable
from pathlib import Path

import numpy as np

Build = Callable[[], tuple]  # returns (figure, update(frame_index))


def _render_chunk(build: Build, frames: np.ndarray, path: str, fps: int, crf: int) -> None:
    fig, update = build()
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgba", "-s", f"{w}x{h}",
           "-r", str(fps), "-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf),
           "-pix_fmt", "yuv420p", path]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for i in frames:
        update(int(i))
        fig.canvas.draw()
        proc.stdin.write(fig.canvas.buffer_rgba())
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError(f"ffmpeg failed on {path}")


def render_mp4(build: Build, n_frames: int, path: Path, jobs: int = 2, fps: int = 25, crf: int = 22) -> Path:
    """Draw frames 0..n_frames-1 with ``jobs`` processes and join them into one MP4.

    Parallel chunks need the "fork" start method (Linux, macOS); elsewhere (Windows) the frames are drawn in this
    process, one chunk after another, with the same result.
    """
    path = Path(path)
    if "fork" not in mp.get_all_start_methods():
        jobs = 1
    chunks = [c for c in np.array_split(np.arange(n_frames), max(1, jobs)) if len(c)]
    with tempfile.TemporaryDirectory() as tmp:
        parts = [str(Path(tmp) / f"part{i:02d}.mp4") for i in range(len(chunks))]
        if len(chunks) == 1:
            _render_chunk(build, chunks[0], parts[0], fps, crf)
        else:
            ctx = mp.get_context("fork")
            procs = [ctx.Process(target=_render_chunk, args=(build, c, p, fps, crf)) for c, p in zip(chunks, parts)]
            for p in procs:
                p.start()
            for p in procs:
                p.join()
                if p.exitcode != 0:
                    raise RuntimeError("a render process failed")
        listing = Path(tmp) / "parts.txt"
        listing.write_text("".join(f"file '{Path(p).as_posix()}'\n" for p in parts))
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(listing),
                        "-c", "copy", "-movflags", "+faststart", str(path)], check=True)
    return path


def gif_excerpt(mp4: Path, gif: Path, start_s: float, seconds: float, fps: int = 10, width: int = 800) -> Path:
    """A short GIF cut from the video (palette per clip, for a small file)."""
    palette = Path(gif).with_suffix(".palette.png")
    vf = f"fps={fps},scale={width}:-1:flags=lanczos"
    clip = ["-ss", f"{start_s}", "-t", f"{seconds}", "-i", str(mp4)]
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *clip, "-vf",
                    f"{vf},palettegen=max_colors=128:stats_mode=diff", str(palette)], check=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *clip, "-i", str(palette), "-lavfi",
                    f"{vf}[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=5:diff_mode=rectangle", str(gif)],
                   check=True)
    palette.unlink()
    return Path(gif)
