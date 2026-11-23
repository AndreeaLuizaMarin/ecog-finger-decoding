"""Display helpers for the animations: a shaded 3-D hand driven by five flexion values, and the display scaling.

The hand is a simple kinematic model (palm towards the viewer; each finger a chain of three tapered segments; the
thumb swings across the palm as it flexes), projected with weak perspective, lit by one light and drawn back to
front. Silhouettes are unions of the segments (shapely), so joints are smooth and fingers emerge from the palm
without a seam.

Display scaling (``display_flexion``) is for drawing only; every score is computed on the raw glove and the raw
decoder output.
"""

from __future__ import annotations

import logging

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D
from matplotlib.patches import PathPatch, Polygon
from matplotlib.path import Path
from scipy.interpolate import splev, splprep
from shapely import affinity
from shapely.geometry import MultiPoint, Point
from shapely.geometry import Polygon as SPolygon
from shapely.ops import unary_union

STEP_S = 0.04
FINGERS = tuple(f.capitalize() for f in ("thumb", "index", "middle", "ring", "little"))
BG, PANEL, TEXT, MUTED, GRID = "#0f1115", "#161a21", "#ecebe6", "#9a988f", "#2a2f39"
REC, REC_EDGE, REC_HI = "#d6cdbf", "#5f574c", "#f7f2ea"
DEC, DEC_EDGE, DEC_HI = "#3f8be6", "#123769", "#b9d6fa"
REC_TRACE, DEC_TRACE = "#cfc9be", "#5aa0ee"

LIVE, LIVE_EDGE, LIVE_HI, LIVE_TRACE = DEC, DEC_EDGE, DEC_HI, DEC_TRACE
OFF, OFF_EDGE, OFF_HI, OFF_TRACE = "#e39a35", "#6e420a", "#f9ddb3", "#f0ad55"

_FONT = "Inter" if any(f.name == "Inter" for f in font_manager.fontManager.ttflist) else "DejaVu Sans"
plt.rcParams.update(
    {
        "font.family": _FONT,
        "text.color": TEXT,
        "axes.labelcolor": MUTED,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.edgecolor": GRID,
    }
)
# hand axes keep their y-limits and let matplotlib widen x for an equal aspect; that is intended, so no log message
logging.getLogger("matplotlib.axes._base").setLevel(logging.ERROR)


# ----------------------------------------------------------------------------- display scaling
def rest_baseline(x: np.ndarray, window_s: float, causal: bool, q: float = 10.0) -> np.ndarray:
    """Running low percentile per column: the resting level. Causal = trailing window (past samples only)."""
    n, w = len(x), max(1, int(round(window_s / STEP_S)))
    out = np.empty(x.shape, dtype=float)
    for t in range(n):
        lo, hi = (max(0, t - w + 1), t + 1) if causal else (max(0, t - w // 2), min(n, t + w // 2 + 1))
        out[t] = np.percentile(x[lo:hi], q, axis=0)
    return out


def display_flexion(
    recorded: np.ndarray, decoded: tuple[np.ndarray, ...] = (), rec_window_s: float = 8.0, dec_window_s: float = 20.0
) -> list[np.ndarray]:
    """Flexion in display units per finger: 0 = rest, 1 = a full recorded flexion (99th percentile).

    The glove's resting reading wanders slowly (in patient 3 the middle finger's rest level moves by about half its
    flexion range over the test segment), so the recorded trace has its own resting level removed: a running 10th
    percentile over a centred 8 s window, smoothed over 2 s. The decoders' resting outputs drift less, so each decoded
    trace has its resting level removed with a trailing (causal) 20 s window, which a live display could also do.
    Decoded traces share the recorded trace's scale, so a decoder that under-shoots is drawn under-shooting.
    """
    base = rest_baseline(recorded, rec_window_s, causal=False)
    k = int(round(2.0 / STEP_S))
    pad = np.pad(base, ((k // 2, k - 1 - k // 2), (0, 0)), mode="edge")
    base = np.stack([np.convolve(pad[:, j], np.ones(k) / k, mode="valid") for j in range(base.shape[1])], 1)
    rec = recorded - base
    scale = np.percentile(rec, 99, axis=0)
    return [rec / scale] + [(d - rest_baseline(d, dec_window_s, causal=True)) / scale for d in decoded]


# ----------------------------------------------------------------------------- 3-D hand model
def rot(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rotation matrix about a unit axis (Rodrigues)."""
    axis = axis / np.linalg.norm(axis)
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle) * k + (1 - np.cos(angle)) * k @ k


def _unit(v) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v)


# palm towards the viewer: x across the palm (thumb side negative), y along the fingers, z towards the viewer
X_AXIS, Z_AXIS = np.array([1.0, 0, 0]), np.array([0, 0, 1.0])
FINGER_MCP = {1: (-0.675, 0.97), 2: (-0.225, 1.03), 3: (0.225, 0.98), 4: (0.64, 0.84)}
FINGER_LEN = {1: (0.80, 0.48, 0.30), 2: (0.88, 0.55, 0.33), 3: (0.83, 0.52, 0.32), 4: (0.64, 0.38, 0.26)}
FINGER_RAD = {
    1: (0.222, 0.2, 0.181, 0.158),
    2: (0.232, 0.208, 0.188, 0.163),
    3: (0.22, 0.198, 0.179, 0.155),
    4: (0.19, 0.171, 0.155, 0.136),
}  # at MCP, PIP, DIP, tip
FINGER_SPLAY = {1: -6.0, 2: -1.0, 3: 4.0, 4: 10.0}  # degrees, relaxed with flexion
JOINT_MAX = (80.0, 100.0, 65.0)  # MCP, PIP, DIP at full flexion, degrees
THUMB_CMC = np.array([-0.44, -0.82, 0.06])
THUMB_LEN = (0.92, 0.6, 0.44)
THUMB_RAD = (0.39, 0.262, 0.235, 0.2)  # thenar mass at the CMC, then MCP, IP, tip
CAMERA = rot(X_AXIS, np.deg2rad(14)) @ rot(np.array([0, 1.0, 0]), np.deg2rad(-20))
LIGHT = _unit([-0.45, 0.6, 0.66])  # in camera coordinates: upper left, in front


def chains(flex: np.ndarray) -> list[dict]:
    """Joint positions, joint radii and per-segment palmar normals of thumb and fingers (hand coordinates)."""
    out = []
    f = float(flex[0])
    opp = rot(_unit([0.0, 0.2, 1.0]), np.deg2rad(-40 * f))  # the thumb swings across the palm...
    d = opp @ _unit([-0.62, 0.74, 0.26])
    bend = opp @ _unit([0.0, -0.05, 1.0])  # ...and its joints curl it towards the fingers
    pts, normals = [THUMB_CMC.copy()], []
    for length, ang in zip(THUMB_LEN, (0.0, 8 + 40 * f, 8 + 25 * f)):
        d = rot(bend, np.deg2rad(-ang)) @ d
        pts.append(pts[-1] + length * d)
        normals.append(_unit(np.cross(d, bend) * 0.45 + opp @ Z_AXIS))
    out.append({"pts": np.array(pts), "rad": THUMB_RAD, "normals": normals})
    for k in range(1, 5):
        splay = rot(Z_AXIS, np.deg2rad(-FINGER_SPLAY[k]) * (1 - 0.6 * flex[k]))
        pts, normals, angle = [np.array([*FINGER_MCP[k], 0.0])], [], 0.0
        for length, jmax in zip(FINGER_LEN[k], JOINT_MAX):
            angle += np.deg2rad(jmax) * flex[k]
            d = splay @ np.array([0, np.cos(angle), np.sin(angle)])
            pts.append(pts[-1] + length * d)
            normals.append(splay @ np.array([0, -np.sin(angle), np.cos(angle)]))
        out.append({"pts": np.array(pts), "rad": FINGER_RAD[k], "normals": normals})
    return out


def project(p: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Weak perspective: screen xy, depth (towards the viewer) and scale for points of shape (..., 3)."""
    c = np.asarray(p) @ CAMERA.T
    scale = 7.0 / (7.0 - c[..., 2])
    return c[..., :2] * scale[..., None], c[..., 2], scale


_CIRCLE = np.c_[
    np.cos(np.linspace(0, 2 * np.pi, 28, endpoint=False)), np.sin(np.linspace(0, 2 * np.pi, 28, endpoint=False))
]


def capsule(pa: np.ndarray, pb: np.ndarray, ra: float, rb: float) -> SPolygon:
    """Tapered capsule: convex hull of two discs."""
    return MultiPoint(np.r_[pa + ra * _CIRCLE, pb + rb * _CIRCLE]).convex_hull


def _smooth_closed(points: list[tuple[float, float]], n: int = 160) -> np.ndarray:
    p = np.array(points + points[:1])
    tck, _ = splprep([p[:, 0], p[:, 1]], s=0, per=True)
    return np.column_stack(splev(np.linspace(0, 1, n), tck))


def _smooth_open(points: list[tuple[float, float]], n: int = 40) -> np.ndarray:
    p = np.array(points)
    tck, _ = splprep([p[:, 0], p[:, 1]], s=0, k=min(3, len(p) - 1))
    return np.column_stack(splev(np.linspace(0, 1, n), tck))


def _flat(xy: np.ndarray) -> np.ndarray:
    return project(np.c_[xy, np.zeros(len(xy))])[0]


# static parts, already projected: palm, forearm, palm creases
PALM = SPolygon(
    _flat(
        _smooth_closed(
            [
                (0.0, -1.34),
                (0.6, -1.22),
                (0.9, -0.72),
                (0.99, -0.05),
                (0.92, 0.62),
                (0.7, 0.93),
                (0.25, 1.05),
                (-0.22, 1.1),
                (-0.66, 1.02),
                (-0.9, 0.78),
                (-0.98, 0.25),
                (-0.93, -0.45),
                (-0.7, -1.12),
            ]
        )
    )
)
FOREARM = SPolygon(_flat(np.array([(-0.7, -1.12), (-0.8, -3.6), (0.84, -3.6), (0.66, -1.15), (0.0, -1.0)])))
PALM_LINES = [
    _flat(_smooth_open(c))
    for c in (
        [(0.93, 0.42), (0.55, 0.5), (0.15, 0.6), (-0.32, 0.76)],  # distal transverse crease
        [(-0.9, 0.5), (-0.45, 0.36), (0.05, 0.2), (0.55, 0.02)],  # proximal transverse crease
        [(-0.88, 0.47), (-0.52, 0.16), (-0.33, -0.3), (-0.26, -0.78), (-0.14, -1.12)],  # thenar crease
        [(-0.42, -1.27), (0.0, -1.31), (0.44, -1.22)],
    )
]  # wrist crease


def tone(fill: str, edge: str, hi: str, b: float) -> tuple:
    """Shade from dark (b = 0) through the fill colour (b = 0.65) to the highlight colour (b = 1)."""
    f, e, h = (np.array(to_rgb(c)) for c in (fill, edge, hi))
    c = e + (f - e) * (0.35 + 0.65 * b / 0.65) if b < 0.65 else f + (h - f) * 0.55 * (b - 0.65) / 0.35
    return tuple(np.clip(c, 0, 1))


_PALM_LIGHT = float((CAMERA @ Z_AXIS) @ LIGHT)


def brightness(normal_hand: np.ndarray) -> tuple[float, float]:
    """Brightness of the visible face (0.65 = as lit as the palm) and how much the palm side faces the viewer."""
    n = CAMERA @ normal_hand
    facing = float(n[2])
    n_vis = n if facing >= 0 else -n

    def lit(c):
        return 0.4 + 0.6 * max(0.0, c)  # ambient + diffuse

    return float(np.clip(0.65 * lit(n_vis @ LIGHT) / lit(_PALM_LIGHT), 0.0, 1.0)), facing


def fade_bottom(ax, y_solid: float, y_clear: float, color: str = BG) -> None:
    """Fade the lower part of an axes into the background (the forearm leaves the picture gently)."""
    rgba = np.zeros((64, 1, 4))
    rgba[..., :3] = to_rgb(color)
    rgba[:, 0, 3] = np.linspace(1, 0, 64)
    x0, x1 = ax.get_xlim()
    ax.imshow(
        rgba,
        extent=(x0 - 10, x1 + 10, y_solid, y_clear),
        origin="lower",
        aspect="auto",
        zorder=50,
        interpolation="bilinear",
    )
    ax.add_patch(
        Polygon(
            [(x0 - 10, y_solid - 10), (x1 + 10, y_solid - 10), (x1 + 10, y_solid), (x0 - 10, y_solid)],
            fc=color,
            ec="none",
            zorder=50,
        )
    )


def _lines_xy(geom) -> np.ndarray:
    """Coordinates of a (multi)line geometry, joined with NaN breaks for one Line2D."""
    if geom.is_empty:
        return np.empty((0, 2))
    parts = getattr(geom, "geoms", [geom])
    out = []
    for g in parts:
        if g.geom_type in ("LineString", "LinearRing"):
            out += [np.asarray(g.coords), np.full((1, 2), np.nan)]
        elif g.geom_type in ("MultiLineString", "GeometryCollection"):
            out.append(_lines_xy(g))
    return np.vstack(out) if out else np.empty((0, 2))


def _poly_path(geom) -> Path:
    polys = (
        [geom] if geom.geom_type == "Polygon" else [g for g in getattr(geom, "geoms", []) if g.geom_type == "Polygon"]
    )
    verts, codes = [], []
    for p in polys:
        xy = np.asarray(p.exterior.coords)
        verts.append(xy)
        codes += [Path.MOVETO] + [Path.LINETO] * (len(xy) - 2) + [Path.CLOSEPOLY]
    return Path(np.vstack(verts), codes)


class HandGeometry:
    """Projected 2-D geometry of one pose."""

    def __init__(self, flex: np.ndarray):
        flex = np.clip(np.asarray(flex, dtype=float), 0.0, 1.05)
        self.chains = chains(flex)
        th = self.chains[0]
        (p0, p1), _, (s0, s1) = project(th["pts"][:2])
        self.palm = unary_union([PALM, FOREARM, capsule(p0, p1, th["rad"][0] * s0, th["rad"][1] * s1)])
        self.digits = []  # per digit: segments [(pa, pb, ra, rb, depth, normal)], union, visible outline, depth
        for c_i, ch in enumerate(self.chains):
            s = 1 if c_i == 0 else 0  # the thumb's first segment is the thenar mass, part of the palm
            xy, z, sc = project(ch["pts"])
            segs = [
                (
                    xy[i],
                    xy[i + 1],
                    ch["rad"][i] * sc[i],
                    ch["rad"][i + 1] * sc[i + 1],
                    (z[i] + z[i + 1]) / 2,
                    ch["normals"][i],
                )
                for i in range(s, 3)
            ]
            union = unary_union([capsule(a, b, ra, rb) for a, b, ra, rb, _, _ in segs])
            knuckle = Point(xy[s]).buffer(1.08 * ch["rad"][s] * sc[s], 24).intersection(self.palm)
            outline = (
                union.exterior.difference(knuckle.buffer(0.005)) if union.geom_type == "Polygon" else union.boundary
            )
            depth = float(np.mean(z[s:])) + (0.6 * flex[0] if c_i == 0 else 0.0)  # a flexed thumb lies over the fingers
            self.digits.append({"segs": segs, "union": union, "outline": outline, "depth": depth})

    def silhouette(self):
        return unary_union([self.palm] + [d["union"] for d in self.digits])


class HandArtist:
    """Draws one hand on an axes; ``update(flex, ghost_flex)`` moves it (ghost = outline of a second pose)."""

    def __init__(self, ax, fill, edge, hi, ghost=False, shadow=True):
        self.ax, self.fill, self.edge, self.hi = ax, fill, edge, hi
        self.shadow = (
            ax.add_patch(PathPatch(Path([[0, 0]]), fc="black", ec="none", alpha=0.35, zorder=1)) if shadow else None
        )
        self.palm = ax.add_patch(
            PathPatch(Path([[0, 0]]), fc=tone(fill, edge, hi, 0.65), ec=edge, lw=1.4, zorder=2, joinstyle="round")
        )
        self.palm_hi = ax.add_patch(
            Polygon(
                _flat(_smooth_closed([(-0.35, 0.55), (0.45, 0.7), (0.75, 0.0), (0.45, -0.7), (-0.2, -0.55)])),
                fc=tone(fill, edge, hi, 0.85),
                ec="none",
                alpha=0.35,
                zorder=2.1,
            )
        )
        self.palm_hi.set_clip_path(self.palm)
        for xy in PALM_LINES:
            ln = ax.add_line(
                Line2D(xy[:, 0], xy[:, 1], color=edge, lw=1.0, alpha=0.45, zorder=2.2, solid_capstyle="round")
            )
            ln.set_clip_path(self.palm)
        self.groups = []
        for _ in range(5):
            g = {
                "segs": [ax.add_patch(Polygon(np.zeros((3, 2)), ec="none", zorder=3)) for _ in range(3)],
                "hi": [ax.add_line(Line2D([], [], color=hi, solid_capstyle="round", zorder=3)) for _ in range(3)],
                "crease": ax.add_line(Line2D([], [], color=edge, lw=0.9, solid_capstyle="round", zorder=3)),
                "nail": ax.add_patch(
                    Polygon(np.zeros((3, 2)), fc=tone(fill, edge, hi, 0.92), ec=edge, lw=0.7, zorder=3)
                ),
                "outline": ax.add_line(
                    Line2D([], [], color=edge, lw=1.4, solid_capstyle="round", solid_joinstyle="round", zorder=3)
                ),
            }
            self.groups.append(g)
        self.ghost = (
            ax.add_line(Line2D([], [], color="white", lw=1.3, alpha=0.7, zorder=40, solid_joinstyle="round"))
            if ghost
            else None
        )
        self.pt_per_unit = None

    def _ppu(self) -> float:
        if self.pt_per_unit is None:
            bbox = self.ax.get_window_extent()
            x0, x1 = self.ax.get_xlim()
            self.pt_per_unit = bbox.width / (x1 - x0) * 72 / self.ax.figure.dpi
        return self.pt_per_unit

    def update(self, flex: np.ndarray, ghost_flex: np.ndarray | None = None) -> None:
        ppu, geo = self._ppu(), HandGeometry(flex)
        if self.shadow is not None:
            self.shadow.set_path(_poly_path(affinity.translate(geo.silhouette(), 0.09, -0.11)))
        self.palm.set_path(_poly_path(geo.palm))
        for rank, i in enumerate(np.argsort([d["depth"] for d in geo.digits])):
            self._draw_digit(self.groups[rank], geo.digits[i], 3 + rank, ppu, thumb=(i == 0))
        if self.ghost is not None and ghost_flex is not None:
            g = HandGeometry(ghost_flex)
            self.ghost.set_data(*_lines_xy(unary_union([d["outline"] for d in g.digits])).T)

    def _draw_digit(self, g: dict, digit: dict, z: float, ppu: float, thumb: bool) -> None:
        segs = digit["segs"]
        order = np.argsort([s[4] for s in segs])
        unused = set(range(3))
        creases, nail_xy, nail_alpha = [], np.zeros((3, 2)), 0.0
        for j, si in enumerate(order):
            a, b, ra, rb, _, normal = segs[si]
            b_light, facing = brightness(normal)
            patch, hi = g["segs"][j], g["hi"][j]
            unused.discard(j)
            patch.set_xy(np.asarray(capsule(a, b, ra, rb).exterior.coords))
            patch.set_facecolor(tone(self.fill, self.edge, self.hi, b_light))
            patch.set_visible(True)
            patch.set_zorder(z + 0.01 * j)
            u = b - a
            length = np.linalg.norm(u)
            u = u / length if length > 1e-6 else np.array([0.0, 1.0])
            perp = np.array([-u[1], u[0]])
            if perp @ np.array([-0.6, 0.8]) < 0:
                perp = -perp  # towards the light
            r = (ra + rb) / 2
            off = 0.38 * r * perp
            hi.set_data(
                [a[0] + off[0] + 0.15 * (b - a)[0], b[0] + off[0] - 0.12 * (b - a)[0]],
                [a[1] + off[1] + 0.15 * (b - a)[1], b[1] + off[1] - 0.12 * (b - a)[1]],
            )
            hi.set_linewidth(max(0.5, 0.55 * r * ppu))
            hi.set_alpha(0.12 + 0.3 * b_light)
            hi.set_zorder(z + 0.04)
            hi.set_visible(True)
            if facing > 0.25:  # palm side visible: flexion creases
                k = min(1.0, (facing - 0.25) * 2.5) * 0.55
                spots = [(a, ra)] if si > 0 or thumb else [(a + 0.3 * (b - a), ra)]
                for c, rr in spots:
                    creases.append((c - 0.62 * rr * perp, c + 0.62 * rr * perp, k))
            if si == len(segs) - 1 and facing < -0.1:  # back of the fingertip visible: nail
                nail_alpha = min(1.0, (-facing - 0.1) * 3)
                centre = a + 0.62 * (b - a)
                half_len = max(0.42 * length, 0.55 * rb) / 2
                t = np.linspace(0, 2 * np.pi, 24)
                nail_xy = centre + np.outer(half_len * np.cos(t), u) + np.outer(0.62 * rb * np.sin(t), perp)
        for j in unused:
            g["segs"][j].set_visible(False)
            g["hi"][j].set_visible(False)
        if creases:
            xy = np.vstack([np.r_[[p, q], [[np.nan, np.nan]]] for p, q, _ in creases])
            g["crease"].set_data(*xy.T)
            g["crease"].set_alpha(max(k for _, _, k in creases))
        else:
            g["crease"].set_data([], [])
        g["crease"].set_zorder(z + 0.05)
        g["nail"].set_xy(nail_xy)
        g["nail"].set_alpha(nail_alpha)
        g["nail"].set_visible(nail_alpha > 0)
        g["nail"].set_zorder(z + 0.06)
        g["outline"].set_data(*_lines_xy(digit["outline"]).T)
        g["outline"].set_zorder(z + 0.08)


# ----------------------------------------------------------------------------- layout shared by the videos
SCORED = (0, 1, 2, 4)  # the competition does not score the ring finger
HAND_YLIM = (-1.45, 3.2)


def hand_axes(fig, rect):
    ax = fig.add_axes(rect, facecolor=BG)
    ax.set_ylim(*HAND_YLIM)
    ax.set_xlim(-2.4, 2.4)
    ax.set_aspect("equal", adjustable="datalim")
    ax.axis("off")
    return ax


def progress_bar(fig, total_s: float):
    """Thin bar under the title: where we are in the test segment."""
    ax = fig.add_axes([0.035, 0.868, 0.93, 0.012], facecolor=GRID)
    ax.set_xlim(0, total_s)
    ax.set_ylim(0, 1)
    ax.axis("off")
    bar = ax.barh(0.5, 0, height=1, left=0, color=MUTED, edgecolor="none")[0]
    return bar


def trace_panel(fig, rect, series: list[tuple[str, float]], legend: list, win_steps: int):
    """Five stacked rows of scrolling traces; returns the axis, the time axis and lines[series][finger]."""
    ax = fig.add_axes(rect, facecolor=PANEL)
    tt = np.arange(-win_steps + 1, 1) * STEP_S
    lines = []
    for k in range(5):
        off = (4 - k) * 1.3
        ax.axhline(off, color=GRID, lw=0.6, zorder=0)
        label = FINGERS[k] if k in SCORED else f"{FINGERS[k]}\n(not scored)"
        ax.text(
            tt[0] - 0.12,
            off + 0.45,
            label,
            ha="right",
            va="center",
            fontsize=9.5 if k in SCORED else 8,
            color=TEXT if k in SCORED else MUTED,
            linespacing=1.0,
        )
    for color, lw in series:
        lines.append([ax.plot([], [], color=color, lw=lw, solid_joinstyle="round")[0] for _ in range(5)])
    ax.axvline(0, color=TEXT, lw=1, alpha=0.6)
    ax.set_xlim(tt[0], 0.05)
    ax.set_ylim(-0.35, 4 * 1.3 + 1.4)
    ax.set_yticks([])
    ax.tick_params(labelsize=8.5, length=0)
    ax.set_xlabel("seconds before now", fontsize=9)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.legend(
        handles=legend,
        loc="lower right",
        bbox_to_anchor=(1.0, 1.0),
        ncol=len(legend),
        frameon=False,
        fontsize=9,
        labelcolor=TEXT,
        handlelength=1.6,
        columnspacing=1.4,
    )
    return ax, tt, lines


def set_traces(lines, tt, data: list[np.ndarray], t: int) -> None:
    lo = t - len(tt) + 1
    for series, x in zip(lines, data):
        seg = np.full((len(tt), 5), np.nan)
        avail = x[max(lo, 0) : t + 1]
        seg[len(tt) - len(avail) :] = avail
        for k in range(5):
            series[k].set_data(tt, np.clip(seg[:, k], -0.25, 1.3) + (4 - k) * 1.3)
