"""
Shared geometry core for skeleton (keypoint + bone) visualizations.

Used by both the video-overlay skeleton (``audiovideo.skeleton_plot.PlotPoints``)
and the TsdFrame-canvas skeleton display mode so the point/line buffer logic
isn't duplicated between the two.
"""

from __future__ import annotations

import itertools
import warnings
from typing import Optional, Sequence, Union

import numpy as np
import pygfx as gfx


def resolve_keypoint_labels(columns: Sequence) -> list:
    """Group TsdFrame columns into per-keypoint labels.

    Consecutive column pairs named ``"<name>_x"``/``"<name>_y"`` (as produced by
    ``nap.from_movement``) are collapsed to ``"<name>"``. Any other column naming
    (e.g. the default integer columns) falls back to positional labels
    ``"0", "1", ...`` — one per (x, y) pair, in column order.

    Parameters
    ----------
    columns : sequence
        The TsdFrame's ``.columns``, of even length (x, y interleaved per keypoint).

    Returns
    -------
    list of str
        One label per keypoint, length ``len(columns) // 2``.
    """
    if len(columns) % 2 != 0:
        raise ValueError("columns must have an even length (x, y pairs).")

    n_points = len(columns) // 2
    labels = []
    for i in range(n_points):
        cx, cy = str(columns[2 * i]), str(columns[2 * i + 1])
        if cx.endswith("_x") and cy.endswith("_y") and cx[:-2] == cy[:-2]:
            labels.append(cx[:-2])
        else:
            labels.append(str(i))
    return labels


def resolve_edges(
    labels: Sequence[str],
    edges: Optional[Sequence[tuple]] = None,
) -> list:
    """Resolve a bone/edge list to index pairs into the keypoint array.

    Parameters
    ----------
    labels : sequence of str
        Per-keypoint labels, as returned by :func:`resolve_keypoint_labels`.
    edges : sequence of (str or int, str or int), optional
        Bone connectivity as pairs of keypoint labels or indices. When ``None``,
        falls back to the complete graph (every keypoint connected to every
        other), matching the original ``PlotPoints`` behavior.

    Returns
    -------
    list of (int, int)
    """
    n_points = len(labels)
    if edges is None:
        if n_points > 2:
            warnings.warn(
                "No `edges` provided: connecting every keypoint pair (complete "
                "graph). Pass an explicit `edges` list of keypoint-name pairs "
                "for an anatomically correct skeleton.",
                stacklevel=2,
            )
        return list(itertools.combinations(range(n_points), 2))

    resolved = []
    for a, b in edges:
        resolved.append((_resolve_index(a, labels), _resolve_index(b, labels)))
    return resolved


def _resolve_index(value: Union[str, int], labels: Sequence[str]) -> int:
    if isinstance(value, (int, np.integer)):
        return int(value)
    try:
        return labels.index(value)
    except ValueError:
        raise ValueError(
            f"Unknown keypoint label {value!r}. Available labels: {list(labels)}"
        ) from None


def edges_from_parent_metadata(data) -> Optional[list]:
    """Derive a (child, parent) skeleton edge list from a TsdFrame's "parent" metadata.

    Each keypoint's `_x`/`_y` column pair is expected to carry the same
    ``"parent"`` metadata value (the keypoint label it's connected to, or
    ``None``/``NaN`` for the root), as written by e.g.
    ``movement_example.set_skeleton_metadata``.

    Parameters
    ----------
    data : nap.TsdFrame

    Returns
    -------
    list of (str, str), or None
        ``None`` when ``data`` has no ``"parent"`` metadata field, so callers
        can fall back to their own default (e.g. the complete graph).
    """
    metadata = getattr(data, "metadata", None)
    if metadata is None or "parent" not in metadata.columns:
        return None

    labels = resolve_keypoint_labels(data.columns)
    edges = []
    for i, label in enumerate(labels):
        parent = metadata["parent"].iloc[2 * i]
        if parent is None or (isinstance(parent, float) and np.isnan(parent)):
            continue
        edges.append((label, parent))
    return edges


class SkeletonGeometry:
    """Pygfx points + connecting lines for a set of 2D keypoints.

    Geometry-only: knows nothing about time series, controllers, or scenes
    beyond adding/removing itself. Shared by the video-overlay skeleton and
    the TsdFrame-canvas skeleton display mode.

    Parameters
    ----------
    xy : np.ndarray
        Initial keypoint positions, shape ``(n_points, 2)``.
    edges : list of (int, int)
        Index pairs (into ``xy``'s rows) defining connecting lines.
    scene : gfx.Scene, optional
        If given, the points/lines are added to it immediately.
    color : str or tuple, default "red"
        Color of the points.
    markersize : float, default 8.0
        Size of the points in pixels.
    thickness : float, default 0.02
        Thickness of the connecting lines, in screen pixels.
    """

    def __init__(
        self,
        xy: np.ndarray,
        edges: Sequence[tuple],
        scene: Optional[gfx.Scene] = None,
        color="red",
        markersize: float = 8.0,
        thickness: float = 0.02,
    ):
        xy = np.asarray(xy, dtype="float32")
        self.n_points = xy.shape[0]
        edges = list(edges)
        self.edges_idx = np.array(edges, dtype=int).flatten() if edges else np.array([], dtype=int)

        positions = np.hstack(
            (xy, np.ones((self.n_points, 1), dtype="float32"))
        ).astype("float32")

        geom_points = gfx.Geometry(positions=positions)
        mat_points = gfx.PointsMaterial(color=color, size=markersize)
        self.points = gfx.Points(geom_points, mat_points)

        self.lines = None
        if len(edges):
            line_positions = positions[self.edges_idx].astype("float32")
            geom_lines = gfx.Geometry(positions=line_positions)
            # LineSegmentMaterial draws each consecutive *pair* of points as an
            # independent segment. A plain gfx.Line would instead draw one
            # continuous polyline through all bones back-to-back, spuriously
            # connecting the end of one bone to the start of the next and
            # producing visible loops across unrelated edges.
            mat_lines = gfx.LineSegmentMaterial(thickness=thickness, color="grey")
            self.lines = gfx.Line(geom_lines, mat_lines)

        self.scene = scene
        if scene is not None:
            scene.add(self.points)
            if self.lines is not None:
                scene.add(self.lines)

    def update(self, xy: np.ndarray) -> None:
        """Update keypoint positions.

        Parameters
        ----------
        xy : np.ndarray
            New keypoint positions, shape ``(n_points, 2)``.
        """
        xy = np.asarray(xy, dtype="float32")
        positions = np.hstack(
            (xy, np.ones((self.n_points, 1), dtype="float32"))
        ).astype("float32")
        self.points.geometry.positions.set_data(positions)
        if self.lines is not None:
            self.lines.geometry.positions.set_data(positions[self.edges_idx])

    def set_color(self, color) -> None:
        """Set color of the points."""
        self.points.material.color = color

    def set_markersize(self, markersize: float) -> None:
        """Set size of the points."""
        self.points.material.size = markersize

    def set_thickness(self, thickness: float) -> None:
        """Set thickness of the connecting lines."""
        if self.lines is None:
            return
        if thickness <= 1e-3:
            self.lines.material.opacity = 0
        else:
            self.lines.material.opacity = 1
            self.lines.material.thickness = thickness

    def remove_from_scene(self, scene: Optional[gfx.Scene] = None) -> None:
        """Remove the points/lines from a scene (defaults to the scene passed at init)."""
        scene = scene if scene is not None else self.scene
        if scene is None:
            return
        scene.remove(self.points)
        if self.lines is not None:
            scene.remove(self.lines)

    def get_state(self) -> dict:
        state = {
            "color": self.points.material.color.rgba,
            "markersize": self.points.material.size,
        }
        if self.lines is not None:
            state["thickness"] = self.lines.material.thickness
        return state

    @classmethod
    def from_state(cls, state: dict, xy: np.ndarray, edges: Sequence[tuple], scene=None):
        return cls(xy, edges, scene=scene, **state)