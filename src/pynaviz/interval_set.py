"""Handling of interval sets associated to base_plot classes."""

import re
import warnings
from typing import Iterable, Optional

import numpy as np
import pygfx
import pynapple as nap

from .utils import GRADED_COLOR_LIST, get_plot_min_max

INTERVAL_PATTERN = re.compile(r"^interval_\d+$")


def get_max_interval_index(labels):
    return max(
        (
            -1,
            *(
                int(lab.split("_")[1])
                for lab in labels
                if re.match(INTERVAL_PATTERN, lab)
            ),
        )
    )


class IntervalSetInterface:
    def __init__(
        self,
        epochs: Optional[Iterable[nap.IntervalSet] | nap.IntervalSet] = None,
        labels: Optional[Iterable[str] | str] = None,
    ):
        self._epochs = dict()
        if epochs is not None:
            self.add_interval_sets(epochs, labels)

        # map label -> single batched gfx.Mesh for all intervals
        self._interval_rects = dict()

        # store colors, alphas for each inteval set
        self._interval_state = {}

    def add_interval_sets(
        self,
        epochs: Iterable[nap.IntervalSet] | nap.IntervalSet,
        colors: Optional[Iterable | str | pygfx.Color] = None,
        alpha: Optional[Iterable[float] | float] = None,
        labels: Optional[Iterable[str] | str] = None,
    ):
        if isinstance(epochs, nap.IntervalSet):
            epochs = [epochs]
        else:
            epochs = list(epochs)
        indx_start = get_max_interval_index(self._epochs.keys()) + 1
        labels = (
            labels
            if labels is not None
            else [f"interval_{i + indx_start}" for i, _ in enumerate(epochs)]
        )
        labels = [labels] if isinstance(labels, str) else list(labels)
        if len(labels) != len(epochs):
            raise ValueError(
                "The number of labels provided does not match the number of epochs."
            )
        new_intervals = dict(zip(labels, epochs))
        self._epochs.update(new_intervals)
        self._plot_intervals(labels, colors, alpha)
        # append the control action if available
        if all(f != self._update_all_isets for f in self.controller._plot_callbacks):
            self.controller._plot_callbacks.append(self._update_all_isets)

    def update_interval_set(self, name: str, colors: str | pygfx.Color, alpha: float) -> None:
        """
        Update the color and transparency of an existing interval set.

        Parameters
        ----------
        name:
            The label of the interval set to be updated.
        colors:
            The new color of the interval set.
        alpha:
            The new transparency level, between 0 and 1.
        """
        if name not in self._epochs:
            warnings.warn(
                message=f"Epochs {name} is not available. Available epochs: {list(self._epochs.keys())}.",
                category=UserWarning,
                stacklevel=2,
            )
            return
        if name not in self._interval_rects:
            warnings.warn(
                message=f"Interval set {name} has not been plotted yet. Use `add_interval_sets` instead.",
                category=UserWarning,
                stacklevel=2,
            )
            return
        if isinstance(colors, str):
            colors = pygfx.Color(colors)
        self._update_rectangles(name, colors, alpha)

    def remove_interval_set(self, label: str) -> None:
        """
        Remove an interval set from the plot.

        Parameters
        ----------
        label:
            The label of the interval set to be removed.
        """
        if label not in self._epochs:
            warnings.warn(
                message=f"Epochs {label} is not available. Available epochs: {list(self._epochs.keys())}.",
                category=UserWarning,
                stacklevel=2,
            )
            return
        if label in self._interval_rects:
            self.scene.remove(self._interval_rects[label])
            del self._interval_rects[label]
            self.canvas.request_draw(self.animate)
        del self._epochs[label]
        # remove the control action if no interval set is left
        if len(self._interval_rects) == 0 and any(
            f == self._update_all_isets for f in self.controller._plot_callbacks
        ):
            self.controller._plot_callbacks.remove(self._update_all_isets)

        # remove from state
        del self._interval_state[label]

    def _plot_intervals(
        self,
        labels: Iterable[str] | str,
        colors: Optional[Iterable] = None,
        alpha: Optional[Iterable[float] | float] = 1.0,
    ) -> None:
        """
        Plot rectangle over label areas.

        This method plot the rectangle. If an interval is already plotted, then
        the method updates the coloring properties only.

        Parameters
        ----------
        labels:
            The label of the epochs to be plotted.
        colors:
            Optional, list of colors, format must be compatible with pygfx.Color.
        alpha:
            The transparency level, between 0 and 1.

        """
        if isinstance(labels, str):
            labels = [labels]

        if alpha is None:
            alpha = [0.4] * len(labels)
        elif isinstance(alpha, float):
            alpha = [alpha] * len(labels)

        if isinstance(colors, str):
            colors = [colors] * len(labels)
        elif colors is None:
            colors = [None] * len(labels)
        else:
            # unpack the color
            if len(labels) != 1:
                # this is a internal design issue, should raise
                raise ValueError("When colors are provided as RGBs (only during layout loading), "
                                 "each IntervalSet is processed one at the time. "
                                 "This call is generating multiple interval sets rectangles with a single color.")
            colors = [pygfx.Color(*colors)]

        color_idx = len(self._interval_rects) + 1
        for label, color, transparency in zip(labels, colors, alpha):
            if label not in self._epochs:
                warnings.warn(
                    message=f"Epochs {label} is not available. Available epochs: {list(self._epochs.keys())}.",
                    category=UserWarning,
                    stacklevel=2,
                )
                continue
            is_new = label not in self._interval_rects
            if is_new:
                col = (
                    pygfx.Color(GRADED_COLOR_LIST[color_idx % len(GRADED_COLOR_LIST)])
                    if color is None
                    else color
                )
                mesh = self._create_and_plot_rectangle(
                    label, col, transparency
                )
                self._interval_rects[label] = mesh
                color_idx += 1
            else:
                self._update_rectangles(label, color, transparency)

    def _update_all_isets(self, *args, **kwargs):
        """Update all interval sets rectangles."""
        for label in self._interval_rects:
            self._update_rectangles(label)

    def _build_batch_geometry(self, epoch, ymin, ymax, depth):
        """Build batched positions and indices for all intervals in epoch.

        Each interval contributes 4 vertices (BL, BR, TR, TL) and 2 triangles.
        Returns positions (n*4, 3) float32 and indices (n*2, 3) uint32.
        """
        n = len(epoch)
        starts = np.asarray(epoch.start, dtype="float32")
        ends = np.asarray(epoch.end, dtype="float32")

        positions = np.zeros((n * 4, 3), dtype="float32")
        # x: BL=start, BR=end, TR=end, TL=start
        positions[0::4, 0] = starts
        positions[1::4, 0] = ends
        positions[2::4, 0] = ends
        positions[3::4, 0] = starts
        # y: BL=ymin, BR=ymin, TR=ymax, TL=ymax
        positions[0::4, 1] = ymin
        positions[1::4, 1] = ymin
        positions[2::4, 1] = ymax
        positions[3::4, 1] = ymax
        positions[:, 2] = depth

        base = np.arange(n, dtype="uint32") * 4
        indices = np.empty((n * 2, 3), dtype="uint32")
        indices[0::2, 0] = base       # triangle 0: BL
        indices[0::2, 1] = base + 1   # triangle 0: BR
        indices[0::2, 2] = base + 2   # triangle 0: TR
        indices[1::2, 0] = base       # triangle 1: BL
        indices[1::2, 1] = base + 2   # triangle 1: TR
        indices[1::2, 2] = base + 3   # triangle 1: TL

        return positions, indices

    def _create_and_plot_rectangle(self, label, color, transparency):
        """Create a single batched mesh for all intervals in epoch."""
        epoch = self._epochs[label]
        _, _, ymin, ymax = get_plot_min_max(self)
        color = pygfx.Color(*pygfx.Color(color).rgb, transparency)
        self._interval_state[label] = {
            "colors": list(color.rgb),
            "alpha": float(color.a)
        }

        ruler = getattr(self, "ruler_x", None)
        depth = (ruler.start_pos[-1] - 1) if ruler is not None else -1001.0

        n = len(epoch)
        if n > 0:
            positions, indices = self._build_batch_geometry(epoch, ymin, ymax, depth)
            geom = pygfx.Geometry(positions=positions, indices=indices)
        else:
            geom = pygfx.Geometry(
                positions=np.zeros((3, 3), dtype="float32"),
                indices=np.zeros((1, 3), dtype="uint32"),
            )

        material = pygfx.MeshBasicMaterial(color=color, pick_write=True)
        mesh = pygfx.Mesh(geom, material)

        self.scene.add(mesh)
        self.canvas.request_draw(self.animate)
        return mesh

    def _update_rectangles(self, label, color=None, transparency=None):
        """Update color and/or y-extent of the batched mesh for label."""
        mesh = self._interval_rects[label]
        epoch = self._epochs[label]
        current_color = mesh.material.color
        if color is None:
            color = current_color
        transparency = transparency if transparency is not None else float(current_color.a)
        self._interval_state[label] = {"colors": list(pygfx.Color(color).rgb), "alpha": float(np.float32(transparency))}

        new_color = pygfx.Color(
            *self._interval_state[label]["colors"],
            self._interval_state[label]["alpha"],
        )
        if mesh.material.color != new_color:
            mesh.material.color = new_color

        if len(epoch) == 0:
            return

        _, _, ymin, ymax = get_plot_min_max(self)
        positions = np.asarray(mesh.geometry.positions.data)
        if len(positions) < 4:
            return

        old_ymin = float(positions[0, 1])
        old_ymax = float(positions[2, 1])
        if old_ymin == ymin and old_ymax == ymax:
            return

        positions = positions.copy()
        positions[0::4, 1] = ymin
        positions[1::4, 1] = ymin
        positions[2::4, 1] = ymax
        positions[3::4, 1] = ymax
        mesh.geometry.positions.set_data(positions)

    def _interval_set_from_state(self, state: dict, available_isets: dict):
        """Restore IntervalSet overlay."""
        for label, props in state.items():
            if label in available_isets:
                self.add_interval_sets(
                    available_isets[label],
                    **props,
                    labels=label,
                )
            else:
                print(f"IntervalSet ``{label}`` not found.")