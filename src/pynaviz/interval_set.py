"""Handling of interval sets associated to base_plot classes."""

import re
import warnings
from typing import Iterable, Optional

import numpy as np
import pygfx
import pynapple as nap

from .utils import GRADED_COLOR_LIST, get_plot_min_max

INTERVAL_PATTERN = re.compile(r"^interval_\d+$")


def is_in_view(screen_xmin, screen_xmax, width, rectangle):
    """Compute if a rectangle is in the field of view.

    Parameters
    ----------
    screen_xmin :
        Lower xlim of the canvas.
    screen_xmax :
        Upper xlim of the canvas.
    width:
        Rectangle width.
    rectangle :
        A rectangle as a pygfx.Mesh.
    """
    rect_center_x = rectangle.local.position[0]
    rect_min, rect_max = rect_center_x - width / 2, rect_center_x + width / 2
    return (screen_xmin < rect_max) and (screen_xmax > rect_min)


def get_max_interval_index(labels):
    return max((-1, *(int(lab.split("_")[1]) for lab in labels if re.match(INTERVAL_PATTERN, lab))))


class IntervalSetInterface:
    def __init__(
        self,
        epochs: Optional[Iterable[nap.IntervalSet] | nap.IntervalSet] = None,
        labels: Optional[Iterable[str] | str] = None,
    ):
        self._epochs = dict()
        if epochs is not None:
            self.add_interval_sets(epochs, labels)

        # map to the rectangle meshes
        self._interval_rects = dict()

    def add_interval_sets(
        self,
        epochs: Iterable[nap.IntervalSet] | nap.IntervalSet,
        colors: Optional[Iterable | str | pygfx.Color] = None,
        alpha: Optional[Iterable[float] | float] = None,
        labels: Optional[Iterable[str] | str] = None,
    ):
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
        if all(f != self._update_all_isets for f in self.controller.plot_updates):
            self.controller.plot_updates.append(self._update_all_isets)

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

        if alpha is None or isinstance(alpha, float):
            alpha = [0.4] * len(labels)

        if isinstance(colors, str):
            colors = [colors] * len(labels)

        if colors is None:
            colors = [None] * len(labels)

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
                meshes = self._create_and_plot_rectangle(
                    self._epochs[label], col, transparency
                )
                self._interval_rects[label] = meshes
                color_idx += 1
            else:
                self._update_rectangles(
                    self._interval_rects[label], color, transparency
                )

    def _update_all_isets(self):
        for rectangles in self._interval_rects.values():
            self._update_rectangles(rectangles)

    def _update_rectangles(self, rectangles, color=None, transparency=None):
        # set to current values if not provided
        color = (
            color
            if color is not None
            else next(iter(rectangles.values())).material.color
        )
        transparency = transparency if transparency is not None else color.a

        xmin, xmax, ymin, ymax = get_plot_min_max(self)
        new_height = ymax - ymin
        for rect in rectangles.values():
            # compute new height
            position = np.asarray(rect.geometry.positions.data)
            width, old_height = np.ptp(position[:, :2], axis=0)
            if (old_height != new_height) and is_in_view(xmin, xmax, width, rect):
                geom = pygfx.plane_geometry(width, new_height)
                rect.geometry = geom
                position = rect.local.position
                rect.local.position = np.array(
                    [position[0], ymin + new_height / 2, position[-1]], dtype=np.float32
                )

            # update color & transparency
            new_color = pygfx.Color(*pygfx.Color(color).rgb, transparency)
            if rect.material.color != new_color:
                rect.material.color = new_color

    def _create_and_plot_rectangle(self, epoch, color, transparency):
        _, _, ymin, ymax = get_plot_min_max(self)
        color = pygfx.Color(*pygfx.Color(color).rgb, transparency)
        height = ymax - ymin
        mesh_dict = dict()
        ruler = getattr(self, "rulerx", None)
        if ruler is not None:
            # plot rect behind ruler.
            depth = ruler.start_pos[-1] - 1
        else:
            # hardcode a background level.
            depth = -1001.0

        for ep in epoch:
            width = ep.end[0] - ep.start[0]
            geom = pygfx.plane_geometry(width, height)
            material = pygfx.MeshBasicMaterial(color=color, pick_write=True)
            mesh = pygfx.Mesh(geom, material)
            mesh.local.position = np.array(
                [ep.start[0] + width / 2, ymin + height / 2, depth], dtype=np.float32
            )
            mesh_dict[ep.start[0], ep.end[0]] = mesh

        self.scene.add(*mesh_dict.values())
        self.canvas.request_draw(self.animate)
        return mesh_dict
