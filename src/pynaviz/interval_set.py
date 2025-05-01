"""Handling of interval sets associated to base_plot classes."""
from locale import currency
from typing import Iterable, Optional
import pynapple as nap
import pygfx
import warnings
from .utils import get_plot_min_max, GRADED_COLOR_LIST
import numpy as np

class IntervalSetInterface:
    def __init__(
            self,
            epochs: Optional[Iterable[nap.IntervalSet] | nap.IntervalSet]=None,
            labels: Optional[Iterable[str] | str]=None
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
            labels: Optional[Iterable[str] | str]=None
    ):
        epochs = list(epochs)
        labels = labels if labels is not None else [f"interval_{i}" for i, _ in enumerate(epochs)]
        labels = [labels] if isinstance(labels, str) else list(labels)
        if len(labels) != len(epochs):
            raise ValueError("The number of labels provided does not match the number of epochs.")
        new_intervals = dict(zip(labels, epochs))
        self._epochs.update(new_intervals)
        self._plot_intervals(labels, colors, alpha)

    def _plot_intervals(
            self,
            labels: Iterable[str] | str,
            colors: Optional[Iterable]=None,
            alpha: Optional[Iterable[float] | float]=1.) -> None:
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
                warnings.warn(message=f"Epochs {label} is not available. Available epochs: {list(self._epochs.keys())}.", category=UserWarning)
                continue
            is_new = label not in self._interval_rects
            if is_new:
                col = pygfx.Color(GRADED_COLOR_LIST[color_idx % len(GRADED_COLOR_LIST)]) if color is None else color
                meshes = self._create_and_plot_rectangle(self._epochs[label], col, transparency)
                self._interval_rects[label] = meshes
                color_idx += 1
            else:
                self._update_rectangles(self._interval_rects[label], color, transparency)

    def _update_all_isets(self):
        for meshes in self._interval_rects.values():
            self._update_rectangles(meshes)

    def _update_rectangles(self, rectangles, color=None, transparency=None):
        # set to current values if not provided
        color = color if color is not None else next(iter(rectangles.values())).material.color
        transparency = transparency if transparency is not None else color.a

        _, _, ymin, ymax = get_plot_min_max(self)
        new_height = ymax - ymin
        for rect in rectangles.values():
            # compute new height
            width, old_height = np.ptp(np.asarray(rect.geometry.positions.data)[:, :2], axis=0)
            if old_height != new_height:
                geom = pygfx.plane_geometry(width, new_height)
                rect.geometry = geom
                position = rect.local.position
                rect.local.position = np.array(
                    [position[0], ymin + new_height / 2, position[-1]],
                    dtype=np.float32
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
            depth = -1001.

        for ep in epoch:
            width = ep.end[0] - ep.start[0]
            geom = pygfx.plane_geometry(width, height)
            material = pygfx.MeshBasicMaterial(
                color=color, pick_write=True
            )
            mesh = pygfx.Mesh(geom, material)
            mesh.local.position = np.array(
                [ep.start[0] + width / 2, ymin + height / 2, depth],
                dtype=np.float32
            )
            mesh_dict[ep.start[0], ep.end[0]] = mesh

        self.scene.add(*mesh_dict.values())
        self.canvas.request_draw(self.animate)
        self.controller.plot_updates.append(self._update_all_isets)
        return mesh_dict


