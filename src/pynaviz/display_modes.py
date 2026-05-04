"""
Display mode strategies for PlotTsdFrame.

Each mode encapsulates how GPU buffers are populated, how rescaling works,
how coloring is applied, and how the mode activates/deactivates in the scene.
"""

from __future__ import annotations

import numpy as np
import pygfx as gfx
from matplotlib.pyplot import colormaps

from .threads.data_streaming import TsdFrameStreaming
from .utils import trim_kwargs


class LinesMode:
    """Standard line-plot display mode with vertex colors.

    Each channel is stored as a contiguous segment in a shared positions buffer,
    separated by NaN gaps. Colors are per-vertex (RGBA).

    Parameters
    ----------
    data : nap.TsdFrame
        The time series data to display.
    manager : PlotTsdFrameManager
        Manages per-channel metadata (offset, scale, visibility, etc.).
    window_size : float, optional
        Streaming window duration in seconds. If None, computed to fit
        up to 256 MB of memory.
    """

    controller_key = "span"

    def __init__(self, data, manager, window_size=None, default_color="white"):
        self.data = data
        self.window_size = window_size
        self.manager = manager
        self._default_color = default_color
        if window_size is None:
            size = (256 * 1024**2) // (data.shape[1] * 60)
            window_size = np.floor(size / data.rate)
            if window_size < 1:
                window_size = 1.0

        self.stream = TsdFrameStreaming(
            data, callback=self.flush, window_size=window_size
        )

        # Shared positions buffer: (max_n+1)*n_channels rows, NaN-separated per channel
        self.buffer = np.full(
            ((self.stream._max_n + 1) * self.data.shape[1], 3), np.nan, dtype="float32"
        )
        self._buffer_slices = {}
        for c, s in zip(
            self.data.columns,
            range(
                0,
                len(self.buffer) - self.stream._max_n + 1,
                self.stream._max_n + 1,
            ),
        ):
            self._buffer_slices[c] = slice(s, s + self.stream._max_n)

        self.buffer[:, 2] = 0.0

    def initialize_graphic(self):
        """Create the gfx.Line graphic (no-op if already created)."""
        if hasattr(self, "graphic"):
            return
        c = gfx.Color(self._default_color)
        colors = np.tile(
            np.array([c.r, c.g, c.b, c.a], dtype=np.float32),
            (self.buffer.shape[0], 1),
        )
        self.graphic = gfx.Line(
            gfx.Geometry(positions=self.buffer, colors=colors),
            gfx.LineMaterial(thickness=1.0, color_mode="vertex"),
        )

    def get_callbacks(self):
        """Return streaming callbacks for the controller, if needed."""
        if self.stream._max_n < self.data.shape[0]:
            return [self.stream.stream]
        return []

    def flush(self, slice_=None):
        """Populate the positions buffer from data and push to the GPU.

        Parameters
        ----------
        slice_ : slice, optional
            Data slice for streaming mode. Ignored when all data fits in memory.
        """
        if self.stream._max_n == self.data.shape[0]:
            for i, c in enumerate(self.data.columns):
                if not self.manager.data.loc[c]["visible"]:
                    continue
                sl = self._buffer_slices[c]
                self.buffer[sl, 0] = self.data.t.astype("float32")
                self.buffer[sl, 1] = self.data.d[:, i].astype("float32")
                self.buffer[sl, 1] *= self.manager.data.loc[c]["scale"]
                self.buffer[sl, 1] += self.manager.data.loc[c]["offset"]
        else:
            time = self.data.t[slice_].astype("float32")

            left_offset = 0
            right_offset = 0
            if time.shape[0] < self.stream._max_n:
                if slice_.start == 0:
                    left_offset = self.stream._max_n - time.shape[0]
                else:
                    right_offset = time.shape[0] - self.stream._max_n

            # Load the current slice of data into the buffer, applying scale and offset.
            data = np.array(self.data.values[slice_, :])

            for i, c in enumerate(self.data.columns):
                if not self.manager.data.loc[c]["visible"]:
                    continue
                sl = self._buffer_slices[c]
                sl = slice(sl.start + left_offset, sl.stop + right_offset)
                self.buffer[sl, 0] = time
                self.buffer[sl, 1] = data[:, i]
                self.buffer[sl, 1] *= self.manager.data.loc[c]["scale"]
                self.buffer[sl, 1] += self.manager.data.loc[c]["offset"]

            if left_offset:
                for sl in self._buffer_slices.values():
                    self.buffer[sl.start : sl.start + left_offset, 0:2] = np.nan
            if right_offset:
                for sl in self._buffer_slices.values():
                    self.buffer[sl.stop + right_offset : sl.stop, 0:2] = np.nan

        self.graphic.geometry.positions.set_data(self.buffer)

    def get_buffer_min_max(self):
        """Return per-channel [min, max] from the current buffer.

        Returns
        -------
        np.ndarray
            Shape (n_channels, 2).
        """
        return np.array(
            [
                [np.nanmin(self.buffer[sl, 1]), np.nanmax(self.buffer[sl, 1])]
                for sl in self._buffer_slices.values()
            ]
        )

    def rescale(self, key):
        """Scale channel amplitudes with 'i' (increase) or 'd' (decrease).

        Returns False if channels are not sorted/grouped.
        """
        if not (self.manager.is_sorted or self.manager.is_grouped):
            return False
        factor = {"i": 0.5, "d": -0.5}[key]
        self.manager.rescale(factor=factor)
        for c, sl in self._buffer_slices.items():
            self.buffer[sl, 1] += factor * (
                self.buffer[sl, 1] - self.manager.data.loc[c]["offset"]
            )
        self.graphic.geometry.positions.set_data(self.buffer)
        return True

    def color_by(self, cmap_name, metadata_name, vmin, vmax, map_to_colors, values):
        """Apply per-channel vertex colors from a metadata field."""
        map_kwargs = trim_kwargs(
            map_to_colors, dict(cmap=colormaps[cmap_name], vmin=vmin, vmax=vmax)
        )
        if len(values):
            map_color = map_to_colors(values, **map_kwargs)
            if map_color:
                for c, sl in self._buffer_slices.items():
                    self.graphic.geometry.colors.data[sl, :] = map_color[values[c]]
                self.graphic.geometry.colors.update_full()

        self.manager.color_by(values, metadata_name, cmap_name=cmap_name, vmin=vmin, vmax=vmax)

    def update_visibility(self):
        """Hide channels by NaN-ing their buffer positions; show them by re-flushing."""
        for c, sl in self._buffer_slices.items():
            if not self.manager.data.loc[c]["visible"]:
                self.buffer[sl, 0:2] = np.nan
        self.flush(self.stream._slice_)

    def get_state(self):
        """Return per-column scale factors and visibility as a JSON-serializable dict."""
        return {
            "scale": self.manager.scale.tolist(),
            "visible": self.manager.visible.tolist(),
        }

    def set_state(self, state):
        """Restore per-column scale factors and visibility.

        Parameters
        ----------
        state : dict
            Value produced by :meth:`get_state`.
        """
        self.manager.scale = state["scale"]
        self.manager.visible = state["visible"]
        self.update_visibility()


class ImageMode:
    """Heatmap display mode with a 2D texture and locked y-axis.

    The image buffer has shape (n_channels, max_n). When some channels are
    hidden, the texture is recreated with only the visible rows so the image
    shrinks vertically.

    Parameters
    ----------
    data : nap.TsdFrame
        The time series data to display.
    manager : PlotTsdFrameManager
        Manages per-channel metadata (offset, scale, visibility, etc.).
    max_n : int, default 16384
        Maximum number of time points in the image buffer (GPU texture width).
    """

    controller_key = "span_image"

    def __init__(self, data, manager, max_n=16384):
        self.data = data
        self.manager = manager
        self.window_size = np.maximum(max_n / data.rate, 1.0)

        self.stream = TsdFrameStreaming(
            data, callback=self.flush, window_size=self.window_size
        )
        # Clamp to max_n: floating-point rounding in _get_slice can yield
        # _max_n = max_n + 1, which would exceed the GPU texture size limit.
        self.stream._max_n = min(self.stream._max_n, max_n)

        self.buffer = np.zeros((data.shape[1], self.stream._max_n), dtype="float32")
        self._n_visible = data.shape[1]

    def initialize_graphic(self):
        """Create the gfx.Image graphic and texture (no-op if already created)."""
        if all(hasattr(self, attr) for attr in ["graphic", "texture"]):
            return
        self.texture = gfx.Texture(self.buffer, dim=2)
        self.graphic = gfx.Image(
            gfx.Geometry(grid=self.texture),
            gfx.ImageBasicMaterial(clim=(0, 1), map=gfx.cm.viridis),
        )
        self.graphic.local.z = -10.0

    def get_callbacks(self):
        """Return streaming callbacks for the controller, if needed."""
        if self.stream._max_n < self.data.shape[0]:
            return [self.stream.stream]
        return []

    def flush(self, slice_=None):
        """Fill the image buffer from data and update the texture.

        Parameters
        ----------
        slice_ : slice, optional
            Data slice for streaming mode. Ignored when all data fits in memory.
        """
        if self.stream._max_n == self.data.shape[0]:
            raw = self.data.d[:, :].astype("float32").T
            raw = self._reorder_image_rows(raw)
            self.buffer[:, :] = 0.0
            if raw.shape[1] > self.stream._max_n:
                idx = np.linspace(0, raw.shape[1] - 1, self.stream._max_n, dtype=int)
                self.buffer[:, :] = raw[:, idx]
                n_filled = self.stream._max_n
            else:
                self.buffer[:, :raw.shape[1]] = raw
                n_filled = raw.shape[1]
            t_start = float(self.data.t[0])
            t_end = float(self.data.t[-1])
        else:
            data = np.array(self.data.values[slice_, :])
            raw_img = data.T.astype("float32")
            raw_img = self._reorder_image_rows(raw_img)
            self.buffer[:, :] = 0.0
            if raw_img.shape[1] > self.stream._max_n:
                idx = np.linspace(0, raw_img.shape[1] - 1, self.stream._max_n, dtype=int)
                self.buffer[:, :] = raw_img[:, idx]
                n_filled = self.stream._max_n
            else:
                self.buffer[:, :raw_img.shape[1]] = raw_img
                n_filled = raw_img.shape[1]
            t_start = float(self.data.t[slice_.start])
            t_end = float(self.data.t[min(slice_.stop, self.data.shape[0]) - 1])

        # Extract only visible rows for the texture
        visible_buf = self._get_visible_buffer()
        n_visible = visible_buf.shape[0]

        if n_visible != self._n_visible:
            self._n_visible = n_visible
            self.texture = gfx.Texture(visible_buf.copy(), dim=2)
            self.graphic.geometry.grid = self.texture
        else:
            self.texture.data[:] = visible_buf
            self.texture.update_full()

        # Align image pixels to actual time positions on the x-axis.
        # Uses n_filled (actual data pixels) not buffer width, otherwise the
        # image is compressed when data doesn't fill the entire buffer.
        if n_filled == 0:
            return
        self.graphic.local.x = t_start
        self.graphic.local.scale_x = (t_end - t_start) / n_filled

    def rescale(self, key):
        """Adjust color limits with 'i' (narrow) or 'd' (widen)."""
        factor = {"i": 0.8, "d": 1.25}[key]
        lo, hi = self.graphic.material.clim
        center = (lo + hi) / 2
        half = (hi - lo) / 2
        new_half = half * factor
        self.graphic.material.clim = (center - new_half, center + new_half)

    def color_by(self, cmap_name, metadata_name, vmin, vmax, map_to_colors, values):
        """Change the image colormap and color limits."""
        cmap_map = getattr(gfx.cm, cmap_name, None)
        if cmap_map is not None:
            self.graphic.material.map = cmap_map
        self.graphic.material.clim = (vmin, vmax)

    def update_visibility(self):
        """Re-flush to shrink/grow the image based on visible channels."""
        self.flush(self.stream._slice_)

    def _get_visible_buffer(self):
        """Extract only visible channel rows from the buffer in display order.

        Returns the full buffer when all channels are visible, or a new array
        with only the visible rows otherwise.
        """
        if self.manager.is_sorted or self.manager.is_grouped:
            offsets = np.array([
                self.manager.data.loc[c]["offset"] for c in self.data.columns
            ])
            order = np.argsort(offsets)
        else:
            order = np.arange(len(self.data.columns))

        visible_mask = np.array([
            self.manager.data.loc[self.data.columns[idx]]["visible"]
            for idx in order
        ])

        if visible_mask.all():
            return self.buffer
        return self.buffer[visible_mask, :]

    def _reorder_image_rows(self, raw):
        """Reorder image rows according to manager offsets (sort/group order).

        Parameters
        ----------
        raw : np.ndarray
            Image data of shape (n_channels, n_timepoints) in original column order.

        Returns
        -------
        np.ndarray
            Reordered array with rows placed according to manager offsets.
        """
        if not (self.manager.is_sorted or self.manager.is_grouped):
            return raw

        offsets = np.array([
            self.manager.data.loc[c]["offset"]
            for c in self.data.columns
        ])
        order = np.argsort(offsets)
        return raw[order]

    def get_buffer_min_max(self):
        """Return per-time-point [min, max] from the current buffer.

        Returns
        -------
        np.ndarray
            Shape (max_n, 2).
        """
        return np.stack([np.nanmin(self.buffer, 0), np.nanmax(self.buffer, 0)]).T

    def get_state(self):
        """Return the colormap intensity range and visibility as a dict."""
        return {
            "clim": self.graphic.material.clim,
            "visible": self.manager.visible.tolist(),
        }

    def set_state(self, state):
        """Restore the colormap intensity range and visibility.

        Parameters
        ----------
        state : dict
            Value produced by :meth:`get_state`.
        """
        self.graphic.material.clim = state["clim"]
        if "visible" in state:
            self.manager.visible = state["visible"]
            self.update_visibility()


class XvsYMode:
    """Scatter/line x-vs-y display mode using a GetController.

    Plots one TsdFrame column against another. A red marker tracks the
    current time point.

    Parameters
    ----------
    data : nap.TsdFrame
        The time series data.
    manager : PlotTsdFrameManager
        Manages per-channel metadata.
    window_size : float, optional
        Unused (kept for interface consistency).
    """

    controller_key = "get"

    def __init__(self, data, manager, window_size=None, default_color="white"):
        self.data = data
        self.manager = manager
        self.window_size = window_size
        self.x_col = None
        self.y_col = None
        self.color = default_color
        self.thickness = 1.0
        self.markersize = 10.0
        self._request_draw = None

    def initialize_graphic(self):
        """Create the line graphic and time-point marker."""
        xy_values = self.data.loc[[self.x_col, self.y_col]].values.astype("float32")
        self.buffer = np.zeros((len(self.data), 3), dtype="float32")
        self.buffer[:, 0:2] = xy_values
        self.graphic = gfx.Line(
            gfx.Geometry(positions=self.buffer),
            gfx.LineMaterial(thickness=self.thickness, color=self.color),
        )
        xy = np.array([[0.0, 0.0, 1.0]], dtype="float32")
        self.time_point = gfx.Points(
            gfx.Geometry(positions=xy),
            gfx.PointsMaterial(size=self.markersize, color="red", opacity=1),
        )

    def update_parameters(self, x_col, y_col, color=None, thickness=1.0, markersize=10.0):
        """Set plotting parameters before calling initialize_graphic."""
        self.x_col = x_col
        self.y_col = y_col
        if color is not None:
            self.color = color
        self.thickness = thickness
        self.markersize = markersize

    def get_callbacks(self):
        """Return the frame-update callback for the GetController."""
        return [self._update_buffer]

    def _update_buffer(self, frame_index, event_type=None):
        """Move the time-point marker to the given frame index."""
        self.time_point.geometry.positions.data[0, 0:2] = self.graphic.geometry.positions.data[frame_index, 0:2]
        self.time_point.geometry.positions.update_full()
        if self._request_draw is not None:
            self._request_draw()

    def get_state(self):
        """Return all parameters needed to reconstruct this x-vs-y view.

        Returns
        -------
        dict
            Keys: ``window_size``, ``x_col``, ``y_col``, ``color``,
            ``thickness``, ``markersize``.
        """
        return {
            "window_size": self.window_size,
            "x_col": self.x_col,
            "y_col": self.y_col,
            "color": self.color,
            "thickness": self.thickness,
            "markersize": self.markersize,
        }

    def set_state(self, state):
        """Restore x-vs-y parameters from a previously saved state.

        Parameters
        ----------
        state : dict
            Value produced by :meth:`get_state`.
        """
        for k, v in state.items():
            setattr(self, k, v)
