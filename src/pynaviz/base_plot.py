"""
Simple plotting class for each pynapple object.
Create a unique canvas/renderer for each class
"""
import os
import sys
import threading
import warnings
from typing import Any, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pygfx as gfx
import pynapple as nap

# from line_profiler import profile
from matplotlib.colors import Colormap
from matplotlib.pyplot import colormaps

from .controller import GetController, SpanController, SpanYLockController
from .display_modes import ImageMode, LinesMode, XvsYMode
from .interval_set import IntervalSetInterface
from .plot_manager import _PlotManager
from .synchronization_rules import (
    _match_pan_on_x_axis,
    _match_set_xlim,
    _match_zoom_on_x_axis,
)
from .threads.metadata_to_color_maps import MetadataMappingThread
from .utils import (
    GRADED_COLOR_LIST,
    RenderTriggerSource,
    get_plot_attribute,
    get_plot_min_max,
    trim_kwargs,
)


def _is_headless():
    """Check if running in a headless environment across all platforms."""
    # Always headless in CI
    if os.environ.get('CI'):
        return True

    # Linux: check DISPLAY
    if sys.platform.startswith('linux'):
        return not os.environ.get('DISPLAY')

    # macOS and Windows: assume we have a display unless explicitly set to offscreen
    if os.environ.get('QT_QPA_PLATFORM') == 'offscreen':
        return True

    return False

if _is_headless():
    from rendercanvas.offscreen import loop
else:
    from rendercanvas.auto import loop


dict_sync_funcs = {
    "pan": _match_pan_on_x_axis,
    "zoom": _match_zoom_on_x_axis,
    "zoom_to_point": _match_zoom_on_x_axis,
    "set_xlim": _match_set_xlim,
}

spike_sdf = """
// Normalize coordinates relative to size
let uv = coord / size;
// Distance to vertical center line (x = 0)
let line_thickness = 0.2;
let dist = abs(uv.x) - line_thickness;
return dist * size;
"""


class _BasePlot(IntervalSetInterface):
    """
    Base class for time-aligned visualizations using pygfx.

    This class sets up the rendering infrastructure, including a canvas, scene,
    camera, rulers, and rendering thread. It is intended to be subclassed by specific
    plot implementations that display time series or intervals data.

    Parameters
    ----------
    data : Ts, Tsd, TsdFrame, IntervalSet or TsGroup object
        The dataset to be visualized. Must be pynapple object.
    parent : Optional[Any], default=None
        Optional parent widget for integration in GUI applications.
    maintain_aspect : bool, default=False
        If True, maintains the aspect ratio in the orthographic camera.

    Attributes
    ----------
    _data : Ts, Tsd, TsdFrame, IntervalSet or TsGroup object
        Pynapple object
    canvas : RendererCanvas
        The rendering canvas using the WGPU backend.
    color_mapping_thread : MetadataMappingThread
        A separate thread for mapping metadata to visual colors.
    renderer : gfx.WgpuRenderer
        The WGPU renderer responsible for drawing the scene.
    scene : gfx.Scene
        The scene graph containing all graphical objects.
    ruler_x : gfx.Ruler
        Horizontal axis ruler with ticks shown on the right.
    ruler_y : gfx.Ruler
        Vertical axis ruler with ticks on the left and minimum spacing.
    ruler_ref_time : gfx.Line
        A vertical line indicating a reference time point (e.g., center).
    camera : gfx.OrthographicCamera
        Orthographic camera with optional aspect ratio locking.
    _cmap : str
        Default colormap name used for visual mapping (e.g., "viridis").
    """

    def __init__(self, data, parent=None, maintain_aspect=False):
        super().__init__()

        # Store the input data for later use
        self._data = data

        # Create a GPU-accelerated canvas for rendering, optionally with a parent widget
        if parent:  # Assuming it's a Qt background
            from rendercanvas.qt import RenderCanvas
            self.canvas = RenderCanvas(parent=parent)
        else:
            if _is_headless():
                from rendercanvas.offscreen import RenderCanvas
            else:
                from rendercanvas.auto import RenderCanvas
            self.canvas = RenderCanvas()

        # Create a WGPU-based renderer attached to the canvas
        self.renderer = gfx.WgpuRenderer(
            self.canvas
        )  ## 97% time of super.__init__(...) when running `large_nwb_main.py`

        # Create a new scene to hold and manage objects
        self.scene = gfx.Scene()

        # Add a horizontal ruler (x-axis) with ticks above
        self.ruler_x = gfx.Ruler(
            tick_side="left"
        )

        # Add a vertical ruler (y-axis) with ticks on the left and minimum spacing
        self.ruler_y = gfx.Ruler(tick_side="left")

        # A vertical reference line, for the center time point
        self.ruler_ref_time = gfx.Line(
            gfx.Geometry(positions=[[0, 0, 0], [0, 0, 0]]),  # Placeholder geometry
            gfx.LineMaterial(thickness=0.5, color="#B4F8C8"),  # Thin light green line
        )

        # Render rulers on top of graphics (lines/images)
        for ruler in (self.ruler_x, self.ruler_y, self.ruler_ref_time):
            ruler.render_order = 1
            for child in ruler.children:
                child.render_order = 1

        # Use an orthographic camera to preserve scale without perspective distortion
        self.camera = gfx.OrthographicCamera(maintain_aspect=maintain_aspect)

        # Initialize a separate thread to handle metadata-to-color mapping
        self.color_mapping_thread = MetadataMappingThread(data)

        # Set default colormap for rendering
        self._cmap = "viridis"

        # Set the plot manager that store past actions only for data with metadata class
        if isinstance(data, (nap.TsGroup, nap.IntervalSet)):
            index = data.index
        elif isinstance(data, nap.TsdFrame):
            index = data.columns
        else:
            index = []
        self._manager = _PlotManager(index=index, base_plot=self)


    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, data):
        self.color_mapping_thread.update_maps(data)
        self._data = data

    @property
    def cmap(self):
        return self._cmap

    @cmap.setter
    def cmap(self, value):
        if isinstance(value, Colormap) and hasattr(value, "name"):
            self._cmap = value.name
        elif not isinstance(value, str):
            warnings.warn(
                message=f"Invalid colormap {value}. 'cmap' must be a matplotlib 'Colormap'.",
                category=UserWarning,
                stacklevel=2,
            )
            return
        if value not in plt.colormaps():
            warnings.warn(
                message=f"Invalid colormap {value}. 'cmap' must be a matplotlib 'Colormap'.",
                category=UserWarning,
                stacklevel=2,
            )
            return
        self._cmap = value

    def animate(self):
        """
        Updates the positions of rulers and reference lines based on the current
        world coordinate bounds and triggers a re-render of the scene.

        This method performs the following:
        - Computes the visible world coordinate bounds.
        - Updates the horizontal (x) and vertical (y) rulers accordingly.
        - Repositions the center time reference line in the scene.
        - Re-renders the scene using the current camera and canvas settings.

        Notes
        -----
        This method should be called whenever the visible region of the plot
        changes (e.g., after zooming, panning, or resizing the canvas).
        """
        world_xmin, world_xmax, world_ymin, world_ymax = get_plot_min_max(self)

        # X axis
        self.ruler_x.start_pos = world_xmin, 0, -10
        self.ruler_x.end_pos = world_xmax, 0, -10
        self.ruler_x.start_value = self.ruler_x.start_pos[0]
        self.ruler_x.update(self.camera, self.canvas.get_logical_size())

        # Y axis
        self.ruler_y.start_pos = 0, world_ymin, -10
        self.ruler_y.end_pos = 0, world_ymax, -10
        self.ruler_y.start_value = self.ruler_y.start_pos[1]
        self.ruler_y.update(self.camera, self.canvas.get_logical_size())

        # Center time Ref axis
        self.ruler_ref_time.geometry.positions.data[:, 0] = (
            world_xmin + (world_xmax - world_xmin) / 2
        )
        self.ruler_ref_time.geometry.positions.data[:, 1] = np.array(
            [world_ymin - 10, world_ymax + 10]
        )
        self.ruler_ref_time.geometry.positions.update_full()

        self.renderer.render(self.scene, self.camera)

    def show(self):
        """To show the canvas in case of GLFW context used"""
        loop.run()

    def color_by(
        self,
        metadata_name: str,
        cmap_name: str = "viridis",
        vmin: float = 0.0,
        vmax: float = 100.0,
    ) -> None:
        """
        Applies color mapping to plot elements based on a metadata field.

        This method retrieves values from the given metadata field and maps them
        to colors using the specified colormap and value range. The mapped colors
        are applied to each plot element's material. If color mappings are still
        being computed in a background thread, the function retries after a short delay.

        Parameters
        ----------
        metadata_name : str
            Name of the metadata field used for color mapping.
        cmap_name : str, default="viridis"
            Name of the colormap to apply (e.g., "jet", "plasma", "viridis").
        vmin : float, default=0.0
            Minimum value for the colormap normalization.
        vmax : float, default=100.0
            Maximum value for the colormap normalization.

        Notes
        -----
        - If the `color_mapping_thread` is still running, the method defers execution
          by 25 milliseconds and retries automatically.
        - If no appropriate color map is found for the metadata, a warning is issued.
        - Requires `self.data` to support `get_info()` for metadata retrieval.
        - Triggers a canvas redraw by calling `self.animate()` after updating colors.

        Warnings
        --------
        UserWarning
            Raised when the specified metadata field has no associated color mapping.
        """
        # If the color mapping thread is still processing, retry in 25 milliseconds
        if self.color_mapping_thread.is_running():
            slot = lambda: self.color_by(
                metadata_name, cmap_name=cmap_name, vmin=vmin, vmax=vmax
            )
            threading.Timer(0.025, slot).start()
            return

        # Set the current colormap
        self.cmap = cmap_name

        # Get the metadata-to-color mapping function for the given metadata field
        map_to_colors = self.color_mapping_thread.color_maps.get(metadata_name, None)

        # Warn the user if the color map is missing
        if map_to_colors is None:
            warnings.warn(
                message=f"Cannot find appropriate color mapping for {metadata_name} metadata.",
                category=UserWarning,
                stacklevel=2,
            )
        else:
            # Prepare keyword arguments for the color mapping function
            map_kwargs = trim_kwargs(
                map_to_colors, dict(cmap=colormaps[self.cmap], vmin=vmin, vmax=vmax)
            )

            # Get the material objects that will have their colors updated
            materials = get_plot_attribute(self, "material")

            # Get the metadata values for each plotted element
            values = (
                self.data.get_info(metadata_name) if hasattr(self.data, "get_info") else {}
            )

            # If metadata is found and mapping works, update the material colors
            if len(values):
                map_color = map_to_colors(values, **map_kwargs)
                if map_color:
                    for c in materials:
                        materials[c].color = map_color[values[c]]

                    # Request a redraw of the canvas to reflect the new colors
                    self.canvas.request_draw(self.animate)
            self._manager.color_by(values, metadata_name=metadata_name, cmap_name=cmap_name, vmin=vmin, vmax=vmax)

    def sort_by(self, metadata_name: str, mode: Optional[str] = "ascending"):
        pass

    def group_by(self, metadata_name: str):
        pass

    def close(self):
        if hasattr(self, "color_mapping_thread"):
            self.color_mapping_thread.shutdown()
        if hasattr(self, "canvas"):
            if self.canvas is not None:
                self.canvas.close()
            self.canvas = None
        for attr in ["renderer", "scene", "camera", "ruler_x", "ruler_y", "ruler_ref_time"]:
            if hasattr(self, attr):
                setattr(self, attr, None)

    @staticmethod
    def _initialize_offset(index: list) -> np.ndarray:
        return np.zeros(len(index))


class PlotTsd(_BasePlot):
    """
    Visualization for 1-dimensional pynapple time series object (``nap.Tsd``)

    This class renders a continuous 1D time series as a line plot and manages
    user interaction through a `SpanController`. It supports optional synchronization
    across multiple plots and rendering via WebGPU.

    Parameters
    ----------
    data : nap.Tsd
        The time series data to be visualized (timestamps + values).
    index : Optional[int], default=None
        Controller index used for synchronized interaction (e.g., panning across multiple plots).
    parent : Optional[Any], default=None
        Optional parent widget (e.g., in a Qt context).

    Attributes
    ----------
    controller : SpanController
        Manages viewport updates, syncing, and linked plot interactions.
    line : gfx.Line
        The main line plot showing the time series.
    """

    def __init__(
        self, data: nap.Tsd, index: Optional[int] = None, parent: Optional[Any] = None
    ) -> None:
        super().__init__(data=data, parent=parent)

        # Create a controller for span-based interaction, syncing, and user inputs
        self.controller = SpanController(
            camera=self.camera,
            renderer=self.renderer,
            controller_id=index,
            dict_sync_funcs=dict_sync_funcs,
            plot_callbacks=[],
        )

        # Prepare geometry: stack time, data, and zeros (Z=0) into (N, 3) float32 positions
        positions = np.stack((data.t, data.d, np.zeros_like(data.d))).T
        positions = positions.astype("float32")

        # Create a line geometry and material to render the time series
        self.line = gfx.Line(
            gfx.Geometry(positions=positions),
            gfx.LineMaterial(thickness=4.0, color="#aaf"),  # light blue line
        )

        # Add rulers and line to the scene
        self.scene.add(self.ruler_x, self.ruler_y, self.ruler_ref_time, self.line)

        # By default showing only the first second.
        # Weirdly rulers don't show if show_rect is not called
        # in the init
        # self.camera.show_rect(0, 1, data.min(), data.max())
        minmax = np.array([np.nanmin(data.d), np.nanmax(data.d)])
        if np.any(np.isnan(minmax)):
            minmax = np.array([0, 1])
        self.controller.set_view(0, 1, float(minmax[0]), float(minmax[1]))

        # Request an initial draw of the scene
        self.canvas.request_draw(self.animate)


class PlotTsdFrame(_BasePlot):
    """Visualization of a multi-column pynapple time series (``nap.TsdFrame``).

    Supports three display modes toggled at runtime:

    - **lines** (default): each column rendered as a line with vertex colors.
    - **image**: all columns as a heatmap texture.
    - **x_vs_y**: one column plotted against another with a time marker.

    Parameters
    ----------
    data : nap.TsdFrame
        The column-based time series data.
    index : int, optional
        Controller ID for synchronizing with other plots.
    parent : optional
        GUI parent widget (e.g. QWidget).
    window_size : float, optional
        Streaming window duration in seconds. If None, computed to fit
        up to 256 MB of memory.
    display_mode : str, default "lines"
        Initial display mode ("lines" or "image"). Toggle with the "m" key.
    """

    def __init__(
        self,
        data: nap.TsdFrame,
        index: Optional[int] = None,
        parent: Optional[Any] = None,
        window_size: Optional[float] = None,
        display_mode: str = "lines",
    ):
        super().__init__(data=data, parent=parent)
        self._data = data

        if display_mode not in ("lines", "image"):
            raise ValueError(f"display_mode must be 'lines' or 'image', got '{display_mode}'")
        self._display_mode = display_mode

        self._modes = {
            "lines": LinesMode(data, self._manager, window_size),
            "image": ImageMode(data, self._manager),
            "x_vs_y": XvsYMode(data, self._manager),
        }
        self._mode = self._modes[display_mode]
        self._mode.initialize_graphic()

        self.scene.add(self.ruler_x, self.ruler_y, self.ruler_ref_time, self._mode.graphic)

        self.renderer.add_event_handler(self._rescale, "key_down")
        self.renderer.add_event_handler(self._reset, "key_down")
        self.renderer.add_event_handler(self._toggle_display_mode, "key_down")

        self._controllers = {
            "span": SpanController(
                camera=self.camera,
                renderer=self.renderer,
                controller_id=index,
                dict_sync_funcs=dict_sync_funcs,
                plot_callbacks=self._modes["lines"].get_callbacks(),
            ),
            "span_ylock": SpanYLockController(
                camera=self.camera,
                renderer=self.renderer,
                dict_sync_funcs=dict_sync_funcs,
                plot_callbacks=self._modes["image"].get_callbacks(),
                enabled=False,
            ),
            "get": GetController(
                camera=self.camera,
                renderer=self.renderer,
                data=None,
                buffer=None,
                callback=self._modes["x_vs_y"]._update_buffer,
                enabled=False,
            ),
        }

        initial_key = self._mode.controller_key
        for key, ctrl in self._controllers.items():
            if key == initial_key:
                ctrl.enabled = True
                if ctrl._controller_id is None:
                    ctrl.controller_id = index
                self.controller = ctrl
            else:
                ctrl.enabled = False

        self._flush(start=0, end=1)

        minmax = self._get_min_max()
        self.controller.set_view(
            xmin=0, xmax=1, ymin=float(np.nanmin(minmax[:, 0])), ymax=float(np.nanmax(minmax[:, 1]))
        )
        self.canvas.request_draw(self.animate)

    @property
    def graphic(self):
        """The active pygfx graphic, delegated to the current display mode."""
        return self._mode.graphic

    def _flush(self, start, end):
        """Compute a data slice from (start, end) and flush the active mode."""
        slice_ = self._mode.stream.get_slice(start, end)
        self._mode.flush(slice_)

    def _get_min_max(self):
        """Return per-column [min, max] array of shape (n_columns, 2).

        Uses the raw data when available in memory, otherwise falls back
        to the current mode's buffer.
        """
        if isinstance(self.data.values, np.ndarray) and not isinstance(self.data.values, np.memmap):
            return np.stack([np.nanmin(self.data, 0), np.nanmax(self.data, 0)]).T

        minmax = self._mode.get_buffer_min_max()
        if np.any(np.isnan(minmax)):
            try:
                return np.array([np.nanmin(self.data[0:100], 0), np.nanmax(self.data[0:100], 0)]).T
            except Exception:
                return np.array([[0, 1]] * self.data.shape[1])
        return minmax

    def _rescale(self, event):
        """Handle 'i'/'d' keys to rescale amplitudes (lines) or clim (image)."""
        if event.type == "key_down" and event.key in ("i", "d"):
            self._mode.rescale(event.key)
            self.canvas.request_draw(self.animate)

    def _reset(self, event):
        """Handle 'r' key to reset the view. Transitions back to lines mode."""
        if event.type == "key_down" and event.key == "r":
            if self._display_mode == "image":
                self.scene.remove(self._mode.graphic)
                self._switch_controller("span_ylock", "span")
                self._display_mode = "lines"
                self._mode = self._modes["lines"]
                self._mode.initialize_graphic()
                self.scene.add(self._mode.graphic)

            elif self._display_mode == "x_vs_y":
                self.scene.remove(self._mode.graphic)
                self.scene.remove(self._mode.time_point)
                self._switch_controller("get", "span")
                self.scene.add(self.ruler_ref_time)
                self._display_mode = "lines"
                self._mode = self._modes["lines"]
                self._mode.initialize_graphic()
                self.scene.add(self._mode.graphic)

            self._manager.reset(self)
            self._flush(*self.controller.get_xlim())

            minmax = self._get_min_max()
            self.controller.set_ylim(float(np.nanmin(minmax[:, 0])), float(np.nanmax(minmax[:, 1])))
            self.canvas.request_draw(self.animate)

    def _switch_controller(self, from_key, to_key):
        """Disable *from_key* controller, enable *to_key*, and transfer the ID."""
        old = self._controllers[from_key]
        new = self._controllers[to_key]
        old.enabled = False
        cid = old._controller_id
        new.enabled = True
        if new._controller_id is None:
            new.controller_id = cid
        self.controller = new
        self.controller._send_switch_event()

    def _toggle_display_mode(self, event):
        """Toggle between 'lines' and 'image' on 'm' key."""
        if event.type == "key_down" and event.key == "m":
            if not isinstance(self.controller, SpanController):
                return

            old_key = self._mode.controller_key
            self.scene.remove(self._mode.graphic)

            self._display_mode = "image" if self._display_mode == "lines" else "lines"
            self._mode = self._modes[self._display_mode]
            self._mode.initialize_graphic()
            self.scene.add(self._mode.graphic)

            if old_key != self._mode.controller_key:
                self._switch_controller(old_key, self._mode.controller_key)

            self._flush(*self.controller.get_xlim())

            minmax = self._get_min_max()
            if self._display_mode == "image":
                self._mode.graphic.material.clim = (
                    float(np.nanmin(minmax[:, 0])),
                    float(np.nanmax(minmax[:, 1])),
                )
                self.controller.set_ylim(0, self.data.shape[1])
            else:
                self.controller.set_ylim(float(np.nanmin(minmax[:, 0])), float(np.nanmax(minmax[:, 1])))
            self.canvas.request_draw(self.animate)

    def _update(self, action_name):
        """Refresh the plot after a sort_by, group_by, or toggle_visibility action."""
        if action_name in ["sort_by", "group_by"]:
            if self._manager.is_sorted ^ self._manager.is_grouped:
                self._manager.scale = 1 / np.diff(self._get_min_max(), 1).flatten()

            self._manager.offset = self._manager.offset + 1 - self._manager.offset.min()
            self._flush(*self.controller.get_xlim())
            self.controller.set_ylim(0, np.max(self._manager.offset) + 1)

        if action_name == "toggle_visibility":
            self._mode.update_visibility()

        self.canvas.request_draw(self.animate)

    def sort_by(self, metadata_name: str, mode: Optional[str] = "ascending") -> None:
        """Sort channels vertically by a metadata field.

        Parameters
        ----------
        metadata_name : str
            Metadata key to sort by.
        mode : str, optional
            "ascending" (default) or "descending".
        """
        values = (
            dict(self.data.get_info(metadata_name))
            if hasattr(self.data, "get_info")
            else {}
        )
        if len(values):
            self._manager.sort_by(values, metadata_name=metadata_name, mode=mode)
            self._update("sort_by")

    def group_by(self, metadata_name: str, **kwargs):
        """Group channels vertically by a metadata field.

        Parameters
        ----------
        metadata_name : str
            Metadata key to group by.
        """
        values = (
            dict(self.data.get_info(metadata_name))
            if hasattr(self.data, "get_info")
            else {}
        )
        if len(values):
            self._manager.group_by(values, metadata_name=metadata_name, **kwargs)
            self._update("group_by")

    def color_by(
        self,
        metadata_name: str,
        cmap_name: str = "viridis",
        vmin: float = 0.0,
        vmax: float = 100.0,
    ) -> None:
        """Apply color mapping to channels based on a metadata field.

        Parameters
        ----------
        metadata_name : str
            Metadata field used for color mapping.
        cmap_name : str, default "viridis"
            Matplotlib colormap name.
        vmin : float, default 0.0
            Minimum value for colormap normalization.
        vmax : float, default 100.0
            Maximum value for colormap normalization.
        """
        if not self.color_mapping_thread.colormap_ready.is_set():
            slot = lambda: self.color_by(
                metadata_name, cmap_name=cmap_name, vmin=vmin, vmax=vmax
            )
            threading.Timer(0.025, slot).start()
            return

        self.cmap = cmap_name
        map_to_colors = self.color_mapping_thread.color_maps.get(metadata_name, None)

        if map_to_colors is None:
            warnings.warn(
                message=f"Cannot find appropriate color mapping for {metadata_name} metadata.",
                category=UserWarning,
                stacklevel=2,
            )
            return

        values = (
            self.data.get_info(metadata_name) if hasattr(self.data, "get_info") else {}
        )

        self._mode.color_by(cmap_name, metadata_name, vmin, vmax, map_to_colors, values)
        self.canvas.request_draw(self.animate)

    def plot_x_vs_y(
        self,
        x_col: Union[str, int, float],
        y_col: Union[str, int, float],
        color: Union[str, tuple] = "white",
        thickness: float = 1.0,
        markersize: float = 10.0,
    ) -> None:
        """Plot one column versus another with a time-point marker.

        Parameters
        ----------
        x_col : str or int or float
            Column name for the x-axis.
        y_col : str or int or float
            Column name for the y-axis.
        color : str or tuple, default "white"
            Line color.
        thickness : float, default 1.0
            Line thickness.
        markersize : float, default 10.0
            Size of the time marker.
        """
        if x_col not in self.data.columns or y_col not in self.data.columns:
            raise ValueError(f"Columns {x_col} and {y_col} must be in data columns.")

        self.scene.remove(self._mode.graphic)
        self._switch_controller(self._mode.controller_key, "get")

        self._display_mode = "x_vs_y"
        self._mode = self._modes["x_vs_y"]
        self._mode._request_draw = lambda: self.canvas.request_draw(self.animate)
        self._mode.update_parameters(x_col, y_col, color, thickness, markersize)
        self._mode.initialize_graphic()
        self.scene.add(self._mode.graphic)
        self.scene.add(self._mode.time_point)

        current_time = self.ruler_ref_time.geometry.positions.data[0][0]
        self.controller.frame_index = self.data.get_slice(current_time).start
        self.controller._current_time = current_time
        self.controller.data = self.data.loc[[x_col, y_col]]
        self.controller.buffer = self._mode.time_point.geometry.positions
        self._mode._update_buffer(self.controller.frame_index)

        self.controller.set_view(
            np.nanmin(self._mode.buffer[:, 0]), np.nanmax(self._mode.buffer[:, 0]),
            np.nanmin(self._mode.buffer[:, 1]), np.nanmax(self._mode.buffer[:, 1]),
        )
        self.canvas.request_draw(self.animate)


class PlotTsGroup(_BasePlot):
    """
    Visualization for plotting multiple spike trains (``nap.TsGroup``) as a raster plot.

    Each unit in the group is displayed as a row, where spike times are rendered as
    point markers (vertical ticks). Units can be sorted or grouped based on metadata.
    A `SpanController` is used to synchronize view ranges across plots.

    Parameters
    ----------
    data : nap.TsGroup
        A Pynapple `TsGroup` object containing multiple spike trains.
    index : int, optional
        Identifier for the controller instance, useful when synchronizing multiple plots.
    parent : QWidget, optional
        Parent widget in a Qt application, if applicable.
    """

    def __init__(self, data: nap.TsGroup, index=None, parent=None):
        # Initialize the base plot with provided data
        super().__init__(data=data, parent=parent)

        # Pynaviz-specific controller that handles pan/zoom and synchronization
        self.controller = SpanController(
            camera=self.camera,
            renderer=self.renderer,
            controller_id=index,
            dict_sync_funcs=dict_sync_funcs,  # shared synchronization registry
        )

        # Store PyGFX graphics objects (one per unit in TsGroup)
        self.graphic = {}

        # Iterate over each unit in the TsGroup and build its spike raster
        for i, n in enumerate(data.keys()):
            # Each spike is represented by its (time, row index, depth=1)
            positions = np.stack(
                (data[n].t, np.ones(len(data[n])) * i, np.ones(len(data[n])))
            ).T
            positions = positions.astype("float32")

            # Create a point cloud for the spikes of unit n
            self.graphic[n] = gfx.Points(
                gfx.Geometry(positions=positions),
                gfx.PointsMarkerMaterial(
                    size=10,
                    color=GRADED_COLOR_LIST[i % len(GRADED_COLOR_LIST)],  # assign color cyclically
                    opacity=1,
                    marker="custom",  # custom marker defined by spike_sdf
                    custom_sdf=spike_sdf,
                ),
            )

        # TODO: Implement streaming logic properly.
        # For now, initialize buffers with the first batch of data.
        self._buffers = {c: self.graphic[c].geometry.positions for c in self.graphic}
        self._flush()

        # Add rulers (axes and reference line) and all graphics to the scene
        self.scene.add(
            self.ruler_x,
            self.ruler_y,
            self.ruler_ref_time,
            *list(self.graphic.values()),
        )

        # Connect a key event handler ("r" key resets the view)
        self.renderer.add_event_handler(self._reset, "key_down")

        # By default, show the first second and full raster vertically
        self.controller.set_view(0, 1, 0, np.max(self._manager.offset) + 1)

        # Request continuous redrawing
        self.canvas.request_draw(self.animate)

    def _flush(self, slice_: slice = None):
        """
        Update the GPU buffers with the latest offsets and data slice.

        Parameters
        ----------
        slice_ : slice, optional
            Data slice to update. Not yet implemented.
        """
        # TODO: Implement slice-based updates (only redraw relevant portion)

        # Currently only updates y-offsets of spikes for each unit
        for c in self._buffers:
            self._buffers[c].data[:, 1] = self._manager.data.loc[c]["offset"].astype(
                "float32"
            )
            self._buffers[c].update_full()

    def _reset(self, event):
        """
        Reset the view to the initial state when pressing the "r" key.

        Parameters
        ----------
        event : gfx.Event
            Key event containing type and pressed key.
        """
        if event.type == "key_down" and event.key == "r":
            if isinstance(self.controller, SpanController):
                # Reset the internal plot manager (sorting, grouping, etc.)
                self._manager.reset(self)
                self._manager.data["offset"] = self.data.index
                self._flush()

            # Reset the vertical axis to show all units
            self.controller.set_ylim(0, np.max(self._manager.offset) + 1)
            self.canvas.request_draw(self.animate)

    def _update(self, action_name=None):
        """
        Update the raster after sorting or grouping operations.

        Parameters
        ----------
        action_name : str
            The action performed ("sort_by" or "group_by").
        """
        if action_name in ["sort_by", "group_by"]:
            self._flush()
            # Ensure camera spans the full y range
            self.controller.set_ylim(0, np.max(self._manager.offset) + 1)

        if action_name in ["toggle_visibility"]:
            # No need to flush. Just change the colors buffer
            for c in self.graphic:
                if not self._manager.data.loc[c]["visible"]:
                    self.graphic[c].material.opacity = 0.0
                else:
                    self.graphic[c].material.opacity = 1.0

        self.canvas.request_draw(self.animate)

    def sort_by(self, metadata_name: str, mode: Optional[str] = "ascending") -> None:
        """
        Sort the raster vertically by a metadata field.

        Parameters
        ----------
        metadata_name : str
            Metadata key to sort by.
        mode : str, optional
            "ascending" (default) or "descending".
        """
        # Grab metadata from TsGroup if available
        values = (
            dict(self.data.get_info(metadata_name))
            if hasattr(self.data, "get_info")
            else {}
        )

        if len(values):
            # Sort units in the plot manager by metadata values
            self._manager.sort_by(values, mode=mode, metadata_name=metadata_name)
            self._update("sort_by")

    def group_by(self, metadata_name: str, **kwargs):
        """
        Group the raster vertically by a metadata field.

        Parameters
        ----------
        metadata_name : str
            Metadata key to group by.
        """
        # Grab metadata from TsGroup if available
        values = (
            dict(self.data.get_info(metadata_name))
            if hasattr(self.data, "get_info")
            else {}
        )

        if len(values):
            # Group units in the plot manager by metadata values
            self._manager.group_by(values, metadata_name=metadata_name, **kwargs)
            self._update("group_by")

    @staticmethod
    def _initialize_offset(index: list) -> np.ndarray:
        return np.arange(len(index))


class PlotTs(_BasePlot):
    """
    Visualization for pynapple timestamps object (``nap.Ts``) as vertical tick marks.

    Parameters
    ----------
    data : nap.Ts
        A Pynapple `Ts` object containing spike timestamps to visualize.
    index : Optional[int], default=None
        Controller index used for synchronized interaction (e.g., panning across multiple plots).
    parent : Optional[Any], default=None
        Optional parent widget (e.g., in a Qt context).

    """

    def __init__(self, data: nap.Ts, index=None, parent=None):
        # Initialize the base plot with provided data
        super().__init__(data=data, parent=parent)

        # Disable aspect ratio lock (x and y can scale independently)
        self.camera.maintain_aspect = False

        # Create the Pynaviz-specific controller to lock/synchronize vertical spans
        self.controller = SpanYLockController(
            camera=self.camera,
            renderer=self.renderer,
            controller_id=index,
            dict_sync_funcs=dict_sync_funcs,  # external sync registry
        )

        # Number of timestamps in the data
        n = len(data)

        # Build positions array for vertical tick marks:
        # For each timestamp, repeat the time 3 times (x coordinate)
        # and pair it with (0, 1, NaN) in y to form a vertical line segment
        # (NaN ensures gaps between line segments in pygfx)
        positions = np.stack(
            (
                np.repeat(data.t, 3),                  # x: same timestamp repeated
                np.tile(np.array([0, 1, np.nan]), n),  # y: line from 0 to 1, then gap
                np.tile(np.array([1, 1, np.nan]), n)   # z: keep constant depth (1)
            )
        ).T
        positions = positions.astype("float32")

        # Create geometry and material for the tick marks
        geometry = gfx.Geometry(positions=positions)
        material = gfx.LineMaterial(thickness=2, color=(1, 0, 0, 1))  # solid red
        self.graphic = gfx.Line(geometry, material)

        # Add objects to the scene:
        # - x ruler (time axis)
        # - reference ruler (time marker)
        # - the spike train graphic
        self.scene.add(self.ruler_x, self.ruler_ref_time, self.graphic)

        # Connect key event handler for jumping to next/previous timestamp
        self.renderer.add_event_handler(self._jump, "key_down")

        # Adjust camera to show full data range:
        self.controller.set_view(0, 1, -0.1, 1.0)

        # Request continuous redrawing with animation
        self.canvas.request_draw(self.animate)

    def _jump(self, event):
        """
        Handle key events for jumping to next/previous timestamp.

        Parameters
        ----------
        event : gfx.Event
            Key event containing type and pressed key.
        """
        if event.type == "key_down":
            if event.key == "ArrowRight" or event.key == "n":
                self.jump_next()
            elif event.key == "ArrowLeft" or event.key == "p":
                self.jump_previous()

    def jump_next(self) -> None:
        """
        Jump to the next timestamp in the data.
        """
        current_t = self.controller._get_camera_state()["position"][0]
        index = np.searchsorted(self.data.index.values, current_t, side="right")
        index = np.clip(index, 0, len(self.data) - 1)
        new_t = self.data.index.values[index]
        if new_t > current_t:
            self.controller.go_to(new_t)

    def jump_previous(self) -> None:
        """
        Jump to the previous timestamp in the data.
        """
        current_t = self.controller._get_camera_state()["position"][0]
        index = np.searchsorted(self.data.index.values, current_t, side="left") - 1
        index = np.clip(index, 0, len(self.data) - 1)
        new_t = self.data.index.values[index]
        if new_t < current_t:
            self.controller.go_to(new_t)


class PlotIntervalSet(_BasePlot):
    """
    A visualization of a set of non-overlapping epochs (``nap.IntervalSet``).

    This class allows dynamic rendering of each interval in a `nap.IntervalSet`, with interactive
    controls for span navigation. It supports coloring, grouping, and sorting of intervals based on metadata.

    Parameters
    ----------
    data : nap.IntervalSet
        The set of intervals to be visualized, with optional metadata.
    index : Optional[int], default=None
        Unique ID for synchronizing with external controllers.
    parent : Optional[Any], default=None
        Optional GUI parent (e.g. QWidget in Qt).

    Attributes
    ----------
    controller : SpanController
        Active interactive controller for zooming or selecting.
    graphic : dict[str, gfx.Mesh] or gfx.Mesh
        Dictionary of rectangle meshes for each interval.
    """

    def __init__(self, data: nap.IntervalSet, index=None, parent=None):
        super().__init__(data=data, parent=parent)
        self.camera.maintain_aspect = False

        # Pynaviz specific controller
        self.controller = SpanYLockController(
            camera=self.camera,
            renderer=self.renderer,
            controller_id=index,
            dict_sync_funcs=dict_sync_funcs,
        )

        self.graphic = self._create_and_plot_rectangle(
            data, color="cyan", transparency=1
        )
        # set to default position
        self._update()
        self.scene.add(self.ruler_x, self.ruler_y, self.ruler_ref_time)

        # Connect specific event handler for IntervalSet
        self.renderer.add_event_handler(self._reset, "key_down")
        self.renderer.add_event_handler(self._jump, "key_down")

        # By default, showing only the first second.
        self.controller.set_view(0, 1, -0.1, 1)

        self.canvas.request_draw(self.animate)

    def _reset(self, event):
        """
        "r" key reset the plot manager to initial view
        """
        if event.type == "key_down":
            if event.key == "r":
                self._manager.reset(self)
                self._update()

    def _update(self, action_name: str = None):
        """
        Update function for sort_by and group_by
        """
        if action_name in ["sort_by", "group_by"] or action_name is None:
            # Update the scale only if one action has been performed
            # Grabbing the material object
            geometries = get_plot_attribute(self, "geometry")  # Dict index -> geometry

            for c in geometries:
                geometries[c].positions.data[:2, 1] = self._manager.data.loc[c][
                    "offset"
                ].astype("float32")
                geometries[c].positions.data[2:, 1] = (
                    self._manager.data.loc[c]["offset"].astype("float32") + 1
                )

                geometries[c].positions.update_full()

        if action_name in ["toggle_visibility"]:
            # No need to flush. Just change the colors buffer
            for c in self.graphic:
                if not self._manager.data.loc[c]["visible"]:
                    self.graphic[c].material.opacity = 0.0
                else:
                    self.graphic[c].material.opacity = 1.0

        # Update camera to fit the full y range
        ymax = np.max(self._manager.offset) + 1
        self.controller.set_ylim(-0.05 * ymax, ymax)

        # if action_name == "sort_by":
        if self._manager.y_ticks is not None:
            # shift y ticks by 0.5
            self.ruler_y.ticks = {k + 0.5: v for k, v in self._manager.y_ticks.items()}
        else:
            self.ruler_y.ticks = {0.5: ""}

        self.canvas.request_draw(self.animate)

    def sort_by(self, metadata_name: str, mode: Optional[str] = "ascending") -> None:
        """
        Vertically sort the plotted intervals by a metadata field.

        Parameters
        ----------
        metadata_name : str
            Metadata key to sort by.
        """

        values = (
            dict(self.data.get_info(metadata_name))
            if hasattr(self.data, "get_info")
            else {}
        )
        # If metadata found
        if len(values):

            # Sorting should happen depending on `groups` and `visible` attributes of _PlotManager
            self._manager.sort_by(values, mode=mode, metadata_name=metadata_name)
            self._update("sort_by")

    def group_by(self, metadata_name: str, **kwargs):
        """
        Group the intervals vertically by a metadata field.

        Parameters
        ----------
        metadata_name : str
            Metadata key to group by.
        """
        # Grabbing the metadata
        values = (
            dict(self.data.get_info(metadata_name))
            if hasattr(self.data, "get_info")
            else {}
        )

        # If metadata found
        if len(values):

            # Grouping positions are computed depending on `order` and `visible` attributes of _PlotManager
            self._manager.group_by(values, metadata_name=metadata_name, **kwargs)
            self._update("group_by")

    def _jump(self, event):
        """
        Handle key events for jumping to next/previous interval.

        Parameters
        ----------
        event : gfx.Event
            Key event containing type and pressed key.
        """
        if event.type == "key_down":
            if event.key == "ArrowRight" or event.key == "n":
                self.jump_next()
            elif event.key == "ArrowLeft" or event.key == "p":
                self.jump_previous()

    def jump_next(self) -> None:
        """
        Jump to the start of the next interval in the data.
        """
        current_t = self.controller._get_camera_state()["position"][0]
        index = np.searchsorted(self.data.start, current_t, side="right")
        index = np.clip(index, 0, len(self.data) - 1)
        new_t = self.data.start[index]
        if new_t > current_t:
            self.controller.go_to(new_t)

    def jump_previous(self) -> None:
        """
        Jump to the start of the previous interval in the data.
        """
        current_t = self.controller._get_camera_state()["position"][0]
        index = np.searchsorted(self.data.start, current_t, side="left") - 1
        index = np.clip(index, 0, len(self.data) - 1)
        new_t = self.data.start[index]
        if new_t < current_t:
            self.controller.go_to(new_t)

