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
from .interval_set import IntervalSetInterface
from .plot_manager import _PlotManager
from .synchronization_rules import (
    _match_pan_on_x_axis,
    _match_set_xlim,
    _match_zoom_on_x_axis,
)
from .threads.data_streaming import TsdFrameStreaming
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

        # Add a horizontal ruler (x-axis) with ticks on the right
        self.ruler_x = gfx.Ruler(tick_side="right")

        # Add a vertical ruler (y-axis) with ticks on the left and minimum spacing
        self.ruler_y = gfx.Ruler(tick_side="left")

        # A vertical reference line, for the center time point
        self.ruler_ref_time = gfx.Line(
            gfx.Geometry(positions=[[0, 0, 0], [0, 0, 0]]),  # Placeholder geometry
            gfx.LineMaterial(thickness=0.5, color="#B4F8C8"),  # Thin light green line
        )

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
    """
    Visualization of a multi-columns pynapple time series (``nap.TsdFrame``).

    This class allows dynamic rendering of each column in a `nap.TsdFrame`, with interactive
    controls for span navigation. It supports switching between
    standard time series display and scatter-style x-vs-y plotting between columns.

    Parameters
    ----------
    data : nap.TsdFrame
        The column-based time series data (columns as features).
    index : Optional[int], default=None
        Unique ID for synchronizing with external controllers.
    parent : Optional[Any], default=None
        Optional GUI parent (e.g. QWidget in Qt).
    window_size : float
        The time duration (in same units as data timestamps) of the streaming window.
        This parameter is optional and if not provided, it will be calculated to
        optimize memory usage (up to 256MB). It is better to provide it if you know
        have memory constraints.

    Attributes
    ----------
    controller : Union[SpanController, GetController]
        Active interactive controller for zooming or selecting.
    graphic : dict[str, gfx.Line] or gfx.Line
        Dictionary of per-column lines or single line for x-vs-y plotting.
    time_point : Optional[gfx.Points]
        A marker showing the selected time point (used in x-vs-y plotting).
    """

    def __init__(
        self,
        data: nap.TsdFrame,
        index: Optional[int] = None,
        parent: Optional[Any] = None,
        window_size: Optional[float] = None,
    ):
        super().__init__(data=data, parent=parent)
        self._data = data

        # To stream data
        if window_size is None:
            # Calculate window size to use up to 256MB of memory
            size = (256 * 1024**2) // (data.shape[1] * 60)
            window_size = np.floor(size / data.rate) # seconds

        self._stream = TsdFrameStreaming(
            data, callback=self._flush, window_size=window_size
        )

        plot_callbacks = []

        if self._stream._max_n < data.shape[0]:
            plot_callbacks = [self._stream.stream] # For controller

        # Create pygfx objects
        self._positions = np.full(
            ((self._stream._max_n + 1) * self.data.shape[1], 3), np.nan, dtype="float32"
        )

        self._buffer_slices = {}
        for c, s in zip(
            self.data.columns,
            range(
                0,
                len(self._positions) - self._stream._max_n + 1,
                self._stream._max_n + 1,
            ),
        ):
            self._buffer_slices[c] = slice(s, s + self._stream._max_n)

        self._positions[:, 2] = 0.0

        # Create pygfx object
        self._initialize_graphic()

        # Add elements to the scene for rendering
        self.scene.add(self.ruler_x, self.ruler_y, self.ruler_ref_time, self.graphic)

        # Connect specific event handler for TsdFrame
        self.renderer.add_event_handler(self._rescale, "key_down")
        self.renderer.add_event_handler(self._reset, "key_down")

        # Controllers for different interaction styles
        self._controllers = {
            "span": SpanController(
                camera=self.camera,
                renderer=self.renderer,
                controller_id=index,
                dict_sync_funcs=dict_sync_funcs,
                plot_callbacks=plot_callbacks,
            ),
            "get": GetController(
                camera=self.camera,
                renderer=self.renderer,
                data=None,
                buffer=None,
                enabled=False,
                callback=self._update_buffer,
            ),
        }

        # Use span controller by default
        self.controller = self._controllers["span"]

        # Used later in x-vs-y plotting
        self.time_point = None

        # By default, showing only the first second.
        # this should flush the whole dataset if data fits into memory
        self._flush(self._stream.get_slice(start=0, end=1))

        # Setting the boundaries of the plot
        # minmax is of shape (n_columns, 2)
        minmax = self._get_min_max()
        self.controller.set_view(0, 1, np.min(minmax[:, 0]), np.max(minmax[:, 1]))

        # Request an initial draw of the scene
        self.canvas.request_draw(self.animate)

    def _initialize_graphic(self):
        colors = np.ones((self._positions.shape[0], 4), dtype=np.float32)

        self.graphic = gfx.Line(
            gfx.Geometry(positions=self._positions, colors=colors),
            gfx.LineMaterial(
                thickness=1.0, color_mode="vertex"
            ),  # , color=GRADED_COLOR_LIST[1 % len(GRADED_COLOR_LIST)]),
        )

    def _flush(self, slice_: slice = None):
        """
        Flush the data stream from slice_ argument.
        The slice argument should be obtained from the _get_slice method of the TsdFrameStreaming object.
        """
        if self._stream._max_n == self.data.shape[0]:
            # If data fit into memory
            for i, c in enumerate(self.data.columns):
                sl = self._buffer_slices[c]
                self._positions[sl, 0] = self.data.t.astype("float32")
                self._positions[sl, 1] = self.data.d[:, i].astype("float32")
                self._positions[sl, 1] *= self._manager.data.loc[c]["scale"]
                self._positions[sl, 1] += self._manager.data.loc[c]["offset"]

        else:
            if slice_ is None:
                slice_ = self._stream.get_slice(*self.controller.get_xlim())

            time = self.data.t[slice_].astype("float32")

            left_offset = 0
            right_offset = 0
            if time.shape[0] < self._stream._max_n:
                if slice_.start == 0:
                    left_offset = self._stream._max_n - time.shape[0]
                else:
                    right_offset = time.shape[0] - self._stream._max_n

            # Read
            data = np.array(self.data.values[slice_, :])

            # Copy the data
            for i, c in enumerate(self.data.columns):
                sl = self._buffer_slices[c]
                sl = slice(sl.start + left_offset, sl.stop + right_offset)
                self._positions[sl, 0] = time
                self._positions[sl, 1] = data[:, i]
                self._positions[sl, 1] *= self._manager.data.loc[c]["scale"]
                self._positions[sl, 1] += self._manager.data.loc[c]["offset"]

            # Put back some nans on the edges
            if left_offset:
                for sl in self._buffer_slices.values():
                    self._positions[sl.start : sl.start + left_offset, 0:2] = np.nan
            if right_offset:
                for sl in self._buffer_slices.values():
                    self._positions[sl.stop + right_offset : sl.stop, 0:2] = np.nan

        self.graphic.geometry.positions.set_data(self._positions)

    def _get_min_max(self):
        """
        If the data object is a numpy array, get min max directly from it
        otherwise get min max from the buffer.
        """
        if isinstance(self.data.values, np.ndarray) and not isinstance(self.data.values, np.memmap):
            return np.stack([np.nanmin(self.data, 0), np.nanmax(self.data, 0)]).T
        else:
            minmax = np.array(
                [
                    [np.nanmin(self._positions[sl, 1]), np.nanmax(self._positions[sl, 1])]
                    for sl in self._buffer_slices.values()
                ]
            )
            if np.any(np.isnan(minmax)):
                # Try to get min max from 100 points of each column
                try:
                    minmax = np.array([np.nanmin(self.data[0:100], 0), np.nanmax(self.data[0:100], 0)]).T
                    return minmax
                except Exception:
                    return np.array([[0, 1]] * self.data.shape[1])
            else:
                return minmax

    def _rescale(self, event):
        """
        "i" key increase the scale by 50%.
        "d" key decrease the scale by 50%
        """
        if self._manager.is_sorted or self._manager.is_grouped:
            if event.type == "key_down":
                if event.key == "i" or event.key == "d":
                    factor = {"i": 0.5, "d": -0.5}[event.key]

                    # Update the scale of the PlotManager
                    self._manager.rescale(factor=factor)

                    # Update the current buffers to avoid re-reading from disk
                    for c, sl in self._buffer_slices.items():
                        self._positions[sl, 1] += factor * (
                            self._positions[sl, 1] - self._manager.data.loc[c]["offset"]
                        )

                    # Update the gpu data
                    self.graphic.geometry.positions.set_data(self._positions)
                    self.canvas.request_draw(self.animate)

    def _reset(self, event):
        """
        "r" key reset the plot manager to initial view
        """
        if event.type == "key_down":
            if event.key == "r":
                if isinstance(self.controller, SpanController):
                    self._manager.reset(self)
                    self._flush()

                if isinstance(self.controller, GetController):
                    self.scene.remove(self.graphic, self.time_point)
                    # Switch back to SpanController
                    self.controller.enabled = False
                    controller_id = self.controller._controller_id
                    self.controller = self._controllers["span"]
                    self.controller._controller_id = controller_id
                    self.controller.enabled = True
                    self._initialize_graphic()
                    self.scene.add(self.graphic)
                    self.scene.add(self.ruler_ref_time)
                    self._manager.reset(self)
                    self._flush()

                minmax = self._get_min_max()

                self.controller.set_ylim(np.min(minmax[:, 0]), np.max(minmax[:, 1]))
                self.controller.set_xlim(0, 1)
                self.canvas.request_draw(self.animate)

    def _update(self, action_name):
        """
        Update function for sort_by and group_by. Because of mode of sort_by, it's not possible
        to just update the buffer.
        """
        if action_name in ["sort_by", "group_by"]:
            # Update the scale only if one action has been performed
            if self._manager.is_sorted ^ self._manager.is_grouped:
                self._manager.scale = 1 / np.diff(self._get_min_max(), 1).flatten()

            # Specific to PloTsdFrame, the first row should be at 1.
            self._manager.offset = self._manager.offset + 1 - self._manager.offset.min()

            # Update the buffer
            self._flush()

            # Update camera to fit the full y range
            self.controller.set_ylim(0, np.max(self._manager.offset) + 1)

        if action_name in ["toggle_visibility"]:
            # No need to flush. Just change the colors buffer
            new_colors = self.graphic.geometry.colors.data.copy()
            for c, sl in self._buffer_slices.items():
                if not self._manager.data.loc[c]["visible"]:
                    new_colors[sl, -1] = 0.0
                else:
                    new_colors[sl, -1] = 1.0
            self.graphic.geometry.colors.set_data(new_colors)

        self.canvas.request_draw(self.animate)

    def sort_by(self, metadata_name: str, mode: Optional[str] = "ascending") -> None:
        """
        Sort the plotted time series lines vertically by a metadata field.

        Parameters
        ----------
        metadata_name : str
            Metadata key to sort by.
        mode : str, optional
            "ascending" (default) or "descending".
        """
        # The current controller should be a span controller.

        # Grabbing the metadata
        values = (
            dict(self.data.get_info(metadata_name))
            if hasattr(self.data, "get_info")
            else {}
        )

        # If metadata found
        if len(values):
            # Sorting should happen depending on `groups` and `visible` attributes of _PlotManager
            self._manager.sort_by(values, metadata_name=metadata_name, mode=mode)
            self._update("sort_by")

    def group_by(self, metadata_name: str, **kwargs):
        """
        Group the plotted time series lines vertically by a metadata field.

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
        # print(self.color_mapping_thread.color_maps, self.color_mapping_thread.colormap_ready.is_set())
        if not self.color_mapping_thread.colormap_ready.is_set():
            # print(self.color_mapping_thread.color_maps, self.color_mapping_thread.colormap_ready.is_set())
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

        # Prepare keyword arguments for the color mapping function
        map_kwargs = trim_kwargs(
            map_to_colors, dict(cmap=colormaps[self.cmap], vmin=vmin, vmax=vmax)
        )

        # # Get the material objects that will have their colors updated
        # materials = get_plot_attribute(self, "material")

        # Get the metadata values for each plotted element
        values = (
            self.data.get_info(metadata_name) if hasattr(self.data, "get_info") else {}
        )

        # If metadata is found and mapping works, update the material colors
        if len(values):
            map_color = map_to_colors(values, **map_kwargs)
            if map_color:
                for c, sl in self._buffer_slices.items():
                    self.graphic.geometry.colors.data[sl, :] = map_color[values[c]]
                    # self.graphic.material.color = map_color[values[c]]
                self.graphic.geometry.colors.update_full()
                # Request a redraw of the canvas to reflect the new colors
                self.canvas.request_draw(self.animate)
        self._manager.color_by(values, metadata_name, cmap_name=cmap_name, vmin=vmin, vmax=vmax)

    def plot_x_vs_y(
        self,
        x_col: Union[str, int, float],
        y_col: Union[str, int, float],
        color: Union[str, tuple] = "white",
        thickness: float = 1.0,
        markersize: float = 10.0,
    ) -> None:
        """
        Plot one column versus another as a line plot.

        Parameters
        ----------
        x_col : str or int or float
            Column name for the x-axis.
        y_col : str or int or float
            Column name for the y-axis.
        color : str or hex or RGB, default="white"
            Line color.
        thickness : float, default=1.0
            Thickness of the connecting line.
        markersize : float, default=10.0
            Size of the time marker.
        """
        if x_col not in self.data.columns or y_col not in self.data.columns:
            raise ValueError(f"Columns {x_col} and {y_col} must be in data columns.")

        # Remove time series line graphics from the scene
        self.scene.remove(self.graphic)
        self.scene.remove(self.time_point) if self.time_point else None

        # Remove intervals if any
        if len(self._epochs):
            keys = list(self._epochs.keys())
            for epoch in keys:
                self.remove_interval_set(epoch)

        # Get current time from the center reference line
        current_time = self.ruler_ref_time.geometry.positions.data[0][0]
        self.scene.remove(self.ruler_ref_time)

        # Build new geometry for x-y data
        xy_values = self.data.loc[[x_col, y_col]].values.astype("float32")
        positions = np.zeros((len(self.data), 3), dtype="float32")
        positions[:, 0:2] = xy_values

        # Create new line and add it to the scene
        self.graphic = gfx.Line(
            gfx.Geometry(positions=positions),
            gfx.LineMaterial(thickness=thickness, color=color),
        )
        self.scene.add(self.graphic)

        # Create and add a point marker at the current time
        current_xy = self.data.loc[[x_col, y_col]].get(current_time)
        xy = np.hstack((current_xy, 1), dtype="float32")[None, :]
        self.time_point = gfx.Points(
            gfx.Geometry(positions=xy),
            gfx.PointsMaterial(size=markersize, color="red", opacity=1),
        )
        self.scene.add(self.time_point)

        # Disable span controller and switch to get controller
        self.controller.enabled = False
        controller_id = self.controller._controller_id

        get_controller = self._controllers["get"]
        get_controller.n_frames = len(self.data)
        get_controller.frame_index = self.data.get_slice(current_time).start
        get_controller._current_time = current_time
        get_controller.enabled = True
        get_controller._controller_id = controller_id
        get_controller.data = self.data.loc[[x_col, y_col]]
        get_controller.buffer = self.time_point.geometry.positions

        self.controller = get_controller
        # In case the plot is part of a ControllerGroup, we need to notify the change
        self.controller._send_switch_event()

        # Update camera to fit the full x-y range
        self.controller.set_view(
            np.nanmin(positions[:, 0]),
            np.nanmax(positions[:, 0]),
            np.nanmin(positions[:, 1]),
            np.nanmax(positions[:, 1])
        )

        self.canvas.request_draw(self.animate)

    def _update_buffer(self, frame_index: int, event_type: Optional[RenderTriggerSource] = None):
        """
        For get controller
        """
        self.time_point.geometry.positions.data[0,0:2] = self.graphic.geometry.positions.data[frame_index,0:2]
        self.time_point.geometry.positions.update_full()
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

