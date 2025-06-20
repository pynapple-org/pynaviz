"""
Simple plotting class for each pynapple object.
Create a unique canvas/renderer for each class
"""

import abc
import atexit
import pathlib
import queue
import signal
import sys
import threading
import warnings
import weakref
from abc import ABC, abstractmethod
from multiprocessing import Event, Process, Queue, set_start_method, shared_memory
from typing import Any, Optional, Union

import av
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pygfx as gfx
import pynapple as nap
from matplotlib.colors import Colormap
from matplotlib.pyplot import colormaps
from numpy.typing import NDArray
from wgpu.gui.auto import (
    WgpuCanvas,  # Should use auto here or be able to select qt if parent passed
    run,
)
from wgpu.gui.glfw import GlfwWgpuCanvas

from .controller import GetController, SpanController
from .interval_set import IntervalSetInterface
from .plot_manager import _PlotManager
from .synchronization_rules import _match_pan_on_x_axis, _match_zoom_on_x_axis
from .threads.data_streaming import TsdFrameStreaming
from .threads.metadata_to_color_maps import MetadataMappingThread
from .utils import GRADED_COLOR_LIST, get_plot_attribute, get_plot_min_max, trim_kwargs
from .video_handling import VideoHandler
from .video_worker import RenderTriggerSource, video_worker_process

# WeakSet to avoid keeping dead references
_active_plot_videos = weakref.WeakSet()


def _cleanup_all_plot_videos():
    for video in list(_active_plot_videos):
        try:
            video.close()
        except Exception as e:
            print(f"[WARN] Error during close: {e}")
    _active_plot_videos.clear()


# Register cleanup at process exit
atexit.register(_cleanup_all_plot_videos)
signal.signal(signal.SIGINT, lambda *_: (_cleanup_all_plot_videos(), sys.exit(0)))
signal.signal(signal.SIGTERM, lambda *_: (_cleanup_all_plot_videos(), sys.exit(0)))

if sys.platform != "win32":
    try:
        set_start_method("fork", force=True)
    except RuntimeError:
        pass


dict_sync_funcs = {
    "pan": _match_pan_on_x_axis,
    "zoom": _match_zoom_on_x_axis,
    "zoom_to_point": _match_zoom_on_x_axis,
}


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
    canvas : WgpuCanvas
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
            self.canvas = WgpuCanvas(parent=parent)
        else:  # Default to glfw for single canvas
            self.canvas = WgpuCanvas()

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
        self._manager = _PlotManager(index=index)

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
                message=f"Invalid colormap {value}. 'cmap' must be matplotlib 'Colormap'.",
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
        if isinstance(self.canvas, GlfwWgpuCanvas):
            run()

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
            slot = lambda: self.color_by(metadata_name, cmap_name=cmap_name, vmin=vmin, vmax=vmax)
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

        # Get the material objects that will have their colors updated
        materials = get_plot_attribute(self, "material")

        # Get the metadata values for each plotted element
        values = self.data.get_info(metadata_name) if hasattr(self.data, "get_info") else {}

        # If metadata is found and mapping works, update the material colors
        if len(values):
            map_color = map_to_colors(values, **map_kwargs)
            if map_color:
                for c in materials:
                    materials[c].color = map_color[values[c]]

                # Request a redraw of the canvas to reflect the new colors
                self.canvas.request_draw(self.animate)

    def sort_by(self, metadata_name: str, mode: Optional[str] = "ascending"):
        pass

    def group_by(self, metadata_name: str):
        pass

    def close(self):
        self.color_mapping_thread.shutdown()


class PlotTsd(_BasePlot):
    """
    A time series plot for `nap.Tsd` objects using GPU-accelerated rendering.

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
        positions = np.stack((data.t, data.d, np.zeros_like(data))).T
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
        self.camera.show_rect(0, 1, data.min(), data.max())

        # Request an initial draw of the scene
        self.canvas.request_draw(self.animate)


class PlotTsdFrame(_BasePlot):
    """
    A GPU-accelerated visualization of a multi-columns time series (nap.TsdFrame).

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
    ):
        super().__init__(data=data, parent=parent)
        self.data = data

        # Initialize lines for each column in the TsdFrame
        self.graphic: dict[str, gfx.Line] = {}

        # To stream data
        self._stream = TsdFrameStreaming(data, callback=self._flush, window_size=3)  # seconds

        # Create pygfx objects
        for i, c in enumerate(self.data.columns):
            positions = np.zeros((len(self._stream), 3), dtype="float32")
            self.graphic[c] = gfx.Line(
                gfx.Geometry(positions=positions),
                gfx.LineMaterial(
                    thickness=1.0, color=GRADED_COLOR_LIST[i % len(GRADED_COLOR_LIST)]
                ),
            )

        # Stream the first batch of data
        self._buffers = {c: self.graphic[c].geometry.positions for c in self.graphic}
        self._flush(self._stream.get_slice(start=0, end=1))

        # Add elements to the scene for rendering
        self.scene.add(self.ruler_x, self.ruler_y, self.ruler_ref_time, *self.graphic.values())

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
                plot_callbacks=[self._stream.stream],
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
        minmax = self._get_min_max()
        self.controller.set_view(0, 1, np.min(minmax[:, 0]), np.max(minmax[:, 1]))

        # Request an initial draw of the scene
        self.canvas.request_draw(self.animate)

    def _update_buffer(self, frame_index: int):
        _update_buffer(self, frame_index=frame_index)
        self.controller.renderer_request_draw()

    def _flush(self, slice_: slice = None):
        """
        Flush the data stream from slice_ argument
        """
        if slice_ is None:
            slice_ = self._stream.get_slice(*self.controller.get_xlim())

        time = self.data.t[slice_]
        n = time.shape[0]

        for i, c in enumerate(self._buffers):
            self._buffers[c].data[-n:, 0] = time.astype("float32")
            self._buffers[c].data[-n:, 1] = (
                self.data.values[slice_, i] * self._manager.data.loc[c]["scale"]
                + self._manager.data.loc[c]["offset"]
            ).astype("float32")
            self._buffers[c].update_full()

    def _get_min_max(self):
        return np.array(
            [
                [self._buffers[c].data[:, 1].min(), self._buffers[c].data[:, 1].max()]
                for c in self._buffers
            ]
        )

    def _rescale(self, event):
        """
        "i" key increase the scale by 50%.
        "d" key decrease the scale by 50%
        """
        if event.type == "key_down":
            if event.key == "i" or event.key == "d":
                factor = {"i": 0.5, "d": -0.5}[event.key]

                # Update the scale of the PlotManager
                self._manager.rescale(factor=factor)

                # Update the current buffers
                for c in self._buffers:
                    self._buffers[c].data[:, 1] = self._buffers[c].data[:, 1] + factor * (
                        self._buffers[c].data[:, 1] - self._manager.data.loc[c]["offset"]
                    )
                    self._buffers[c].update_full()

                self.canvas.request_draw(self.animate)

    def _reset(self, event):
        """
        "r" key reset the plot manager to initial view
        """
        # TODO set the reset for the get controller
        if event.type == "key_down":
            if event.key == "r":
                if isinstance(self.controller, SpanController):
                    self._manager.reset()
                    self._flush()

                minmax = self._get_min_max()
                self.controller.set_ylim(np.min(minmax[:, 0]), np.max(minmax[:, 1]))
                self.canvas.request_draw(self.animate)

    def _update(self, action_name):
        """
        Update function for sort_by and group_by. Because of mode of sort_by, it's not possible
        to just update the buffer.
        """
        # Update the scale only if one action has been performed
        if self._manager._sorted ^ self._manager._grouped:
            self._manager.scale = 1 / np.diff(self._get_min_max(), 1).flatten()

        self._flush()

        # Update camera to fit the full y range
        self.controller.set_ylim(0, np.max(self._manager.offset) + 1)

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
        values = dict(self.data.get_info(metadata_name)) if hasattr(self.data, "get_info") else {}

        # If metadata found
        if len(values):
            # Sorting should happen depending on `groups` and `visible` attributes of _PlotManager
            self._manager.sort_by(values, mode)
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
        values = dict(self.data.get_info(metadata_name)) if hasattr(self.data, "get_info") else {}

        # If metadata found
        if len(values):
            # Grouping positions are computed depending on `order` and `visible` attributes of _PlotManager
            self._manager.group_by(values)
            self._update("group_by")

    def plot_x_vs_y(
        self,
        x_label: Union[str, int, float],
        y_label: Union[str, int, float],
        color: Union[str, tuple] = "white",
        thickness: float = 1.0,
        markersize: float = 10.0,
    ) -> None:
        """
        Plot one metadata column versus another as a line plot.

        Parameters
        ----------
        x_label : str or int or float
            Column name for the x-axis.
        y_label : str or int or float
            Column name for the y-axis.
        color : str or hex or RGB, default="white"
            Line color.
        thickness : float, default=1.0
            Thickness of the connecting line.
        markersize : float, default=10.0
            Size of the time marker.
        """
        # Remove time series line graphics from the scene
        self.scene.remove(*self.graphic.values())

        # Get current time from the center reference line
        current_time = self.ruler_ref_time.geometry.positions.data[0][0]
        self.scene.remove(self.ruler_ref_time)

        # Build new geometry for x-y data
        xy_values = self.data.loc[[x_label, y_label]].values.astype("float32")
        positions = np.zeros((len(self.data), 3), dtype="float32")
        positions[:, 0:2] = xy_values

        # Create new line and add it to the scene
        self.graphic = gfx.Line(
            gfx.Geometry(positions=positions),
            gfx.LineMaterial(thickness=thickness, color=color),
        )
        self.scene.add(self.graphic)

        # Create and add a point marker at the current time
        current_xy = self.data.loc[[x_label, y_label]].get(current_time)
        xy = np.hstack((current_xy, 0), dtype="float32")[None, :]
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
        get_controller.enabled = True
        get_controller._controller_id = controller_id
        get_controller.data = self.data.loc[[x_label, y_label]]
        get_controller.buffer = self.time_point.geometry.positions

        self.controller = get_controller

        # Update camera to fit the full x-y range
        self.controller.set_view(
            xmin=np.min(self.data.loc[x_label]),
            xmax=np.max(self.data.loc[x_label]),
            ymin=np.min(self.data.loc[y_label]),
            ymax=np.max(self.data.loc[y_label]),
        )

        self.canvas.request_draw(self.animate)


class PlotTsGroup(_BasePlot):
    def __init__(self, data: nap.TsGroup, index=None, parent=None):
        super().__init__(data=data, parent=parent)

        # Pynaviz specific controller
        self.controller = SpanController(
            camera=self.camera,
            renderer=self.renderer,
            controller_id=index,
            dict_sync_funcs=dict_sync_funcs,
        )

        self.graphic = {}
        for i, n in enumerate(data.keys()):
            positions = np.stack((data[n].t, np.ones(len(data[n])) * i, np.zeros(len(data[n])))).T
            positions = positions.astype("float32")

            self.graphic[n] = gfx.Points(
                gfx.Geometry(positions=positions),
                gfx.PointsMaterial(
                    size=5,
                    color=GRADED_COLOR_LIST[i % len(GRADED_COLOR_LIST)],
                    opacity=1,
                ),
            )

        self.scene.add(self.ruler_x, self.ruler_y, *list(self.graphic.values()))
        self.canvas.request_draw(self.animate)

    def sort_by(self, metadata_name: str, mode: Optional[str] = "ascending"):
        """
        Sort by metadata entry.

        Parameters
        ----------
        metadata_name : str
            Metadata columns to sort lines
        mode : str, optional
            Options are ["ascending"[default], "descending"]
        """
        # Grabbing the material object
        geometries = get_plot_attribute(self, "geometry")  # Dict index -> geometry

        # Grabbing the metadata
        values = dict(self.data.get_info(metadata_name)) if hasattr(self.data, "get_info") else {}
        # If metadata found
        if len(values):
            values = pd.Series(values)
            idx_sorted = values.sort_values(ascending=(mode == "ascending"))
            idx_map = {idx: i for i, idx in enumerate(idx_sorted.index)}

            for c in geometries:
                geometries[c].positions.data[:, 1] = idx_map[c]
                geometries[c].positions.update_full()

            self.canvas.request_draw(self.animate)

    def group_by(self, metadata_name: str, spacing: Optional = None):
        pass


class PlotTs(_BasePlot):
    def __init__(self, data: nap.Ts, index=None, parent=None):
        super().__init__(parent=parent)
        self.data = data

        # Pynaviz specific controller
        self.controller = SpanController(
            camera=self.camera,
            renderer=self.renderer,
            controller_id=index,
            dict_sync_funcs=dict_sync_funcs,
        )

    def sort_by(self, metadata_name: str, mode: Optional[str] = "ascending"):
        pass

    def group_by(self, metadata_name: str, spacing: Optional = None):
        pass


class PlotBaseVideoTensor(_BasePlot, ABC):
    def __init__(
        self, data: Any, index: Optional[int] = None, parent: Optional[Any] = None
    ) -> None:
        super().__init__(data, parent=parent, maintain_aspect=True)

        # Image texture (subclass provides the texture data)
        texture_data = self._get_initial_texture_data()
        self.texture = gfx.Texture(texture_data.astype("float32"), dim=2)
        self.image = gfx.Image(
            gfx.Geometry(grid=self.texture),
            gfx.ImageBasicMaterial(clim=(0, 1)),
        )

        # Time display
        self.time_text = gfx.Text(
            text="0.0",
            font_size=0.5,
            anchor="bottom-left",
            material=gfx.TextMaterial(
                color="#B4F8C8", outline_color="#000", outline_thickness=0.15
            ),
        )
        self.time_text.geometry.anchor = "bottom-left"

        self.scene.add(self.image, self.time_text)
        self.camera.show_object(self.scene)

        self.controller = GetController(
            camera=self.camera,
            renderer=self.renderer,
            controller_id=index,
            data=self._data,
            buffer=self.texture,
            callback=self._update_buffer,
        )

        self.canvas.request_draw(
            draw_function=lambda: self.renderer.render(self.scene, self.camera)
        )

    @abstractmethod
    def _get_initial_texture_data(self) -> np.ndarray:
        """Subclasses must return a 2D ndarray to initialize the texture."""
        pass

    def _set_time_text(self, frame_index: int):
        if self.time_text:
            self.time_text.set_text(str(self.data.t[frame_index]))

    def set_frame(self, target_time: float):
        """
        Display the video frame closest to the given time.

        Updates the current video display to show the frame corresponding
        to the specified time. This can be used to navigate to a specific
        moment in the video.

        Parameters
        ----------
        target_time : float
            The time (in seconds) to display.
        """
        self.controller.set_frame(target_time)

    def sort_by(self, metadata_name: str, mode: Optional[str] = "ascending"):
        pass

    def group_by(self, metadata_name: str, spacing: Optional = None):
        pass

    @abc.abstractmethod
    def _update_buffer(self, frame_index: int, event_type: Optional[RenderTriggerSource] = None):
        pass


class PlotTsdTensor(PlotBaseVideoTensor):
    def __init__(self, data: nap.TsdTensor, index=None, parent=None):
        self._data = data
        super().__init__(data, index=index, parent=parent)

    def _get_initial_texture_data(self):
        return self._data.values[0]

    def _update_buffer(self, frame_index, event_type: Optional[RenderTriggerSource] = None):
        # this update is fast, do not need async rendering as in PlotVideo
        # event_type is not used here
        _update_buffer(self, frame_index)
        self.controller.renderer_request_draw()


class PlotVideo(PlotBaseVideoTensor):
    _debug = False

    def __init__(
        self,
        video_path: str | pathlib.Path,
        t: Optional[NDArray] = None,
        stream_index: int = 0,
        index=None,
        parent=None,
    ):
        self._closed = False
        data = VideoHandler(video_path, time=t, stream_index=stream_index)
        self._data = data
        super().__init__(data, index=index, parent=parent)

        # Shared memory and comms
        self.shape = self.texture.data.shape
        self.shm_frame = shared_memory.SharedMemory(
            create=True, size=np.prod(self.shape) * np.float32().nbytes
        )
        self.shm_index = shared_memory.SharedMemory(create=True, size=np.float32().nbytes)
        self.shared_frame = np.ndarray(self.shape, dtype=np.float32, buffer=self.shm_frame.buf)
        self.shared_index = np.ndarray(shape=(1,), dtype=np.float32, buffer=self.shm_index.buf)
        self.request_queue = Queue()
        self.response_queue = Queue()
        self.frame_ready = Event()
        self.worker_stop_event = Event()

        # Connect movement event handlers for video
        self.renderer.add_event_handler(self._move_fast, "key_down")

        # Start worker
        self._worker = Process(
            target=video_worker_process,
            args=(
                video_path,
                self.shape,
                self.shm_frame.name,
                self.shm_index.name,
                self.request_queue,
                self.frame_ready,
                self.response_queue,
                self.worker_stop_event,
            ),
            daemon=True,
        )
        self._worker.start()

        # add to registry of active plot video
        # guarantees close at exit.
        _active_plot_videos.add(self)
        self._pending_ui_update_queue = queue.Queue()

        # Start background thread that updates buffers when ready
        # shut thread event
        self._stop_threads = threading.Event()
        self._buffer_thread = threading.Thread(target=self._update_buffer_thread, daemon=True)
        self._buffer_thread.start()

        # for to async/sync lock
        self.buffer_lock = threading.Lock()

        # event requesting a re-draw
        self._needs_redraw = threading.Event()

        # set a pygfx callback for the next draw
        # when the painter draws next it will trigger this
        self.canvas.request_draw(self._render_loop)
        # draw first
        self._last_jump_index = 0

    def _get_initial_texture_data(self):
        # TODO: Get the current time from the controller
        return self._data.get(self._data.time[0])

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, value):
        raise ValueError("Cannot set data for ``PlotVideo``. Data must be a fixed video stream.")

    def close(self):
        if not self._closed:
            try:
                self._data.close()
                # stop update buffer loop
                self._stop_threads.set()
                # stop worker thread
                self.worker_stop_event.set()
                self._worker.join(timeout=2)
                # close shared memory
                self.shm_frame.close()
                self.shm_frame.unlink()
                self.shm_index.close()
                self.shm_index.unlink()
            except Exception:
                pass
            finally:
                # remove form weakref
                _active_plot_videos.discard(self)
                self._closed = True

    def _move_fast(self, event, delta=1):
        """
        "ArrowLeft"/"ArrowRight" key moves between keypoint frames.
        Schedules a move and defers buffer updates to the worker and render loop.
        """
        if event.type == "key_down" and event.key in ("ArrowRight", "ArrowLeft"):
            self.frame_ready.clear()
            while not self.request_queue.empty():
                try:
                    self.request_queue.get_nowait()
                except queue.Empty:
                    break

            # Request: (is_absolute, is_backward, event type)
            self.request_queue.put(
                (False, event.key == "ArrowLeft", RenderTriggerSource.LOCAL_KEY)
            )

            # NOTE: don't wait, don't update buffer or UI here — let the thread handle it

            # Save pre-jump index for sync
            self._last_jump_index = self.controller.frame_index

    def _update_buffer(self, frame_index, event_type: Optional[RenderTriggerSource] = None):
        """Update buffer in response to a sync event."""
        if event_type != RenderTriggerSource.SET_FRAME:
            self.frame_ready.clear()
            while not self.request_queue.empty():
                try:
                    self.request_queue.get_nowait()
                except queue.Empty:
                    break
            event_type = event_type or RenderTriggerSource.UNKNOWN
            self.request_queue.put((frame_index, None, event_type))
            # Track frame index text display which must happen
            # after _update_buffer_thread changes the frame
            # note: the text cannot be set in a thread (since pygfx is not thread
            # safe for these operations) while the buffer can be written safely.
            self._last_requested_frame_index = frame_index
        else:
            # no rendering loop is running, simply set current frame
            # give more priority to request_queue
            frame = self.data[frame_index]
            if isinstance(frame, av.VideoFrame):
                frame = frame.to_ndarray(format="rgb24")[::-1] / 255.0
            with self.buffer_lock:
                self.texture.data[:] = frame

    def _update_buffer_thread(self):
        while not self._stop_threads.is_set():
            # wait until ready then clear notifying the
            # worker sub-process
            if not self.frame_ready.wait(timeout=0.1):
                continue
            # update the buffer (new frame will be displayed)
            with self.buffer_lock:
                self.texture.data[:] = self.shared_frame
            frame_index = int(self.shared_index[0])
            self.frame_ready.clear()
            try:
                trigger_source = self.response_queue.get_nowait()
                self._pending_ui_update_queue.put((frame_index, trigger_source))
            except queue.Empty:
                continue
            # queue the update of the text
            self._needs_redraw.set()  # Ask the main thread to draw

    def __del__(self):
        self.close()

    def _render_loop(self):
        """
        Main render loop scheduled via canvas.request_draw().

        - Polls pending frame updates with trigger metadata.
        - Updates time text and texture in a GUI-safe way.
        - Triggers renderer draw if required.
        - Optionally triggers synchronization if the update was initiated locally.
        """
        update = False
        try:
            # try to get the text label for the frame
            # and update texture if found
            frame_index, trigger_source = self._pending_ui_update_queue.get_nowait()

            # print(trigger_source, self.controller.controller_id)
            self._set_time_text(frame_index)
            with self.buffer_lock:
                self.texture.update_full()
            self.controller.frame_index = frame_index
            self.controller.renderer_request_draw()

            # Sending the sync event
            if trigger_source == RenderTriggerSource.LOCAL_KEY and hasattr(
                self, "_last_jump_index"
            ):
                current_time = self._data.t[frame_index]
                if self._debug:
                    print("keypress", current_time, frame_index)
                del self._last_jump_index  # prevent repeat sync
                self.controller._send_sync_event(update_type="pan", current_time=current_time)

            elif trigger_source == RenderTriggerSource.ZOOM_TO_POINT:
                current_time = self.controller._get_current_time()
                if self._debug:
                    print("zoom", current_time, frame_index)
                self.controller._send_sync_event(update_type="pan", current_time=current_time)

        except queue.Empty:
            update = True

        # redraw in case text is found
        if self._needs_redraw.is_set() or update:
            self.renderer.render(self.scene, self.camera)
            self._needs_redraw.clear()

        # set the callback for the next draw
        self.canvas.request_draw(self._render_loop)


def _update_buffer(plot_object: PlotTsdTensor | PlotTsdFrame, frame_index: int):
    if (
        plot_object.texture.data.shape[0] == 1 and plot_object.texture.data.shape[1] == 3
    ):  # assume single point
        plot_object.texture.data[0, 0:2] = plot_object.data.values[frame_index].astype("float32")
    else:
        img_array = plot_object.data.values[frame_index]
        plot_object.texture.data[:] = img_array.astype("float32")
    plot_object.texture.update_full()
    plot_object._set_time_text(frame_index)
    return
