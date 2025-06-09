"""
Simple plotting class for each pynapple object.
Create a unique canvas/renderer for each class
"""

import threading
import warnings
from typing import Any, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pygfx as gfx
import pynapple as nap
from matplotlib.colors import Colormap
from matplotlib.pyplot import colormaps
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
        if parent: # Assuming it's a Qt background
            self.canvas = WgpuCanvas(parent=parent)
        else: # Default to glfw for single canvas
            self.canvas = WgpuCanvas()

        # Create a WGPU-based renderer attached to the canvas
        self.renderer = gfx.WgpuRenderer(self.canvas)  ## 97% time of super.__init__(...) when running `large_nwb_main.py`

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
            index=data.index
        elif isinstance(data, nap.TsdFrame):
            index=data.columns
        else:
            index=[]
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

    def color_by(self,
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

    def sort_by(self, metadata_name: str, mode: Optional[str] = "ascending"):
        pass

    def group_by(self, metadata_name: str):
        pass


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

    def __init__(self, data: nap.Tsd, index: Optional[int] = None, parent: Optional[Any] = None) -> None:
        super().__init__(data=data, parent=parent)

        # Create a controller for span-based interaction, syncing, and user inputs
        self.controller = SpanController(camera=self.camera, renderer=self.renderer, controller_id=index,
                                         dict_sync_funcs=dict_sync_funcs, plot_callbacks=[])

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

    def __init__(self, data: nap.TsdFrame, index: Optional[int] = None, parent: Optional[Any] = None):
        super().__init__(data=data, parent=parent)
        self.data = data

        # Initialize lines for each column in the TsdFrame
        self.graphic: dict[str, gfx.Line] = {}

        # To stream data
        self._stream = TsdFrameStreaming(data, callback=self._flush, window_size = 3) # seconds

        # Create pygfx objects
        for i, c in enumerate(self.data.columns):
            positions = np.zeros(
                (len(self._stream), 3), dtype="float32"
            )
            self.graphic[c] = gfx.Line(
                gfx.Geometry(positions=positions),
                gfx.LineMaterial(thickness=1.0, color=GRADED_COLOR_LIST[i % len(GRADED_COLOR_LIST)]),
            )

        # Stream the first batch of data
        self._buffers = {c: self.graphic[c].geometry.positions for c in self.graphic}
        self._flush(self._stream.get_slice(start=0, end=1))

        # Add elements to the scene for rendering
        self.scene.add(
            self.ruler_x, self.ruler_y, self.ruler_ref_time, *self.graphic.values()
        )

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
                plot_callbacks=[self._stream.stream]
            ),
            "get": GetController(
                camera=self.camera,
                renderer=self.renderer,
                data=None,
                buffer=None,
                enabled=False,
            ),
        }

        # Use span controller by default
        self.controller = self._controllers["span"]

        # Used later in x-vs-y plotting
        self.time_point = None

        # By default, showing only the first second.
        minmax = self._get_min_max()
        self.controller.set_view(0, 1, np.min(minmax[:,0]), np.max(minmax[:,1]))

        # Request an initial draw of the scene
        self.canvas.request_draw(self.animate)

    def _flush(self, slice_: slice = None):
        """
        Flush the data stream from slice_ argument
        """
        if slice_ is None:
            slice_ = self._stream.get_slice(*self.controller.get_xlim())

        time = self.data.t[slice_]
        n = time.shape[0]

        for i, c in enumerate(self._buffers):
            self._buffers[c].data[-n:,0] = time.astype("float32")
            self._buffers[c].data[-n:,1] = (
                self.data.values[slice_, i] * self._manager.data.loc[c]['scale'] + self._manager.data.loc[c]['offset']
            ).astype("float32")
            self._buffers[c].update_full()

    def _get_min_max(self):
        return np.array([[
                self._buffers[c].data[:,1].min(),
                self._buffers[c].data[:,1].max()
                ] for c in self._buffers])

    def _rescale(self, event):
        """
        "i" key increase the scale by 50%.
        "d" key decrease the scale by 50%
        """
        if event.type == "key_down":
            if event.key == "i" or event.key == "d":
                factor = {"i":0.5, "d":-0.5}[event.key]

                # Update the scale of the PlotManager
                self._manager.rescale(factor=factor)

                # Update the current buffers
                for c in self._buffers:
                    self._buffers[c].data[:, 1] = self._buffers[c].data[:, 1] + factor * ( self._buffers[c].data[:, 1] - self._manager.data.loc[c]['offset'])
                    self._buffers[c].update_full()

                self.canvas.request_draw(self.animate)

    def _reset(self, event):
        """
        "r" key reset the plot manager to initial view
        """
        #TODO set the reset for the get controller
        if event.type == "key_down":
            if event.key == "r":
                if isinstance(self.controller, SpanController):
                    self._manager.reset()
                    self._flush()

                minmax = self._get_min_max()
                self.controller.set_ylim(np.min(minmax[:,0]), np.max(minmax[:,1]))
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
        values = (
            dict(self.data.get_info(metadata_name))
            if hasattr(self.data, "get_info")
            else {}
        )

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
        values = (
            dict(self.data.get_info(metadata_name))
            if hasattr(self.data, "get_info")
            else {}
        )

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
        self.controller = SpanController(camera=self.camera, renderer=self.renderer, controller_id=index,
                                         dict_sync_funcs=dict_sync_funcs, min=0, max=len(data) + 1)

        self.graphic = {}
        for i, n in enumerate(data.keys()):
            positions = np.stack(
                (data[n].t, np.ones(len(data[n])) * i, np.zeros(len(data[n])))
            ).T
            positions = positions.astype("float32")

            self.graphic[n] = gfx.Points(
                gfx.Geometry(positions=positions),
                gfx.PointsMaterial(size=5, color=GRADED_COLOR_LIST[i % len(GRADED_COLOR_LIST)], opacity=1),
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
        geometries = get_plot_attribute(self, "geometry") # Dict index -> geometry

        # Grabbing the metadata
        values = (
            dict(self.data.get_info(metadata_name))
            if hasattr(self.data, "get_info")
            else {}
        )
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

class PlotTsdTensor(_BasePlot):
    def __init__(self, data: nap.TsdTensor, index=None, parent=None):
        super().__init__(data, parent=parent, maintain_aspect=True)

        # Image
        texture = gfx.Texture(self.data.values[0].astype("float32"), dim=2)
        self.image = gfx.Image(
            gfx.Geometry(grid=texture),
            gfx.ImageBasicMaterial(clim=(0, 1)),
        )
        # Text of the current time
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
        self.camera.show_object(self.scene)  # , view_dir=(0, 0, -1))

        # # Pynaviz specific controller
        self.controller = GetController(
            camera=self.camera,
            renderer=self.renderer,
            controller_id=index,
            data=data,
            buffer=texture,
            time_text=self.time_text,
        )

        # In the draw event, the draw function is called
        # This assign the function draw_frame of WgpuCanvasBase
        self.canvas.request_draw(
            draw_function=lambda: self.renderer.render(self.scene, self.camera)
        )

    def sort_by(self, metadata_name: str, mode: Optional[str] = "ascending"):
        pass

    def group_by(self, metadata_name: str, spacing: Optional = None):
        pass

class PlotTs(_BasePlot):
    def __init__(self, data: nap.Ts, index=None, parent=None):
        super().__init__(parent=parent)
        self.data = data

        # Pynaviz specific controller
        self.controller = SpanController(camera=self.camera, renderer=self.renderer, controller_id=index,
                                         dict_sync_funcs=dict_sync_funcs)

    def sort_by(self, metadata_name: str, mode: Optional[str] = "ascending"):
        pass

    def group_by(self, metadata_name: str, spacing: Optional = None):
        pass
