"""
Simple plotting class for each pynapple object.
Create a unique canvas/renderer for each class
"""

import threading
import warnings
from abc import ABC
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pygfx as gfx
import pynapple as nap
import PyQt6
from matplotlib.colors import Colormap
from matplotlib.pyplot import colormaps
from pylinalg import vec_transform, vec_unproject
from wgpu.gui.qt import (
    WgpuCanvas,  # Should use auto here or be able to select qt if parent passed
)

from .controller import GetController, SpanController
from .synchronization_rules import _match_pan_on_x_axis, _match_zoom_on_x_axis

from .threads.metadata_to_color_maps import MetadataMappingThread
from .utils import get_plot_attribute, trim_kwargs
import fastplotlib as fpl


COLORS = [
    "hotpink",
    "lightpink",
    "cyan",
    "orange",
    "lightcoral",
    "lightsteelblue",
    "lime",
    "lightgreen",
    "magenta",
    "pink",
    "aliceblue",
]

dict_sync_funcs = {
    "pan": _match_pan_on_x_axis,
    "zoom": _match_zoom_on_x_axis,
    "zoom_to_point": _match_zoom_on_x_axis,
}


def map_screen_to_world(camera, pos, viewport_size):
    # first convert position to NDC
    x = pos[0] / viewport_size[0] * 2 - 1
    y = -(pos[1] / viewport_size[1] * 2 - 1)
    pos_ndc = (x, y, 0)
    pos_ndc += vec_transform(camera.world.position, camera.camera_matrix)
    # unproject to world space
    pos_world = vec_unproject(pos_ndc[:2], camera.camera_matrix)
    return pos_world


class _BasePlot(ABC):
    def __init__(self, data, parent=None, maintain_aspect=False):
        self.canvas = WgpuCanvas(parent=parent)
        self._data = data
        self.color_mapping_thread = MetadataMappingThread(data)
        self.renderer = gfx.WgpuRenderer(self.canvas)
        self.scene = gfx.Scene()
        self.rulerx = gfx.Ruler(tick_side="right")
        self.rulery = gfx.Ruler(tick_side="left", min_tick_distance=40)
        self.ruler_ref_time = gfx.Line(
            gfx.Geometry(positions=[[0, 0, 0], [0, 0, 0]]),
            gfx.LineMaterial(thickness=0.5, color="#B4F8C8"),
        )
        self.camera = gfx.OrthographicCamera(maintain_aspect=maintain_aspect)
        self._cmap = "jet"

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
            )
            return
        if not value in plt.colormaps():
            warnings.warn(
                message=f"Invalid colormap {value}. 'cmap' must be matplotlib 'Colormap'.",
                category=UserWarning,
            )
            return
        self._cmap = value

    def animate(self):
        # get range of screen space in pixels
        xmin, ymin = 0, self.renderer.logical_size[1]
        xmax, ymax = self.renderer.logical_size[0], 0

        # Given the camera position and the range of screen space, convert to world space.
        # Get the bottom corner and top corner
        world_xmin, world_ymin, _ = map_screen_to_world(
            self.camera, pos=(xmin, ymin), viewport_size=self.renderer.logical_size
        )
        world_xmax, world_ymax, _ = map_screen_to_world(
            self.camera, pos=(xmax, ymax), viewport_size=self.renderer.logical_size
        )

        # X axis
        self.rulerx.start_pos = world_xmin, 0, -1000
        self.rulerx.end_pos = world_xmax, 0, -1000
        self.rulerx.start_value = self.rulerx.start_pos[0]
        self.rulerx.update(self.camera, self.canvas.get_logical_size())

        # Y axis
        self.rulery.start_pos = 0, world_ymin, -1000
        self.rulery.end_pos = 0, world_ymax, -1000
        self.rulery.start_value = self.rulery.start_pos[1]
        self.rulery.update(self.camera, self.canvas.get_logical_size())

        # Center time Ref axis
        self.ruler_ref_time.geometry.positions.data[:, 0] = (
            world_xmin + (world_xmax - world_xmin) / 2
        )
        self.ruler_ref_time.geometry.positions.data[:, 1] = np.array(
            [world_ymin - 10, world_ymax + 10]
        )
        self.ruler_ref_time.geometry.positions.update_full()

        self.renderer.render(self.scene, self.camera)

    def color_by(self, metadata_name, cmap_name="viridis", vmin=0.0, vmax=100.0):
        # if still computing color maps, set a 25ms timer and try again
        if self.color_mapping_thread.is_running():
            slot = lambda: self.color_by(
                metadata_name, cmap_name=cmap_name, vmin=vmin, vmax=vmax
            )
            threading.Timer(0.025, slot).start()
            return

        try:
            self.cmap = cmap_name
        except:
            self.cmap = "jet"

        map_to_colors = self.color_mapping_thread.color_maps.get(metadata_name, None)

        if map_to_colors is None:
            warnings.warn(
                message=f"Cannot find appropriate color mapping for {metadata_name} metadata.",
                category=UserWarning,
            )

        map_kwargs = trim_kwargs(
            map_to_colors, dict(cmap=colormaps[self.cmap], vmin=vmin, vmax=vmax)
        )

        # Grabbing the material object
        materials = get_plot_attribute(self, "material")

        # Grabbing the metadata
        values = (
            self.data.get_info(metadata_name) if hasattr(self.data, "get_info") else {}
        )

        # If metadata found
        if len(values):
            map_color = map_to_colors(values, **map_kwargs)
            if map_color:
                for c in materials:
                    materials[c].color = map_color[values[c]]
                self.canvas.request_draw(self.animate)  # To fix

    def sort_by(self, metadata_name: str, order: Optional[str] = "ascending"):
        """

        Parameters
        ----------
        metadata_name : str
            Metadata columns to sort lines
        order
        """
        # Grabbing the material object
        geometries = get_plot_attribute(self, "geometry")

        # Grabbing the metadata
        values = (
            dict(self.data.get_info(metadata_name))
            if hasattr(self.data, "get_info")
            else {}
        )

        # If metadata found
        if len(values):
            values = pd.Series(values)
            idx_sorted = values.sort_values(ascending=(order == "ascending"))
            idx_map = {idx: i for i, idx in enumerate(idx_sorted.index)}

            for c in geometries:
                geometries[c].positions.data[:, 1] = idx_map[c]
                geometries[c].positions.update_full()

            self.canvas.request_draw(self.animate)

    def update(self, event):
        """
        Apply an action to the widget plot.
        Actions can be "color_by", "sort_by" and "group_by"
        """
        metadata_name = event["metadata_name"]
        action_name = event["action"]
        if action_name == "color_by":
            self.color_by(metadata_name)
        elif action_name == "sort_by":
            self.sort_by(metadata_name)

        metadata = (
            dict(self.data.get_info(metadata_name))
            if hasattr(self.data, "get_info")
            else {}
        )
        # action_caller(self, action, metadata=metadata, **kwargs)
        # TODO: make it more targeted than update all

    def set_metadata_maps(self):
        if not hasattr(self.data, "metadata"):
            return


class PlotTsd(_BasePlot):
    def __init__(self, data: nap.Tsd, index=None, parent=None):
        super().__init__(data=data, parent=parent)

        # Pynaviz specific controller
        self.controller = SpanController(
            camera=self.camera,
            renderer=self.renderer,
            controller_id=index,
            dict_sync_funcs=dict_sync_funcs,
            min=np.min(data),
            max=np.max(data),
        )

        # Passing the data
        positions = np.stack((data.t, data.d, np.zeros_like(data))).T
        positions = positions.astype("float32")

        self.line = gfx.Line(
            gfx.Geometry(positions=positions),
            gfx.LineMaterial(thickness=4.0, color="#aaf"),
        )

        self.scene.add(self.rulerx, self.rulery, self.ruler_ref_time, self.line)
        self.canvas.request_draw(self.animate)
        # self.controller.show_interval(start=0, end=1)


class PlotTsdFrame(_BasePlot):
    def __init__(self, data: nap.TsdFrame, index=None, parent=None):
        super().__init__(data=data, parent=parent)

        # Possible controllers for TsdFrame
        self._controllers = {
            "span": SpanController(
                camera=self.camera,
                renderer=self.renderer,
                controller_id=index,
                dict_sync_funcs=dict_sync_funcs,
                min=np.min(data),
                max=np.max(data),
            ),
            "get": GetController(
                camera=self.camera,
                renderer=self.renderer,
                data=None,
                buffer=None,
                enabled=False,
            ),
        }

        # First controller
        self.controller = self._controllers["span"]

        # Passing the data
        self.graphic: dict = {}
        for i, c in enumerate(self.data.columns):
            positions = np.stack((data.t, data.d[:, i], np.zeros(data.shape[0]))).T
            positions = positions.astype("float32")
            self.graphic[c] = gfx.Line(
                gfx.Geometry(positions=positions),
                gfx.LineMaterial(thickness=4.0, color=COLORS[i % len(COLORS)]),
            )

        # Extra init depending on the context
        self.time_point = None

        self.scene.add(
            self.rulerx, self.rulery, self.ruler_ref_time, *list(self.graphic.values())
        )
        self.canvas.request_draw(self.animate)

    def plot_x_vs_y(self, x_label, y_label, color="white", thickness=1, markersize=10):
        """
        Plot one column vs the other. The x-axis can be selected with the `x_label`
        and the y-axis can be selected with the `y_label` argument.

        Parameters
        ----------
        x_label : string or int or float
            Column name for the x-axis
        y_label : string or int or float
            Column name for the y-axis
        color : string or hex or rgb
            Color of the line.
        thickness : Number
            Thickness of the line
        markersize : Number
            Size of the marker
        """
        # Removing objects
        self.scene.remove(*list(self.graphic.values()))
        self.scene.remove(self.ruler_ref_time)

        # Adding line object
        positions = np.zeros((len(self.data), 3), dtype="float32")
        positions[:, 0:2] = self.data.loc[[x_label, y_label]].values.astype("float32")

        self.graphic = gfx.Line(
            gfx.Geometry(positions=positions),
            gfx.LineMaterial(thickness=thickness, color=color),
        )
        self.scene.add(self.graphic)

        # Adding point object to track time
        xy = np.zeros((1, 3), dtype="float32")
        self.time_point = gfx.Points(
            gfx.Geometry(positions=xy),
            gfx.PointsMaterial(size=30, color="red", opacity=1),
        )
        self.scene.add(self.time_point)

        # Disable old controller
        self.controller.enabled = False

        # Instantiating new controller
        self.controller = self._controllers["get"]
        self.controller.n_frames = len(self.data)
        self.controller.frame_index = 0
        self.controller.enabled = True
        self.controller.data = self.data.loc[[x_label, y_label]]
        self.controller.buffer = self.time_point.geometry.positions

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
            min=0,
            max=len(data) + 1,
        )

        self.graphic = {}
        for i, n in enumerate(data.keys()):
            positions = np.stack(
                (data[n].t, np.ones(len(data[n])) * i, np.zeros(len(data[n])))
            ).T
            positions = positions.astype("float32")

            self.graphic[n] = gfx.Points(
                gfx.Geometry(positions=positions),
                gfx.PointsMaterial(size=5, color=COLORS[i % len(COLORS)], opacity=1),
            )

        self.scene.add(self.rulerx, self.rulery, *list(self.graphic.values()))
        self.canvas.request_draw(self.animate)


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
