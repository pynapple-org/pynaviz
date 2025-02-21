"""
Simple plotting class for each pynapple object.
Create a unique canvas/renderer for each class
"""

import pynapple as nap
from PyQt6.QtWidgets import QWidget
from wgpu.gui.qt import (
    WgpuCanvas,
)  # Should use auto here or be able to select qt if parent passed
import pygfx as gfx
from abc import ABC
from pylinalg import vec_transform, vec_unproject
import numpy as np
from .controller import PynaVizController
from .synchronization_rules import _match_pan_on_x_axis, _match_zoom_on_x_axis


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
    def __init__(self, index=0, start=0, end=100, parent=None):
        self.canvas = WgpuCanvas(parent=parent)
        self.renderer = gfx.WgpuRenderer(self.canvas)
        self.scene = gfx.Scene()
        self.rulerx = gfx.Ruler(tick_side="right")
        self.rulery = gfx.Ruler(tick_side="left", min_tick_distance=40)
        self.ruler_ref_time = self._get_ruler_ref_time(start=start, end=end)
        self.camera = gfx.OrthographicCamera(maintain_aspect=False)
        self.camera.show_rect(  # Uses world coordinates
            left=start, right=end, top=-5, bottom=5
        )
        self.controller = PynaVizController(
            camera=self.camera,
            renderer=self.renderer,
            controller_id=index,
            dict_sync_funcs=dict_sync_funcs
        )

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
        self.ruler_ref_time.geometry.positions.data[:,0] = world_xmin+(world_xmax-world_xmin)/2
        self.ruler_ref_time.geometry.positions.data[:,1] = np.array([world_ymin-10, world_ymax+10])
        self.ruler_ref_time.geometry.positions.update_full()
        
        self.renderer.render(self.scene, self.camera)

    def _get_ruler_ref_time(self, start, end):
        """set the center ruler between start and end
        as a gfx.Line
        """
        c = start + (end-start)/2
        positions = [[c, -5, 0], [c, 5, 0]]
        return gfx.Line(
            gfx.Geometry(positions=positions),
            gfx.LineMaterial(thickness=0.5, color="#aaf"),
            )
        
class PlotTsd(_BasePlot):
    def __init__(self, data: nap.Tsd, index=0, parent=None):
        super().__init__(index=index, parent=parent)
        self.data = data

        positions = np.stack((data.t, data.d, np.zeros_like(data))).T
        positions = positions.astype("float32")

        self.line = gfx.Line(
            gfx.Geometry(positions=positions),
            gfx.LineMaterial(thickness=4.0, color="#aaf"),
        )
        self.scene.add(self.rulerx, self.rulery, self.ruler_ref_time, self.line)
        self.canvas.request_draw(self.animate)


class PlotTsdFrame(_BasePlot):
    def __init__(self, data: nap.TsdFrame, index=0, parent=None):
        super().__init__(index=index, parent=parent)
        self.data = data
        self.lines = []

        for i in range(self.data.shape[1]):
            positions = np.stack((data.t, data.d[:, i], np.zeros(data.shape[0]))).T
            positions = positions.astype("float32")
            self.lines.append(
                gfx.Line(
                    gfx.Geometry(positions=positions),
                    gfx.LineMaterial(thickness=4.0, color=COLORS[i % len(COLORS)]),
                )
            )
        self.scene.add(self.rulerx, self.rulery, *self.lines)
        self.canvas.request_draw(self.animate)


class PlotTsGroup(_BasePlot):
    def __init__(self, data: nap.TsGroup, index=0, parent=None):
        super().__init__(index=index, parent=parent)
        self.data = data
        self.raster = []

        for i, n in enumerate(data.keys()):
            positions = np.stack(
                (data[n].t, np.ones(len(data[n])) * i, np.zeros(len(data[n])))
            ).T
            positions = positions.astype("float32")

            self.raster.append(
                gfx.Points(
                    gfx.Geometry(positions=positions),
                    gfx.PointsMaterial(
                        size=5, color=COLORS[i % len(COLORS)], opacity=0.5
                    ),
                )
            )

        self.scene.add(self.rulerx, self.rulery, *self.raster)
        self.canvas.request_draw(self.animate)


class PlotTsdTensor(_BasePlot):
    def __init__(self, data: nap.TsdTensor, index=0, parent=None):
        super().__init__(index=index, parent=parent)
        self.data = data

        image = gfx.Image(
            gfx.Geometry(
                grid=gfx.Texture(self.data.values[0].astype("float32"), dim=2)
            ),
            gfx.ImageBasicMaterial(clim=(0, 1)),
        )
        self.scene.add(image)
        self.camera = gfx.OrthographicCamera(self.data.shape[0], self.data.shape[1])
        self.camera.show_object(self.scene, view_dir=(0, 0, -1))
        self.camera.local.scale_y = -1

        self.canvas.request_draw(lambda: self.renderer.render(self.scene, self.camera))


class PlotTs(_BasePlot):
    def __init__(self, data: nap.Ts, index=0, parent=None):
        super().__init__(index=index, parent=parent)
        self.data = data
