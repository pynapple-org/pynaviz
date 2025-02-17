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
    def __init__(self, index=0):
        self.canvas = WgpuCanvas()
        self.renderer = gfx.WgpuRenderer(self.canvas)
        self.scene = gfx.Scene()
        self.rulerx = gfx.Ruler(tick_side="right")
        self.rulery = gfx.Ruler(tick_side="left", min_tick_distance=40)
        self.camera = gfx.OrthographicCamera(maintain_aspect=False)
        self.camera.show_rect(-100, 1100, -5, 5)
        self.controller = PynaVizController(
            camera=self.camera,
            register_events=self.renderer,
            controller_id=index,
            dict_sync_funcs=dict_sync_funcs,
        )

    def animate(self):
        # get range of screen space
        xmin, ymin = 0, self.renderer.logical_size[1]
        xmax, ymax = self.renderer.logical_size[0], 0

        world_xmin, world_ymin, _ = map_screen_to_world(
            self.camera, (xmin, ymin), self.renderer.logical_size
        )
        world_xmax, world_ymax, _ = map_screen_to_world(
            self.camera, (xmax, ymax), self.renderer.logical_size
        )

        # set start and end positions of rulers
        self.rulerx.start_pos = world_xmin, 0, -1000
        self.rulerx.end_pos = world_xmax, 0, -1000

        self.rulerx.start_value = self.rulerx.start_pos[0]

        statsx = self.rulerx.update(self.camera, self.canvas.get_logical_size())

        self.rulery.start_pos = 0, world_ymin, -1000
        self.rulery.end_pos = 0, world_ymax, -1000

        self.rulery.start_value = self.rulery.start_pos[1]
        statsy = self.rulery.update(self.camera, self.canvas.get_logical_size())
        # major_step_x, major_step_y = statsx["tick_step"], statsy["tick_step"]
        # grid.material.major_step = major_step_x, major_step_y
        # grid.material.minor_step = 0.2 * major_step_x, 0.2 * major_step_y
        # print(statsx)

        self.renderer.render(self.scene, self.camera)


class PlotTsd(_BasePlot):
    def __init__(self, data: nap.Tsd, index=0):
        super().__init__(index=index)
        self.data = data

        positions = np.stack((data.t, data.d, np.zeros_like(data))).T
        positions = positions.astype("float32")

        self.line = gfx.Line(
            gfx.Geometry(positions=positions),
            gfx.LineMaterial(thickness=4.0, color="#aaf"),
        )
        self.scene.add(self.rulerx, self.rulery, self.line)
        self.canvas.request_draw(self.animate)


class PlotTsdFrame(_BasePlot):
    def __init__(self, data: nap.TsdFrame, index=0):
        super().__init__(index=index)
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
    def __init__(self, data: nap.TsGroup, index=0):
        super().__init__(index=index)
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
    def __init__(self, data: nap.TsdTensor, index=0):
        super().__init__(index=index)
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
    def __init__(self, data: nap.Ts, index=0):
        super().__init__(index=index)
        self.data = data
