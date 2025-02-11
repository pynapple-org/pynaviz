"""
Simple plotting class for each pynapple object.
Create a unique canvas/renderer for each class
"""

import pynapple as nap
from wgpu.gui.auto import WgpuCanvas, run
import pygfx as gfx
from abc import ABC, abstractmethod
from pylinalg import vec_transform, vec_unproject
import numpy as np

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
    def __init__(self):
        self.canvas = WgpuCanvas()
        self.renderer = gfx.WgpuRenderer(self.canvas)
        self.scene = gfx.Scene()
        self.rulerx = gfx.Ruler(tick_side="right")
        self.rulery = gfx.Ruler(tick_side="left", min_tick_distance=40)
        self.camera = gfx.OrthographicCamera(maintain_aspect=False)
        self.camera.show_rect(-100, 1100, -5, 5)
        self.controller = gfx.PanZoomController(self.camera, register_events=self.renderer)


    def animate(self):
        renderer = self.renderer
        canvas = self.canvas
        camera = self.camera
        scene = self.scene
        rulerx = self.rulerx
        rulery = self.rulery
        # get range of screen space
        xmin, ymin = 0, renderer.logical_size[1]
        xmax, ymax = renderer.logical_size[0], 0

        world_xmin, world_ymin, _ = map_screen_to_world(camera,(xmin, ymin), renderer.logical_size)
        world_xmax, world_ymax, _ = map_screen_to_world(camera,(xmax, ymax), renderer.logical_size)

        # set start and end positions of rulers
        rulerx.start_pos = world_xmin, 0, -1000
        rulerx.end_pos = world_xmax, 0, -1000

        rulerx.start_value = rulerx.start_pos[0]

        statsx = rulerx.update(camera, canvas.get_logical_size())

        rulery.start_pos = 0, world_ymin, -1000
        rulery.end_pos = 0, world_ymax, -1000

        rulery.start_value = rulery.start_pos[1]
        statsy = rulery.update(camera, canvas.get_logical_size())

        # major_step_x, major_step_y = statsx["tick_step"], statsy["tick_step"]
        # grid.material.major_step = major_step_x, major_step_y
        # grid.material.minor_step = 0.2 * major_step_x, 0.2 * major_step_y

        # print(statsx)

        renderer.render(scene, camera)

class PlotTsd(_BasePlot):
    def __init__(self, data: nap.Tsd):
        super().__init__()
        self.data = data

        positions = np.stack((data.t, data.d, np.zeros_like(data))).T
        positions = positions.astype('float32')

        self.line = gfx.Line(
            gfx.Geometry(positions=positions),
            gfx.LineMaterial(thickness=4.0, color="#aaf"),
        )
        self.scene.add(self.rulerx, self.rulery, self.line)
        self.canvas.request_draw(self.animate)
        run()


class PlotTsdFrame(_BasePlot):
    def __init__(self, data: nap.TsdFrame):
        super().__init__()
        self.data = data

class PlotTsdTensor(_BasePlot):
    def __init__(self, data:nap.TsdTensor):
        super().__init__()
        self.data = data

class PlotTs(_BasePlot):
    def __init__(self, data: nap.Ts):
        super().__init__()
        self.data = data

class PlotTsGroup(_BasePlot):
    def __init__(self, data: nap.TsGroup):
        super().__init__()
        self.data = data