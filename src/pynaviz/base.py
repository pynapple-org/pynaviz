"""
    The base class that controls everything.
"""

import fastplotlib as fpl
import ipdb
import numpy as np
import pygfx as gfx
from fastplotlib.layouts._utils import make_canvas_and_renderer

from .controller import ControllerGroup, PynaVizController
from .synchronization_rules import _match_pan_on_x_axis, _match_zoom_on_x_axis


class Base:

    def __init__(self, *args, dict_sync_funcs=None):
        self._args = args
        self._controllers = {}
        self._views = {}
        self._cameras = {}
        self._renderers = {}
        if dict_sync_funcs is None:
            dict_sync_funcs = dict(
                pan=_match_pan_on_x_axis,
                zoom=_match_zoom_on_x_axis,
                zoom_to_point=_match_zoom_on_x_axis,
            )

        for i, obj in enumerate(args):

            # Instantiate canvas and renderer
            canvas, renderer = make_canvas_and_renderer(None, None)

            # Instantiate the camera
            camera = gfx.OrthographicCamera(500, 400, maintain_aspect=False)

            # # Instantiate the controler
            ctrl = PynaVizController(
                camera=camera,
                register_events=renderer,
                controller_id=i,
                dict_sync_funcs=dict_sync_funcs,
            )

            # instantiate fastplotlib
            fig = fpl.Figure(
                cameras=camera, canvas=canvas, renderer=renderer, controllers=ctrl
            )

            tmp = np.vstack((obj.index.values, obj.values)).T
            tmp = tmp.astype(np.float32)
            fig[0, 0].add_line(data=tmp)
            fig.show(maintain_aspect=False)
            # ipdb.set_trace()

            self._views[i] = fig
            self._cameras[i] = fig.cameras
            self._controllers[i] = ctrl
            self._renderers[i] = renderer

        controllers_and_renderers = [
            (self._controllers[i], self._renderers[i]) for i in self._controllers.keys()
        ]
        self._ctrl_group = ControllerGroup(*controllers_and_renderers)

        fpl.run()
