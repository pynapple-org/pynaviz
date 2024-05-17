import numpy as np
import pygfx
from pygfx import controllers, cameras, renderers
from pynaviz.controller import PynaVizController
from wgpu.gui.auto import WgpuCanvas
import pytest
from contextlib import nullcontext as does_not_raise
from typing import Callable


def test_controller_state_dict():
    """Test that the pygfx state dictionary API is maintained."""
    ctrl = controllers.PanZoomController(camera=cameras.PerspectiveCamera())
    cam_state = ctrl._get_camera_state()
    assert tuple(cam_state.keys()) == ('position', 'rotation', 'scale', 'reference_up', 'fov', 'width', 'height', 'zoom', 'maintain_aspect', 'depth_range')
    assert isinstance(cam_state["position"], np.ndarray)
    assert cam_state["position"].ndim == 1
    assert cam_state["position"].shape[0] == 3

    assert isinstance(cam_state["rotation"], np.ndarray)
    assert cam_state["rotation"].ndim == 1
    assert cam_state["rotation"].shape[0] == 4

    assert isinstance(cam_state["scale"], np.ndarray)
    assert cam_state["scale"].ndim == 1
    assert cam_state["scale"].shape[0] == 3

    assert isinstance(cam_state["reference_up"], np.ndarray)
    assert cam_state["reference_up"].ndim == 1
    assert cam_state["reference_up"].shape[0] == 3

    assert isinstance(cam_state["fov"], float)
    assert isinstance(cam_state["height"], float)
    assert isinstance(cam_state["width"], float)
    assert isinstance(cam_state["zoom"], float)
    assert isinstance(cam_state["maintain_aspect"], bool)


class TestPynaVizController:

    @pytest.mark.parametrize(
        "ctrl_id, expectation",
        [
            (0, does_not_raise()),
            (None, does_not_raise()),
            ("id", pytest.raises(TypeError, match="f provided, `controller_id` must"))
        ]
    )
    def test_init_controller_id(self, ctrl_id, expectation):
        camera = pygfx.OrthographicCamera()
        canvas = WgpuCanvas()
        renderer = renderers.WgpuRenderer(canvas)
        try:
            with expectation:
                ctrl = PynaVizController(camera, register_events=renderer, controller_id=ctrl_id)
                assert ctrl.controller_id == ctrl_id
        finally:
            canvas.close()

    @pytest.mark.parametrize(
        "dict_sync, expectation",
        [
            (None, does_not_raise()),
            (dict(), does_not_raise()),
            (dict(abc=lambda x:x), does_not_raise()),
            ("not a dict", pytest.raises(TypeError, match="When provided, `dic")),
            (dict(abc="not a callable"), pytest.raises(TypeError, match="`dict_sync_funcs` items must be of")),
        ]
    )
    def test_init_sync_func_dict(self, dict_sync, expectation):
        camera = pygfx.OrthographicCamera()
        canvas = WgpuCanvas()
        renderer = renderers.WgpuRenderer(canvas)
        try:
            with expectation:
                PynaVizController(camera, register_events=renderer, dict_sync_funcs=dict_sync)
        finally:
            canvas.close()

    def test_control_id_setter(self):
        camera = pygfx.OrthographicCamera()
        canvas = WgpuCanvas()
        renderer = renderers.WgpuRenderer(canvas)
        try:
            ctrl = PynaVizController(camera, register_events=renderer, controller_id=None)
            ctrl.controller_id = 1
            with pytest.raises(ValueError, match="Controller id can be set only once"):
                ctrl.controller_id = 1
        finally:
            canvas.close()

    def test_get_event_handle(self):
        canvas = WgpuCanvas()
        renderer = renderers.WgpuRenderer(canvas)
        try:
            func = PynaVizController._get_event_handle(renderer)
            assert isinstance(func, Callable)
        finally:
            canvas.close()

    @pytest.mark.parametrize("auto_update", [True, False])
    def test_request_draw(self, auto_update):
        camera = pygfx.OrthographicCamera()
        canvas = WgpuCanvas()
        renderer = renderers.WgpuRenderer(canvas)
        try:
            ctrl = PynaVizController(camera, register_events=renderer, controller_id=None, auto_update=auto_update)
            ctrl._request_draw(renderer)
        finally:
            canvas.close()

    @pytest.mark.parametrize(
        "update_type, kwargs",
        [
            ("pan", dict(delta=(0.001, 0.001), vecx=np.zeros((3, )), vecy=np.zeros((3, )))),
            ("zoom", dict(delta=0.0001, screen_position=(100, 100), rect=(0, 0, 200, 300))),
            ("zoom_to_point", dict(delta=0.0001, screen_position=(100, 100), rect=(0, 0, 200, 300)))
        ]
    )
    def test_update_event(self, update_type, kwargs):
        camera = pygfx.OrthographicCamera()
        canvas = WgpuCanvas()
        renderer = renderers.WgpuRenderer(canvas)
        try:
            ctrl = PynaVizController(camera, register_events=renderer)
            state = ctrl._get_camera_state()
            ctrl._update_event(update_type=update_type, cam_state=state, **kwargs)
        finally:
            canvas.close()

    def test_compatibility_with_fpl(self):
        pass

    def test_update_event(self):
        pass

    def test_update_rule(self):
        pass

