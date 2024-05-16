import numpy as np
from pygfx import controllers, cameras
from pynaviz.controller import PynaVizController


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
    def test_compatibility_with_fpl(self):
        pass

    def test_update_rule(self):
        pass

    def test_update_event(self):
        pass

    def test_update_rule(self):
        pass




