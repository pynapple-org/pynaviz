"""Test suite for controller group."""
from unittest.mock import MagicMock, Mock

import pygfx as gfx
import pytest
from pygfx import Viewport
from rendercanvas.offscreen import RenderCanvas

from pynaviz.controller_group import ControllerGroup
from pynaviz.events import SyncEvent, SwitchEvent


class MockControllerNoXLim:
    def __init__(self, enabled: bool):
        self._controller_id = None
        self.enabled = enabled
        self.xlim = None
        self.sync_called_with = []  # Track sync calls
        self.advance_called_with = []  # Track advance calls

    @property
    def controller_id(self):
        return self._controller_id

    @controller_id.setter
    def controller_id(self, value):
        self._controller_id = value

    def advance(self, *args, **kwargs):
        self.advance_called_with.append((args, kwargs))

    def sync(self, event):
        self.sync_called_with.append(event)


class MockController(MockControllerNoXLim):
    def set_xlim(self, xmin, xmax):
        self.xlim = (xmin, xmax)


@pytest.fixture
def mock_plots():
    """Create mock plots with controllers and renderers."""
    plots = []
    for _ in range(3):
        p = MagicMock()
        p.controller = MockController(enabled=True)
        canvas = RenderCanvas()
        p.renderer = gfx.WgpuRenderer(canvas)
        plots.append(p)
    return plots


@pytest.fixture
def mock_plots_no_set_xlim():
    """Create mock plots with controllers and renderers."""
    plots = []
    for _ in range(3):
        p = MagicMock()
        p.controller = MockControllerNoXLim(enabled=True)
        canvas = RenderCanvas()
        p.renderer = gfx.WgpuRenderer(canvas)
        plots.append(p)
    return plots


def test_controller_group_init(mock_plots):
    """Test basic initialization of ControllerGroup."""
    cg = ControllerGroup(mock_plots, interval=(1, 2))

    # Verify xlim was set on first controller
    assert mock_plots[0].controller.xlim == (1, 2), "xlim should be set to interval"

    # Verify controller IDs are assigned sequentially
    for i, p in enumerate(mock_plots):
        assert p.controller.controller_id == i, f"Controller {i} should have ID {i}"

    # Verify internal structure
    assert isinstance(cg._controller_group, dict), "_controller_group should be a dict"
    assert len(cg._controller_group) == len(mock_plots), "Should have one entry per plot"

    # Verify dictionary contains the actual controller objects
    for i, plt in enumerate(mock_plots):
        assert cg._controller_group[i] is plt.controller, f"Controller {i} should match"

    # Verify event handlers are registered correctly
    for plt in mock_plots:
        viewport = Viewport.from_viewport_or_renderer(plt.renderer)
        sync_handlers = viewport.renderer._event_handlers.get("sync", set())
        switch_handlers = viewport.renderer._event_handlers.get("switch", set())

        assert len(sync_handlers) == 1, "Should have exactly one sync handler"
        assert cg.sync_controllers in sync_handlers, "sync_controllers should be registered"

        assert len(switch_handlers) == 1, "Should have exactly one switch handler"
        assert cg.switch_controller in switch_handlers, "switch_controller should be registered"

    # Verify current_time is set to middle of interval
    assert cg.current_time == 1.5, "current_time should be at interval midpoint"
    assert cg.interval == (1, 2), "interval should be stored correctly"


def test_controller_group_init_empty():
    """Test initialization with no plots."""
    cg = ControllerGroup(plots=None, interval=(0, 10))
    assert len(cg._controller_group) == 0, "Should have no controllers"
    assert cg.current_time == 5.0, "current_time should be at interval midpoint"


def test_controller_group_init_invalid_interval():
    """Test that invalid intervals raise ValueError."""
    with pytest.raises(ValueError, match="must be a tuple or list"):
        ControllerGroup(plots=None, interval=5)

    with pytest.raises(ValueError, match="must be a 2-tuple"):
        ControllerGroup(plots=None, interval=(1, 2, 3))

    with pytest.raises(ValueError, match="must be a 2-tuple"):
        ControllerGroup(plots=None, interval=(1, "two"))


def test_controller_group_with_callback(mock_plots):
    """Test that callback is called during initialization."""
    callback = Mock()
    ControllerGroup(mock_plots, interval=(0, 10), callback=callback)
    callback.assert_called_once_with(5.0)  # Should be called with midpoint


def test_sync_controller_cam_state(mock_plots):
    callback = Mock()
    # disable a controller
    mock_plots[0].controller.enabled = False
    cg = ControllerGroup(mock_plots, callback=callback, interval=(0, 10))
    callback.assert_called_with(5)
    # mock event
    event = Mock()
    event.kwargs = {"cam_state": {"position": [11, 12, 13]}}
    event.controller_id = 1
    cg.sync_controllers(event)
    # check that callable was triggered
    callback.assert_called_with(11)
    # check that controller 1 (sender) wasn't sync
    assert cg._controller_group[1].sync_called_with == []
    # check that controller 0 (disabled) was not synced
    assert cg._controller_group[0].sync_called_with == []
    # check that controller 2 was sync
    assert cg._controller_group[2].sync_called_with == [event]
    # check current time
    assert cg.current_time == 11

def test_sync_controller_time(mock_plots):
    callback = Mock()
    # disable a controller
    mock_plots[2].controller.enabled = False
    cg = ControllerGroup(mock_plots, callback=callback, interval=(0, 10))
    callback.assert_called_with(5)
    # mock event
    event = Mock()
    event.kwargs = {"current_time": 11}
    event.controller_id = 1
    cg.sync_controllers(event)
    # check that callable was triggered
    callback.assert_called_with(11)
    # check that controller 1 (sender) wasn't sync
    assert cg._controller_group[1].sync_called_with == []
    # check that controller 0 (disabled) was not synced
    assert cg._controller_group[2].sync_called_with == []
    # check that controller 2 was sync
    assert cg._controller_group[0].sync_called_with == [event]
    # check current time
    assert cg.current_time == 11


@pytest.mark.parametrize("interval", [(1, 2), (1, 2.), (1., 2.), (1, None)])
def test_set_interval_with_set_xlim(mock_plots, interval: tuple[float | int, float | int | None]):
    callback = Mock()
    # disable a controller
    mock_plots[2].controller.enabled = False
    cg = ControllerGroup(mock_plots, callback=callback)
    cg.set_interval(*interval)
    msg = "Current time not set at the midpoint of the interval." if interval[1] is not None else "Current time not set to start."
    assert cg.current_time == interval[0] if interval[1] is None else (interval[1] + interval[0]) * 0.5, msg
    callback.assert_called_with(cg.current_time)
    # enabled controller:
    if interval[1] is None:
        # enabled controllers
        for i in [0, 1]:
            ctrl = cg._controller_group[i]
            assert len(ctrl.sync_called_with) == 1, f"sync called {len(ctrl.sync_called_with)} times. It should have been called once."
            event = ctrl.sync_called_with[0]
            assert event.kwargs["current_time"] == cg.current_time, f"controller set to the wrong time."
            assert isinstance(event, SyncEvent)
            assert event.update_type == "pan"
        # disabled controller
        ctrl = cg._controller_group[2]
        assert len(ctrl.sync_called_with) == 0, f"sync called for a disabled controller."

    else:
        ctrl = cg._controller_group[0]
        assert ctrl.xlim == interval, "did not set xlim properly."
        # assert that only the first controller sets
        for i in [1, 2]:
            ctrl = cg._controller_group[i]
            assert ctrl.xlim is None, "xlim called for more than one controller."


@pytest.mark.parametrize("interval", [(1, 2), (1, 2.), (1., 2.), (1, None)])
def test_set_interval_without_set_xlim(mock_plots_no_set_xlim, interval: tuple[float | int, float | int | None]):
    callback = Mock()
    # disable a controller
    mock_plots_no_set_xlim[2].controller.enabled = False
    cg = ControllerGroup(mock_plots_no_set_xlim, callback=callback)

    cg.set_interval(*interval)
    msg = "Current time not set at the midpoint of the interval." if interval[
                                                                         1] is not None else "Current time not set to start."
    assert cg.current_time == interval[0] if interval[1] is None else (interval[1] + interval[0]) * 0.5, msg
    callback.assert_called_with(cg.current_time)
    # enabled controllers
    for i in [0, 1]:
        ctrl = cg._controller_group[i]
        print(ctrl.sync_called_with[0].kwargs)
        print(ctrl.sync_called_with[1].kwargs)
        assert len(
            ctrl.sync_called_with) == 2, f"sync called {len(ctrl.sync_called_with)} times. It should have been called twice, onece at the init and onece when setting time."
        event = ctrl.sync_called_with[1]
        assert event.kwargs["current_time"] == cg.current_time, f"controller set to the wrong time."
        assert isinstance(event, SyncEvent)
        assert event.update_type == "pan"
    # disabled controller
    ctrl = cg._controller_group[2]
    assert len(ctrl.sync_called_with) == 0, f"sync called for a disabled controller."