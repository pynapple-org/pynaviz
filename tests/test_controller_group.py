"""Test suite for controller group."""
from unittest.mock import MagicMock, Mock

import pygfx as gfx
import pytest
from pygfx import Viewport
from rendercanvas.offscreen import RenderCanvas

from pynaviz.controller_group import ControllerGroup
from pynaviz.events import SyncEvent


class MockControllerNoXLim:
    def __init__(self, enabled: bool, renderer: gfx.Renderer | None = None):
        self._controller_id = None
        self.enabled = enabled
        self.xlim = None
        self.sync_called_with = []  # Track sync calls
        self.advance_called_with = []  # Track advance calls
        self.renderer = renderer

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
        canvas = RenderCanvas()
        p.renderer = gfx.WgpuRenderer(canvas)
        p.controller = MockController(enabled=True, renderer=p.renderer)
        plots.append(p)
    return plots


@pytest.fixture
def mock_plots_no_set_xlim():
    """Create mock plots with controllers and renderers."""
    plots = []
    for _ in range(3):
        p = MagicMock()
        canvas = RenderCanvas()
        p.renderer = gfx.WgpuRenderer(canvas)
        p.controller = MockControllerNoXLim(enabled=True, renderer=p.renderer)
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
            assert event.kwargs["current_time"] == cg.current_time, "controller set to the wrong time."
            assert isinstance(event, SyncEvent)
            assert event.update_type == "pan"
        # disabled controller
        ctrl = cg._controller_group[2]
        assert len(ctrl.sync_called_with) == 0, "sync called for a disabled controller."

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
        assert event.kwargs["current_time"] == cg.current_time, "controller set to the wrong time."
        assert isinstance(event, SyncEvent)
        assert event.update_type == "pan"
    # disabled controller
    ctrl = cg._controller_group[2]
    assert len(ctrl.sync_called_with) == 0, "sync called for a disabled controller."


def test_switch_controller(mock_plots):
    """Test switching a controller in the group."""
    cg = ControllerGroup(mock_plots, interval=(0, 10))

    # Create a new controller to switch to
    new_controller = MockController(enabled=True)
    new_controller.controller_id = 1

    # Create a switch event
    event = Mock()
    event.controller_id = 1
    event.new_controller = new_controller

    # Verify original controller is in place
    original_controller = cg._controller_group[1]
    assert original_controller is mock_plots[1].controller

    # Trigger the switch
    cg.switch_controller(event)

    # Verify the controller was replaced
    assert cg._controller_group[1] is new_controller
    assert cg._controller_group[1] is not original_controller


def test_switch_controller_no_new_controller(mock_plots):
    """Test that switch_controller does nothing when new_controller is None."""
    cg = ControllerGroup(mock_plots, interval=(0, 10))

    original_controller = cg._controller_group[1]

    # Event without new_controller attribute
    event = Mock(spec=[])
    delattr(event, 'new_controller')  # Ensure attribute doesn't exist
    event.controller_id = 1

    cg.switch_controller(event)

    # Verify nothing changed
    assert cg._controller_group[1] is original_controller

    # Event with new_controller = None
    event.new_controller = None
    cg.switch_controller(event)

    # Verify nothing changed
    assert cg._controller_group[1] is original_controller


def test_switch_controller_invalid_id(mock_plots):
    """Test switching with a controller_id that doesn't exist."""
    cg = ControllerGroup(mock_plots, interval=(0, 10))

    new_controller = MockController(enabled=True)

    event = Mock()
    event.controller_id = 999  # Non-existent ID
    event.new_controller = new_controller

    # Should not raise an error, just do nothing
    cg.switch_controller(event)

    # Verify original controllers are unchanged
    for i in range(3):
        assert cg._controller_group[i] is mock_plots[i].controller


def test_advance(mock_plots):
    """Test advancing time in the controller group."""
    callback = Mock()
    cg = ControllerGroup(mock_plots, callback=callback, interval=(0, 10))

    # Reset the callback to clear initialization call
    callback.reset_mock()

    # Store initial current_time
    initial_time = cg.current_time

    # Advance by default delta (0.025)
    cg.advance()

    # Verify current_time was incremented
    assert cg.current_time == initial_time + 0.025

    # Verify advance was called on the first controller
    first_controller = cg._controller_group[0]
    assert len(first_controller.advance_called_with) == 1
    assert first_controller.advance_called_with[0] == ((0.025,), {})

    # Verify other controllers were not called directly
    for i in [1, 2]:
        ctrl = cg._controller_group[i]
        assert len(ctrl.advance_called_with) == 0


def test_advance_custom_delta(mock_plots):
    """Test advancing time with a custom delta."""
    callback = Mock()
    cg = ControllerGroup(mock_plots, callback=callback, interval=(0, 10))

    initial_time = cg.current_time
    delta = 0.5

    # Advance by custom delta
    cg.advance(delta=delta)

    # Verify current_time was incremented by custom delta
    assert cg.current_time == initial_time + delta

    # Verify advance was called with custom delta
    first_controller = cg._controller_group[0]
    assert len(first_controller.advance_called_with) == 1
    assert first_controller.advance_called_with[0] == ((delta,), {})


def test_advance_multiple_times(mock_plots):
    """Test advancing time multiple times."""
    cg = ControllerGroup(mock_plots, interval=(0, 10))

    initial_time = cg.current_time

    # Advance 3 times
    cg.advance(delta=0.1)
    cg.advance(delta=0.2)
    cg.advance(delta=0.3)

    # Verify current_time accumulated correctly
    assert cg.current_time == pytest.approx(initial_time + 0.6)

    # Verify advance was called 3 times on first controller
    first_controller = cg._controller_group[0]
    assert len(first_controller.advance_called_with) == 3
    assert first_controller.advance_called_with[0] == ((0.1,), {})
    assert first_controller.advance_called_with[1] == ((0.2,), {})
    assert first_controller.advance_called_with[2] == ((0.3,), {})


def test_advance_empty_group(capsys):
    """Test advancing when controller group is empty."""
    cg = ControllerGroup(plots=None, interval=(0, 10))

    # Should handle empty group gracefully
    cg.advance()

    # Verify message was printed
    captured = capsys.readouterr()
    assert "Controller group is empty, nothing to advance." in captured.out


def test_add_plot(mock_plots):
    """Test adding a plot to an existing controller group."""
    # Create a group with 2 plots
    cg = ControllerGroup(mock_plots[:2], interval=(0, 10))

    # Create a new plot to add
    new_plot = MagicMock()
    new_plot.controller = MockController(enabled=True)
    canvas = RenderCanvas()
    new_plot.renderer = gfx.WgpuRenderer(canvas)

    # Add the new plot with controller_id 2
    cg.add(new_plot, controller_id=2)

    # Verify the controller was added
    assert 2 in cg._controller_group
    assert cg._controller_group[2] is new_plot.controller
    assert new_plot.controller.controller_id == 2

    # Verify event handlers were registered
    viewport = Viewport.from_viewport_or_renderer(new_plot.renderer)
    sync_handlers = viewport.renderer._event_handlers.get("sync", set())
    switch_handlers = viewport.renderer._event_handlers.get("switch", set())

    assert {cg.sync_controllers} == sync_handlers
    assert {cg.switch_controller} == switch_handlers
    assert new_plot.controller.controller_id == 2

    # Verify the new controller was synced to current_time
    assert len(new_plot.controller.sync_called_with) == 1
    sync_event = new_plot.controller.sync_called_with[0]
    assert isinstance(sync_event, SyncEvent)
    assert sync_event.type == "sync"
    assert sync_event.controller_id == 2
    assert sync_event.update_type == "pan"
    assert sync_event.kwargs["current_time"] == cg.current_time
    # check that the other controller were not sync
    for i in [0,1]:
        assert len(cg._controller_group[i].sync_called_with) == 0


def test_add_plot_with_wrapper():
    """Test adding a plot that's wrapped (has .plot.controller and .plot.renderer)."""
    cg = ControllerGroup(plots=None, interval=(0, 10))

    # Create a wrapped plot
    class PlotWrapper:
        def __init__(self):
            self.plot = MagicMock()
    wrapper = PlotWrapper()
    wrapper.plot.controller = MockController(enabled=True)
    canvas = RenderCanvas()
    wrapper.plot.renderer = gfx.WgpuRenderer(canvas)

    # Add the wrapped plot
    cg.add(wrapper, controller_id=0)

    # Verify the controller was added
    assert 0 in cg._controller_group
    assert cg._controller_group[0] is wrapper.plot.controller
    assert wrapper.plot.controller.controller_id == 0


def test_add_plot_duplicate_id(mock_plots):
    """Test that adding a plot with duplicate controller_id raises RuntimeError."""
    cg = ControllerGroup(mock_plots, interval=(0, 10))

    # Create a new plot
    new_plot = MagicMock()
    new_plot.controller = MockController(enabled=True)
    canvas = RenderCanvas()
    new_plot.renderer = gfx.WgpuRenderer(canvas)

    # Try to add with an existing ID
    with pytest.raises(RuntimeError, match="Controller ID 1 already exists in the group"):
        cg.add(new_plot, controller_id=1)


def test_add_plot_no_controller():
    """Test that adding a plot without controller/renderer raises RuntimeError."""
    cg = ControllerGroup(plots=None, interval=(0, 10))

    # Create a plot without required attributes
    bad_plot = MagicMock(spec=[])  # No controller or renderer

    with pytest.raises(RuntimeError, match="Plot object must have a controller and renderer"):
        cg.add(bad_plot, controller_id=0)


def test_add_plot_controller_id_already_assigned():
    """Test adding a plot when controller already has an ID assigned."""
    cg = ControllerGroup(plots=None, interval=(0, 10))

    # create a plot with pre-assigned controller_id
    plot = MagicMock()
    plot.controller = MockController(enabled=True)
    plot.controller.controller_id = 99  # Pre-assigned ID
    canvas = RenderCanvas()
    plot.renderer = gfx.WgpuRenderer(canvas)

    # add the plot
    cg.add(plot, controller_id=0)
    # verify the pre-assigned ID is re-assigned
    assert plot.controller.controller_id == 0
    assert cg._controller_group[0] is plot.controller


def test_remove_controller(mock_plots):
    """Test removing a controller from the group."""
    cg = ControllerGroup(mock_plots, interval=(0, 10))

    # Verify controller 1 exists
    assert 1 in cg._controller_group

    # Remove controller 1
    cg.remove(controller_id=1)

    # Verify it was removed
    assert 1 not in cg._controller_group
    assert len(cg._controller_group) == 2

    # Verify other controllers are still there
    assert 0 in cg._controller_group
    assert 2 in cg._controller_group


def test_remove_controller_invalid_id(mock_plots):
    """Test that removing a non-existent controller raises KeyError."""
    cg = ControllerGroup(mock_plots, interval=(0, 10))

    with pytest.raises(KeyError, match="Controller ID 999 not found in the group"):
        cg.remove(controller_id=999)


def test_remove_controller_event_handlers(mock_plots):
    """Test that both sync and switch event handlers are removed when controller is removed."""
    cg = ControllerGroup(mock_plots, interval=(0, 10))

    # Get the renderer for controller 1
    renderer = mock_plots[1].renderer
    viewport = Viewport.from_viewport_or_renderer(renderer)

    # Verify both handlers are registered before removal
    sync_handlers_before = viewport.renderer._event_handlers.get("sync", set())
    switch_handlers_before = viewport.renderer._event_handlers.get("switch", set())
    assert cg.sync_controllers in sync_handlers_before
    assert cg.switch_controller in switch_handlers_before

    # Remove controller 1
    cg.remove(controller_id=1)

    # Verify both handlers were removed
    sync_handlers_after = viewport.renderer._event_handlers.get("sync", set())
    switch_handlers_after = viewport.renderer._event_handlers.get("switch", set())
    assert sync_handlers_after == set()
    assert switch_handlers_after == set()

    # Verify controller was removed from group
    assert 1 not in cg._controller_group


def test_remove_all_controllers(mock_plots):
    """Test removing all controllers leaves an empty group."""
    cg = ControllerGroup(mock_plots, interval=(0, 10))

    # Remove all controllers
    cg.remove(controller_id=0)
    cg.remove(controller_id=1)
    cg.remove(controller_id=2)

    # Verify group is empty
    assert len(cg._controller_group) == 0


def test_remove_controller_no_renderer_attribute():
    """Test removing a controller when it doesn't have a renderer attribute (fallback case)."""
    cg = ControllerGroup(plots=None, interval=(0, 10))

    # Manually add a controller without proper renderer reference
    mock_controller = MockController(enabled=True)
    mock_controller.controller_id = 0
    # Don't set renderer attribute
    cg._controller_group[0] = mock_controller

    # Should not raise an error, just skip event handler removal
    cg.remove(controller_id=0)

    # Verify controller was still removed
    assert 0 not in cg._controller_group
