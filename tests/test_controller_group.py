"""Test suite for controller group."""
from unittest.mock import MagicMock, Mock

import pygfx as gfx
import pytest
from pygfx import Viewport
from rendercanvas.offscreen import RenderCanvas

from pynaviz.controller_group import ControllerGroup


class MockController:
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
