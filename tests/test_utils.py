from typing import Callable

from pygfx import renderers
from wgpu.gui.offscreen import WgpuCanvas


def test_get_event_handle():
    from pynaviz.utils import _get_event_handle
    canvas = WgpuCanvas()
    renderer = renderers.WgpuRenderer(canvas)
    try:
        func = _get_event_handle(renderer)
        assert isinstance(func, Callable)
    finally:
        canvas.close()
