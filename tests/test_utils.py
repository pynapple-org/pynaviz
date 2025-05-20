from wgpu.gui.offscreen import WgpuCanvas
from pygfx import renderers
from typing import Callable

def test_get_event_handle():
    from pynaviz.utils import _get_event_handle
    canvas = WgpuCanvas()
    renderer = renderers.WgpuRenderer(canvas)
    try:
        func = _get_event_handle(renderer)
        assert isinstance(func, Callable)
    finally:
        canvas.close()