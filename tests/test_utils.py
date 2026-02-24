from typing import Callable

from pygfx import renderers
from rendercanvas.offscreen import RenderCanvas


def test_get_event_handle():
    from pynaviz.utils import _get_event_handle
    canvas = RenderCanvas()
    renderer = renderers.WgpuRenderer(canvas)
    try:
        func = _get_event_handle(renderer)
        assert isinstance(func, Callable)
    finally:
        canvas.close()
