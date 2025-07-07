"""
Test for PlotTsdFrame
"""
import pathlib

import numpy as np
import pygfx as gfx
import pynapple as nap
import pytest
from PIL import Image
from .config import TsdFrameConfig
import uviz as viz




def test_plot_tsdframe_init(dummy_tsdframe):
    v = viz.PlotTsdFrame(dummy_tsdframe)

    assert isinstance(v.controller, viz.controller.SpanController)
    assert isinstance(v.graphic, gfx.Line)


@pytest.mark.parametrize(
    "func, kwargs",
    TsdFrameConfig.parameters,
)
def test_plot_tsdframe_action(dummy_tsdframe, func, kwargs):
    v = viz.PlotTsdFrame(dummy_tsdframe)
    if func is not None:
        getattr(v, func)(**kwargs)
    v.animate()
    image_data = v.renderer.snapshot()

    filename = TsdFrameConfig._build_filename(func, kwargs)
    image = Image.open(
        pathlib.Path(__file__).parent / "screenshots" / filename
    ).convert("RGBA")
    np.allclose(np.array(image), image_data)

