"""
Test for PlotTsdFrame
"""
import pathlib
import sys

import numpy as np
import pygfx as gfx
import pytest
from PIL import Image

import pynaviz as viz

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from config import TsdFrameConfig


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
        if isinstance(func, (list, tuple)):
            for n, k in zip(func, kwargs):
                getattr(v, n)(**k)
        else:
            getattr(v, func)(**kwargs)
    v.animate()
    image_data = v.renderer.snapshot()
    filename = TsdFrameConfig._build_filename(func, kwargs)
    image = Image.open(
        pathlib.Path(__file__).parent / "screenshots" / filename
    ).convert("RGBA")
    np.allclose(np.array(image), image_data)

