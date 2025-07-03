"""
Test for PlotTsdFrame
"""
import pathlib

import numpy as np
import pygfx as gfx
import pynapple as nap
import pytest
from PIL import Image

import uviz as viz




def test_plot_tsdframe_init(dummy_tsdframe):
    v = viz.PlotTsdFrame(dummy_tsdframe)

    assert isinstance(v.controller, viz.controller.SpanController)
    assert isinstance(v.graphic, gfx.Line)


def test_plot_tsdframe(dummy_tsdframe):
    v = viz.PlotTsdFrame(dummy_tsdframe)
    v.animate()
    image_data = v.renderer.snapshot()

    image = Image.open(
        pathlib.Path(__file__).parent / "screenshots/test_plot_tsdframe.png"
    ).convert("RGBA")

    np.allclose(np.array(image), image_data)

@pytest.mark.parametrize(
    "func, kwargs, extension",
    [
        ("group_by", {"metadata_name":"group"}, ""),
        ("sort_by", {"metadata_name":"channel"}, ""),
        ("plot_x_vs_y", {"x_label":0,"y_label":1}, ""),
    ],
)
def test_plot_tsdframe_action(dummy_tsdframe, func, kwargs, extension):
    v = viz.PlotTsdFrame(dummy_tsdframe)
    getattr(v, func)(**kwargs)
    v.animate()
    image_data = v.renderer.snapshot()
    image = Image.open(
        pathlib.Path(__file__).parent / f"screenshots/test_plot_tsdframe_{func}{extension}.png"
    ).convert("RGBA")
    np.allclose(np.array(image), image_data)

