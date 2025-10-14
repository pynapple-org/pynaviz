"""
Test for PlotTsd.
"""
import pathlib
import sys

import numpy as np
import pygfx as gfx
import pytest
from PIL import Image

import pynaviz as viz

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from config import TsdConfig


def test_plot_tsd_init(dummy_tsd):
    v = viz.PlotTsd(dummy_tsd)

    assert isinstance(v.controller, viz.controller.SpanController)
    assert isinstance(v.line, gfx.Line)
    v.close()


@pytest.mark.parametrize(
    "func, kwargs",
    TsdConfig.parameters,
)
def test_plot_tsd_action(dummy_tsd, func, kwargs):
    v = viz.PlotTsd(dummy_tsd)
    if func is not None:
        if isinstance(func, (list, tuple)):
            for n, k in zip(func, kwargs):
                getattr(v, n)(**k)
        else:
            getattr(v, func)(**kwargs)
    v.animate()
    image_data = v.renderer.snapshot()
    filename = TsdConfig._build_filename(func, kwargs)
    image = Image.open(
        pathlib.Path(__file__).parent / "screenshots" / filename
    ).convert("RGBA")
    np.allclose(np.array(image), image_data)
    v.close()

def test_plot_tsd_actions(dummy_tsd):
    # For coverage
    v = viz.PlotTsd(dummy_tsd)
    v.sort_by("a")
    v.group_by("b")
    v.animate()
    image_data = v.renderer.snapshot()

    image = Image.open(
        pathlib.Path(__file__).parent / "screenshots/test_plot_tsd.png"
    ).convert("RGBA")

    np.allclose(np.array(image), image_data)
    v.close()
