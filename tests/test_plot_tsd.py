"""
Test for PlotTsd.
"""
import pathlib

import numpy as np
import pygfx as gfx
from PIL import Image

import uviz as viz


def test_plot_tsd_init(dummy_tsd):
    v = viz.PlotTsd(dummy_tsd)

    assert isinstance(v.controller, viz.controller.SpanController)
    assert isinstance(v.line, gfx.Line)


def test_plot_tsd(dummy_tsd):
    v = viz.PlotTsd(dummy_tsd)
    v.animate()
    image_data = v.renderer.snapshot()

    image = Image.open(
        pathlib.Path(__file__).parent / "screenshots/test_plot_tsd.png"
    ).convert("RGBA")

    np.allclose(np.array(image), image_data)

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
