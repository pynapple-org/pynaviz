"""
Test for PlotTsd.
"""

import os

import numpy as np
import pygfx as gfx
import pynapple as nap
import pytest
from PIL import Image

import uviz as viz


@pytest.fixture
def dummy_tsd():
    return nap.Tsd(t=np.arange(0, 10, 0.1), d=np.sin(np.arange(0, 10, 0.1)))


def test_plot_tsd_init(dummy_tsd):
    v = viz.PlotTsd(dummy_tsd)

    assert isinstance(v.controller, viz.controller.SpanController)
    assert isinstance(v.line, gfx.Line)


def test_plot_tsd(dummy_tsd):
    v = viz.PlotTsd(dummy_tsd)
    v.animate()
    image_data = v.renderer.snapshot()

    try:
        image = Image.open(
            os.path.expanduser("tests/screenshots/test_plot_tsd.png")
        ).convert("RGBA")
    except Exception:
        image = Image.open(
            os.path.expanduser("screenshots/test_plot_tsd.png")
        ).convert("RGBA")

    np.allclose(np.array(image), image_data)
