"""
Test for IntervalSet.
"""

import os

import numpy as np
import pygfx as gfx
import pynapple as nap
import pytest
from PIL import Image

import uviz as viz


def test_plot_iset_init(dummy_intervalset):
    v = viz.PlotIntervalSet(dummy_intervalset)

    assert isinstance(v.controller, viz.controller.SpanController)
    assert isinstance(v.graphic, dict)
    for k, v in v.graphic.items():
        assert isinstance(v, gfx.Mesh)


def test_plot_iset(dummy_intervalset):
    v = viz.PlotIntervalSet(dummy_intervalset)
    v.animate()
    image_data = v.renderer.snapshot()

    try:
        image = Image.open(
            os.path.expanduser("tests/screenshots/test_plot_intervalset.png")
        ).convert("RGBA")
    except Exception:
        image = Image.open(
            os.path.expanduser("screenshots/test_plot_intervalset.png")
        ).convert("RGBA")

    np.allclose(np.array(image), image_data)
