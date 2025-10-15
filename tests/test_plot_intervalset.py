"""
Test for IntervalSet.
"""
import pathlib
import sys

import numpy as np
import pygfx as gfx
import pytest
from PIL import Image

import pynaviz as viz

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from config import IntervalSetConfig


def test_plot_iset_init(dummy_intervalset):
    v = viz.PlotIntervalSet(dummy_intervalset)

    assert isinstance(v.controller, viz.controller.SpanController)
    assert isinstance(v.graphic, dict)
    for m in v.graphic.values():
        assert isinstance(m, gfx.Mesh)
    v.close()

@pytest.mark.parametrize(
    "func, kwargs",
    IntervalSetConfig.parameters,
)
def test_plot_intervalset_action(dummy_intervalset, func, kwargs):
    v = viz.PlotIntervalSet(dummy_intervalset)
    if func is not None:
        if isinstance(func, (list, tuple)):
            for n, k in zip(func, kwargs):
                getattr(v, n)(**k)
        else:
            getattr(v, func)(**kwargs)
    v.animate()
    image_data = v.renderer.snapshot()
    filename = IntervalSetConfig._build_filename(func, kwargs)
    image = Image.open(
        pathlib.Path(__file__).parent / "screenshots" / filename
    ).convert("RGBA")
    np.allclose(np.array(image), image_data)
    v.close()
