"""
Test for PlotTs.
"""
import pathlib
import sys

import numpy as np
import pygfx as gfx
import pytest
from PIL import Image

import pynaviz as viz

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from config import TsConfig


def test_plot_ts_init(dummy_ts):
    v = viz.PlotTs(dummy_ts)

    assert isinstance(v.controller, viz.controller.SpanYLockController)
    assert isinstance(v.graphic, gfx.Line)
    v.close()


@pytest.mark.parametrize(
    "func, kwargs",
    TsConfig.parameters,
)
def test_plot_ts_action(dummy_ts, func, kwargs):
    v = viz.PlotTs(dummy_ts)
    if func is not None:
        if isinstance(func, (list, tuple)):
            for n, k in zip(func, kwargs):
                getattr(v, n)(**k)
        else:
            getattr(v, func)(**kwargs)
    v.animate()
    image_data = v.renderer.snapshot()
    filename = TsConfig._build_filename(func, kwargs)
    image = Image.open(
        pathlib.Path(__file__).parent / "screenshots" / filename
    ).convert("RGBA")
    np.allclose(np.array(image), image_data)
    v.close()

def test_plot_ts_actions(dummy_ts):
    # For coverage
    v = viz.PlotTs(dummy_ts)
    v.sort_by("a")
    v.group_by("b")
    v.animate()
    image_data = v.renderer.snapshot()

    image = Image.open(
        pathlib.Path(__file__).parent / "screenshots/test_plot_ts.png"
    ).convert("RGBA")

    np.allclose(np.array(image), image_data)
    v.close()
