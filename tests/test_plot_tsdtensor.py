"""
Test for PlotTsdTensorTensor.
"""
import pathlib
import sys

import numpy as np
import pygfx as gfx
import pytest
from PIL import Image

import pynaviz as viz

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from config import TsdTensorConfig


def test_plot_tsdtensor_init(dummy_tsdtensor):
    v = viz.PlotTsdTensor(dummy_tsdtensor)

    assert isinstance(v.controller, viz.controller.GetController)
    assert isinstance(v.image, gfx.Image)
    v.close()


@pytest.mark.parametrize(
    "func, kwargs",
    TsdTensorConfig.parameters,
)
def test_plot_tsdtensor_action(dummy_tsdtensor, func, kwargs):
    v = viz.PlotTsdTensor(dummy_tsdtensor)
    if func is not None:
        if isinstance(func, (list, tuple)):
            for n, k in zip(func, kwargs):
                getattr(v, n)(**k)
        else:
            getattr(v, func)(**kwargs)
    v.animate()
    image_data = v.renderer.snapshot()
    filename = TsdTensorConfig._build_filename(func, kwargs)
    image = Image.open(
        pathlib.Path(__file__).parent / "screenshots" / filename
    ).convert("RGBA")
    np.allclose(np.array(image), image_data)
    v.close()

def test_plot_tsdtensor_actions(dummy_tsdtensor):
    # For coverage
    v = viz.PlotTsdTensor(dummy_tsdtensor)
    v.sort_by("a")
    v.group_by("b")
    v.animate()
    image_data = v.renderer.snapshot()

    image = Image.open(
        pathlib.Path(__file__).parent / "screenshots/test_plot_tsdtensor.png"
    ).convert("RGBA")

    np.allclose(np.array(image), image_data)
    v.close()
