"""
Test for PlotTsGroup.
"""
import pathlib
import sys

import numpy as np
import pytest
from PIL import Image

import pynaviz as viz

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from config import TsGroupConfig


def test_plot_tsgroup_init(dummy_tsgroup):
    v = viz.PlotTsGroup(dummy_tsgroup)

    assert isinstance(v.controller, viz.controller.SpanController)
    v.close()


@pytest.mark.parametrize(
    "func, kwargs",
    TsGroupConfig.parameters,
)
def test_plot_tsgroup_action(dummy_tsgroup, func, kwargs):
    v = viz.PlotTsGroup(dummy_tsgroup)
    if func is not None:
        if isinstance(func, (list, tuple)):
            for n, k in zip(func, kwargs):
                getattr(v, n)(**k)
        else:
            getattr(v, func)(**kwargs)
    v.animate()
    image_data = v.renderer.snapshot()
    filename = TsGroupConfig._build_filename(func, kwargs)
    image = Image.open(
        pathlib.Path(__file__).parent / "screenshots" / filename
    ).convert("RGBA")
    np.allclose(np.array(image), image_data)
    v.close()
