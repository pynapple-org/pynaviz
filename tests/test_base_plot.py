"""
Test for _BasePlot. This should not render anything. Just testing for
the instantiation of _BasePlot. Methods of _BasePlot for acting on the
objects should be tested in the public classes.
"""

import numpy as np
import pygfx as gfx
import pynapple as nap
import pytest

from pynaviz.base_plot import _BasePlot


def test_baseplot_init(dummy_tsd):
    base_plot = _BasePlot(data=dummy_tsd)
    assert isinstance(base_plot.data, nap.Tsd)
    # assert isinstance(base_plot.canvas, gfx.WgpuCanvas)
    assert isinstance(base_plot.renderer, gfx.Renderer)
    assert isinstance(base_plot.scene, gfx.Scene)
    assert isinstance(base_plot.ruler_x, gfx.Ruler)
    assert isinstance(base_plot.ruler_y, gfx.Ruler)
    # assert isinstance(base_plot.ruler_ref_time, gfx.Ruler)
    assert isinstance(base_plot.camera, gfx.OrthographicCamera)

    assert base_plot.cmap == "viridis"
