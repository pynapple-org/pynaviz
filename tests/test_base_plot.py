"""
Test for _BasePlot. This should not render anything. Just testing for
the instantiation of _BasePlot. Methods of _BasePlot for acting on the
objects should be tested in the public classes.
"""
import os
import sys

import pygfx as gfx
import pynapple as nap
import pytest
from matplotlib.colors import Colormap

from pynaviz.base_plot import _BasePlot, _is_headless


@pytest.mark.parametrize(
    "env, platform, expected",
    [
        ({"CI": "true"}, "linux", True),  # Always headless in CI
        ({}, "linux", True),              # No DISPLAY -> headless
        ({"DISPLAY": ":0"}, "linux", False),  # DISPLAY present -> not headless
        ({"QT_QPA_PLATFORM": "offscreen"}, "darwin", True),  # macOS offscreen
        ({}, "darwin", False),            # macOS normal display
        ({"QT_QPA_PLATFORM": "offscreen"}, "win32", True),   # Windows offscreen
        ({}, "win32", False),             # Windows normal display
    ]
)
def test_is_headless(monkeypatch, env, platform, expected):
    # clear and patch environment
    monkeypatch.setattr(os, "environ", env.copy())
    monkeypatch.setattr(sys, "platform", platform)
    assert _is_headless() == expected


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
    base_plot.close()


def test_baseplot_change_data(dummy_tsd, dummy_ts):
    base_plot = _BasePlot(data=dummy_tsd)
    assert isinstance(base_plot.data, nap.Tsd)
    base_plot.data = dummy_ts
    assert isinstance(base_plot._data, nap.Ts)
    base_plot.close()

def test_baseplot_change_cmap(dummy_tsdframe):
    base_plot = _BasePlot(data=dummy_tsdframe)
    assert base_plot.cmap == "viridis"
    base_plot.cmap = "plasma"
    assert base_plot.cmap == "plasma"
    base_plot.close()

def test_cmap_setter(dummy_tsdframe):
    base_plot = _BasePlot(data=dummy_tsdframe)
    base_plot.cmap = Colormap("plasma")
    assert base_plot._cmap == "plasma"

    with pytest.warns(UserWarning):
        base_plot.cmap = 123

    with pytest.warns(UserWarning, match = r"Invalid colormap notacmap\. 'cmap' must be a matplotlib 'Colormap'."):
        base_plot.cmap = "notacmap"
    base_plot.close()

