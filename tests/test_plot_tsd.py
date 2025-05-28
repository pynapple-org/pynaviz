"""
Test for PlotTsd.
"""

import pytest
import numpy as np
import pynapple as nap
import pygfx as gfx
import pynaviz as viz


@pytest.fixture
def dummy_tsd():
    return nap.Tsd(t=np.arange(0, 10, 0.1), d=np.sin(np.arange(0, 10, 0.1)))


def test_plot_tsd_init(dummy_tsd):
    v = viz.PlotTsd(dummy_tsd)

    assert isinstance(v.controller, viz.controller.SpanController)
    assert isinstance(v.line, gfx.Line)


