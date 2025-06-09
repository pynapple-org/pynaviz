"""
Test for IntervalSet.
"""

import os

import numpy as np
import pygfx as gfx
import pynapple as nap
import pytest
from PIL import Image

import pynaviz as viz


@pytest.fixture
def dummy_iset():
    ep = nap.IntervalSet(
        [0, 0.2, 0.4, 0.6, 0.8],
        [0.19, 0.39, 0.59, 0.79, 0.99],
        metadata={
            "label": ["a", "b", "c", "d", "e"],
            "choice": [1, 0, 1, 1, 0],
            "reward": [0, 0, 1, 0, 1],
        },
    )
    return ep


def test_plot_iset_init(dummy_iset):
    v = viz.PlotIntervalSet(dummy_iset)

    assert isinstance(v.controller, viz.controller.SpanController)
    assert isinstance(v.line, gfx.Line)
