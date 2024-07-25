"""Tests for LineItem."""

import pytest
import fastplotlib as fpl

from pynaviz import LineItem


def test_line_instantiation(line_data):
    sine, cosine = line_data

    a = LineItem(data=sine)

    # assert stored data is correct
    assert a.data == sine
    # assert correct graphic was made
    assert isinstance(a.graphic, fpl.LineGraphic)
    # assert data of graphic is correct
    assert a.graphic.data.value == sine.d.T

    b = LineItem(data=cosine)
