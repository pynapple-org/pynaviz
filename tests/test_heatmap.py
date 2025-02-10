"""Tests for HeatmapItem."""

import pytest
import numpy as np

import fastplotlib as fpl


from pynaviz import HeatmapItem


def test_heatmap_item_instantiation(cosine):
    """Test instantiation of HeatmapItem."""

    a = HeatmapItem(cosine)

    # assert stored data is correct
    assert isinstance(a.data, type(cosine))
    # assert correct graphic was made
    assert isinstance(a.graphic, fpl.ImageGraphic)
    # assert data of graphic is correct
    assert np.all(a.graphic.data.value == cosine.d.T.astype("float32"))


def test_heatmap_instantiation_error(sine):
    """Assert trying to pass and object that is not a TsdFrame raises an error."""

    # assert error with incorrect pynapple object type
    with pytest.raises(ValueError):
        HeatmapItem(sine)

    # assert error with non-pynapply object as well
    data = np.random.random((5, 5))
    with pytest.raises(ValueError):
        HeatmapItem(data=data)