"""Tests for LineItem."""

import pytest
import numpy as np
import fastplotlib as fpl
import pynapple as nap

from pynaviz import LineItem

def test_single_line_item_instantiation(sine):
    """Test single line LineItem instantiation."""
    a = LineItem(data=sine)

    # assert stored data is correct
    # TODO: no pynapple object comparison capabilities so just making sure correct type for now
    assert isinstance(a.data, type(sine))
    # assert correct graphic was made
    assert isinstance(a.graphic, fpl.LineGraphic)
    # assert data of graphic is correct
    # NOTE: need to cast to float32 to match fastplotlib data conversion
    assert np.all(a.graphic.data.value[:, 1] == sine.d.T.astype("float32"))


def test_multi_line_item_instantiation(cosine):
    """Test multi-line LineItem instantiation."""
    b = LineItem(data=cosine)

    # assert stored data is correct
    assert isinstance(b.data, type(cosine))
    # assert correct graphic was made
    assert isinstance(b.graphic, fpl.LineStack)
    # assert data of graphic is correct
    # NOTE: need to cast to float32 to match fastplotlib data conversion
    assert np.all(b.graphic.data[:][:, :, 1] == cosine.d.T.astype("float32"))
    # assert that the number of data columns of the pynapple object are how many lines were created
    assert len(b.graphic) == len(cosine.columns)


def test_line_instantiation_error():
    """Assert trying to pass an object that is not a Tsd or TsdFrame raises an error."""
    # generate bad data
    data = np.random.rand(10, 256, 256)

    # make pynapple object
    a = nap.TsdTensor(t=data[:, 1, 1], d=data)

    with pytest.raises(ValueError):
        LineItem(data=a)

    # assert trying to pass an array also throws an error
    with pytest.raises(ValueError):
        LineItem(data=data)

