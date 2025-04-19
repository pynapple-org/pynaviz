"""Tests for MovieItem."""

import pytest
import numpy as np

import fastplotlib as fpl

from pynaviz import MovieItem


def test_movie_item_instantiation(movie_data):
    """Test instantiation of MovieItem."""

    a = MovieItem(movie_data)

    # assert stored data is correct
    assert isinstance(a.data, type(movie_data))
    # assert correct graphic is the same
    assert isinstance(a.graphic, fpl.ImageGraphic)
    # assert data of graphic is correct
    # only checking first frame
    assert np.all(a.graphic.data.value == movie_data.get(0).astype("float32"))


def test_movie_instantiation_error(sine):
    """Assert trying to pass and object that is not a TsdFrame raises an error."""

    # assert error with incorrect pynapple object type
    with pytest.raises(ValueError):
        MovieItem(sine)

    # assert error with non-pynapply object as well
    data = np.random.random((5, 5))
    with pytest.raises(ValueError):
        MovieItem(data=data)
