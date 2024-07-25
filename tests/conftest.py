"""
Testing configurations for the pynaviz library.

This module contains test fixtures required to set up and verify the functionality of modules
in the pynaviz library.
"""

import pytest
import numpy as np

import pynapple as nap


@pytest.fixture
def sine():
    """Generate single line data that can be used for testing visual creation."""
    xs = np.linspace(0, 100, 1_000)

    # sine data
    ys = np.sin(xs)

    # associated pynapple array
    sine = nap.Tsd(t=xs, d=ys)

    return sine


@pytest.fixture
def cosine():
    """Generate multi-line data that can be used for testing visual creation."""
    xs = np.linspace(0, 100, 1_000)

    # cosine data
    ys = np.cos(xs)
    # create multiple lines
    cosine = np.array([ys] * 10).T

    # pynapple object
    cosine = nap.TsdFrame(t=xs, d=cosine)

    return cosine


@pytest.fixture
def heatmap_data():
    """Generate 2D data for testing heatmap visual creation."""
    pass


@pytest.fixture
def movie_data():
    """Generate data for testing movie visual creation."""
    pass
