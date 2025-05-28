"""
Testing configurations for the pynaviz library.

This module contains test fixtures required to set up and verify the functionality of modules
in the pynaviz library.
"""

import numpy as np
import pynapple as nap
import pytest

from pynaviz.events import SyncEvent


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
    data = np.random.rand(1000, 512, 512)

    xs = np.array([i for i in range(data.shape[0])])

    movie = nap.TsdTensor(t=xs, d=data)

    return movie


@pytest.fixture
def camera_state():
    cam_state = {
         'position': np.array([10., -10., 250.]),
         'rotation': np.array([0., 0., 0., 1.]),
         'scale': np.array([1., 1., 1.]),
         'reference_up': np.array([0., 1., 0.]),
         'fov': 0.0,
         'width': 500.0,
         'height': 400.0,
         'zoom': 1.0,
         'maintain_aspect': False,
         'depth_range': None
    }
    return cam_state


@pytest.fixture
def event_pan_update(camera_state):
    event = SyncEvent(
        "sync",
        controller_id=0,
        update_type="pan",
        sync_extra_args=dict(args=None,  kwargs=dict(cam_state=camera_state))
    )
    return event


@pytest.fixture
def event_zoom_update(camera_state):
    event = SyncEvent(
        "sync",
        controller_id=0,
        update_type="zoom",
        sync_extra_args=dict(args=None,  kwargs=dict(cam_state=camera_state))
    )
    return event


@pytest.fixture
def event_zoom_to_point_update(camera_state):
    event = SyncEvent(
        "sync",
        controller_id=0,
        update_type="zoom_to_point",
        sync_extra_args=dict(args=None,  kwargs=dict(cam_state=camera_state))
    )
    return event

