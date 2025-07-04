"""
Testing configurations for the uviz library.

This module contains test fixtures required to set up
and verify the functionality of modules in the uviz library.
"""
import pytest
import numpy as np
from . import config

from uviz.events import SyncEvent




# ---------- Fixtures ----------

@pytest.fixture
def dummy_tsd():
    return config.tsd()

@pytest.fixture
def dummy_tsdframe():
    return config.TsdFrameConfig.tsdframe()

@pytest.fixture
def dummy_intervalset():
    return config.intervalset()

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

