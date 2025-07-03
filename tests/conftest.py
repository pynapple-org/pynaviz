"""
Testing configurations for the uviz library.

This module contains test fixtures required to set up and verify the functionality of modules
in the uviz library.
"""

import numpy as np
import pynapple as nap
import pytest

from uviz.events import SyncEvent

# pytest.fixture can't be called directly
# to generate screenshots

def tsd():
    return nap.Tsd(t=np.arange(0, 10, 0.1),
                   d=np.sin(np.arange(0, 10, 0.1))
                   )

def tsdframe():
    t = np.arange(0, 10, 0.1)
    offsets = np.linspace(0, 2 * np.pi, 5, endpoint=False)
    d = np.cos(t[None, :] + offsets[:, None])
    return nap.TsdFrame(
        t=t,
        d=d.T,
        metadata = {
            "group":[0, 0, 1, 0, 1],
            "channel":[1, 3, 0, 2, 4],
            "random":np.random.randn(5)
        }
    )

def intervalset():
    return nap.IntervalSet(
        [0, 0.2, 0.4, 0.6, 0.8],
        [0.19, 0.39, 0.59, 0.79, 0.99],
        metadata={
            "label": ["a", "b", "c", "d", "e"],
            "choice": [1, 0, 1, 1, 0],
            "reward": [0, 0, 1, 0, 1],
        },
    )




@pytest.fixture
def dummy_tsd():
    return tsd()

@pytest.fixture
def dummy_tsdframe():
    return tsdframe()

@pytest.fixture
def dummy_intervalset():
    return intervalset()

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

