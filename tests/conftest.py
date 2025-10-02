"""
Testing configurations for the pynaviz library.

This module contains test fixtures required to set up
and verify the functionality of modules in the pynaviz library.
"""
import os
import sys

import numpy as np
import pytest

from pynaviz.events import SyncEvent

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import config

# ---------- Fixtures ----------

@pytest.fixture
def dummy_tsd():
    return config.TsdConfig.get_data()

@pytest.fixture
def dummy_ts():
    return config.TsConfig.get_data()

@pytest.fixture
def dummy_tsdframe():
    return config.TsdFrameConfig.get_data()

@pytest.fixture
def dummy_intervalset():
    return config.IntervalSetConfig.get_data()

@pytest.fixture
def dummy_tsgroup():
    return config.TsGroupConfig.get_data()

@pytest.fixture
def dummy_tsdtensor():
    return config.TsdTensorConfig.get_data()

@pytest.fixture
def nap_var():
    return {
        "tsdframe": config.TsdFrameConfig.get_data(),
        "tsd": config.TsdConfig.get_data(),
        "ts": config.TsConfig.get_data(),
        "interval_set": config.IntervalSetConfig.get_data(),
        "tsgroup": config.TsGroupConfig.get_data(),
        "tsdtensor": config.TsdTensorConfig.get_data(),
    }

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

