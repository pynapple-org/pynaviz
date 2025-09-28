import json
import os
import sys
import time
from pathlib import Path
from typing import Literal
from unittest.mock import patch

import numpy as np
import pynapple as nap
import pytest
from PyQt6.QtGui import QImage
from PyQt6.QtWidgets import QApplication, QDockWidget

import pynaviz as viz
from pynaviz import (
    IntervalSetWidget,
    TsdFrameWidget,
    TsdTensorWidget,
    TsdWidget,
    TsGroupWidget,
)
from pynaviz.qt.mainwindow import MainDock


@pytest.fixture(autouse=True)
def mock_qt_geometry_for_headless():
    """Mock Qt widget geometry methods using real values from local testing.

    Values based on actual widget dimensions:
    - Main widgets: 640x480
    - Dock widgets: 127x480 or 640x480
    - All widgets positioned at (0,0)
    """
    if os.environ.get('QT_QPA_PLATFORM') == 'offscreen' or os.environ.get('CI'):
        with patch('PyQt6.QtWidgets.QWidget.width', return_value=640), \
                patch('PyQt6.QtWidgets.QWidget.height', return_value=480), \
                patch('PyQt6.QtWidgets.QWidget.x', return_value=0), \
                patch('PyQt6.QtWidgets.QWidget.y', return_value=0), \
                patch('PyQt6.QtWidgets.QMainWindow.saveState', return_value=b'mock_window_state'), \
                patch('PyQt6.QtWidgets.QMainWindow.restoreState', return_value=True):
            yield
    else:
        yield

def capture_widget_geometry(widget, widget_name="widget"):
    """Capture real widget geometry values when running locally"""
    if os.environ.get('QT_QPA_PLATFORM') != 'offscreen' and not os.environ.get('CI'):
        geometry_data = {
            'widget_name': widget_name,
            'width': widget.width(),
            'height': widget.height(),
            'x': widget.x(),
            'y': widget.y(),
            'size': {'width': widget.size().width(), 'height': widget.size().height()},
            'rect': {
                'x': widget.rect().x(),
                'y': widget.rect().y(),
                'width': widget.rect().width(),
                'height': widget.rect().height()
            }
        }

        # Save to file for later use in CI
        geom_dir = Path(__file__).parent / "geom_save"
        geom_dir.mkdir(exist_ok=True)
        geometry_file = geom_dir / f'{widget_name}_test_geometry_values.json'

        if geometry_file.exists():
            with open(geometry_file, 'r') as f:
                all_data = json.load(f)
        else:
            all_data = {}

        all_data[widget_name] = geometry_data

        with open(geometry_file, 'w') as f:
            json.dump(all_data, f, indent=2)

        print(f"Captured geometry for {widget_name}: {geometry_data}")
        return geometry_data
    return None

def pixmap_to_array(pixmap):
    """Convert QPixmap to numpy array"""
    # Convert to QImage first
    qimage = pixmap.toImage()

    # Convert to RGB format if needed
    qimage = qimage.convertToFormat(QImage.Format.Format_RGB888)

    # Get image dimensions
    width = qimage.width()
    height = qimage.height()

    # Get raw data and convert to numpy array
    ptr = qimage.bits()
    ptr.setsize(height * width * 3)
    arr = np.frombuffer(ptr, np.uint8).reshape((height, width, 3))

    return arr


@pytest.fixture()
def nap_vars():
    tsdframe = nap.TsdFrame(
        t=np.arange(1000) / 30,
        d=np.random.randn(1000, 10),
        metadata={
            "area": ["pfc"] * 4 + ["ppc"] * 6,
            "type": ["exc", "inh"] * 5,
            "channel": np.random.permutation(np.arange(10))
        }
    )

    tsdtensor = nap.TsdTensor(
        t=np.arange(10000) / 30,
        d=np.random.randn(10000, 10, 10)
    )

    # Create TsGroup with some spikes
    spike_times = {
        0: nap.Ts(t=np.sort(np.random.uniform(0, 30, 100))),
        1: nap.Ts(t=np.sort(np.random.uniform(0, 30, 150))),
        2: nap.Ts(t=np.sort(np.random.uniform(0, 30, 80))),
    }
    tsgroup = nap.TsGroup(
        spike_times,
        metadata=dict(area=['pfc', 'pfc', 'ppc'], type=['exc', 'inh', 'exc'], channel=[2, 1, 3])
    )

    # Create IntervalSet
    interval_set = nap.IntervalSet(
        start=np.array([0, 10, 20]),
        end=np.array([5, 15, 25])
    )

    variables = {
        'tsdframe': tsdframe,
        'tsdtensor': tsdtensor,
        'tsgroup': tsgroup,
        'interval_set': interval_set
    }
    return variables

@pytest.fixture
def app__main_window__dock(nap_vars):
    """
    Set up an app, a MainWindow and a MainDock.

    Set up a MainWindow and populate it with test data:
    - TsdFrame with metadata area, type and channel.
    - TsdTensor of white noise.
    - TsGroup for testing timestamp data
    - IntervalSet for testing intervals
    """
    # Set up Qt application (ensure it's headless for testing)
    if not QApplication.instance():
        app = QApplication(sys.argv)
        # Make headless for testing
        app.setQuitOnLastWindowClosed(False)
    else:
        app = QApplication.instance()

    main_window = viz.qt.mainwindow.MainWindow()

    dock = viz.qt.mainwindow.MainDock(nap_vars, main_window)

    return app, main_window, dock, nap_vars


def apply_action(
        widget: MainDock | TsdWidget | TsdFrameWidget | TsdTensorWidget | IntervalSetWidget | TsGroupWidget,
        action_type: Literal[
            "group_by",
            "sort_by",
            "color_by",
            "skip_forward",
            "skip_backward",
            "set_time",
            "play_pause",
            "stop"
        ],
        action_kwargs: dict | None,
        app: QApplication,
):
    """
    Apply a specific action to the dock/widgets.

    Parameters:
    -----------
    widget:
        The widget to apply the action to.
    action_type :
        (action_type, widget, action_kwargs) or None for no action.
        Action types:
        - "group_by"
        - "sort_by"
        - "color_by"
        - "skip_forward"
        - "skip_backward"
        - "play_pause"
        - "stop"
        - "set_time"
    action_kwargs:
        The kwargs for the action method.
    app:
        The QtApplication used for simulating playing the gui.
    """
    if action_type is None:
        return
    action_kwargs = action_kwargs or {}

    if action_type not in ["play_pause", "stop", "set_time"]:
        # add the underscore for private method, clear unnecessary kwargs
        if action_type in ["skip_forward", "skip_backward"]:
            action_type = "_" + action_type
            if action_kwargs:
                print("No kwargs needed for skip_forward or skip_backward")
            action_kwargs = {}
        if hasattr(widget.plot, action_type):
            # group_by, sort_by, color_by are action of _BasePlot
            action = getattr(widget.plot, action_type)
            action(**action_kwargs)
        elif hasattr(widget, action_type):
            # the rest of the actions are of dock
            action = getattr(widget, action_type)
            action(**action_kwargs)
        else:
            raise AttributeError(f"{widget} has no action {action_type}.")

    elif action_type == "set_time":
        time_to_set = action_kwargs.get("time", None)
        if time_to_set is None:
            raise ValueError("'set_time' action requires a 'time' kwarg.")
        # Set specific time
        widget.ctrl_group.set_interval(action_kwargs["time"], None)
        widget._update_time_label(widget.ctrl_group.current_time)

    elif action_type == "play_pause":
        # make sure that we are in the correct config
        widget.playing = False
        widget._toggle_play()
        # for debugging purposes
        assert widget.playing is True, "not toggled correctly"

        # Run event loop for specified duration
        start_time = time.time()
        duration = 1
        while time.time() - start_time < duration:
            app.processEvents()
            time.sleep(0.01)

        widget._toggle_play()
        assert widget.playing is False, "not paused correctly"
        print("\ncurrent_time after playing: ", widget.ctrl_group.current_time)

    elif action_type == "stop":
        # Stop playback
        widget._stop()

@pytest.mark.parametrize(
    "group_by_kwargs", [None, dict(metadata_name="area")]
)
@pytest.mark.parametrize(
    "sort_by_kwargs", [None, dict(metadata_name="channel")]
)
@pytest.mark.parametrize(
    "color_by_kwargs", [None, dict(metadata_name="channel", cmap_name="rainbow", vmin=5, vmax=80)]
)
@pytest.mark.parametrize("apply_to", [("tsgroup",), ("tsdframe", ), ("tsgroup", "tsdframe")])
def test_save_load_layout_tsdframe(apply_to, app__main_window__dock, color_by_kwargs, group_by_kwargs, sort_by_kwargs, tmp_path):
    app, main_window, dock, variables = app__main_window__dock
    # add widgets
    widget = None
    for varname in variables.keys():
        dock_widget = dock.add_dock_widget(varname)
        if varname in apply_to:
            widget = dock_widget.widget()
            apply_action(widget=widget, action_type="group_by" if group_by_kwargs is not None else None,
                         action_kwargs=group_by_kwargs, app=app)
            apply_action(widget=widget, action_type="sort_by" if sort_by_kwargs is not None else None,
                         action_kwargs=sort_by_kwargs, app=app)
            apply_action(widget=widget, action_type="color_by" if color_by_kwargs is not None else None,
                         action_kwargs=color_by_kwargs, app=app)

    # debug purposes, should not trigger.
    assert widget is not None, "widget not created."

    layout_path = tmp_path / "layout.json"
    layout_dict_orig = dock._get_layout_dict()
    dock.save_layout(layout_path)

    # load a main window with the same configs.
    main_window_new = viz.qt.mainwindow.MainWindow()
    dock_new = viz.qt.mainwindow.MainDock(variables, main_window_new, layout_path=layout_path)
    layout_dict_new = dock_new._get_layout_dict()
    print("\nold", layout_dict_orig)
    print("new", layout_dict_new)
    # discard geometry bytes, the initial window position may differ
    layout_dict_new.pop("geometry_b64")
    layout_dict_orig.pop("geometry_b64")
    # check dict
    assert layout_dict_orig == layout_dict_new

    # # For each widget you create, capture its geometry (this is use)
    # for i, dock_widget in enumerate(main_window.findChildren(QDockWidget)):
    #     capture_widget_geometry(dock_widget, f"dock_widget_{i}")


@pytest.mark.parametrize(
    "group_by_kwargs", [None, dict(metadata_name="area")]
)
@pytest.mark.parametrize(
    "sort_by_kwargs", [None, dict(metadata_name="channel")]
)
@pytest.mark.parametrize(
    "color_by_kwargs", [None, dict(metadata_name="channel", cmap_name="rainbow", vmin=5, vmax=80)]
)
@pytest.mark.parametrize("apply_to", [("tsgroup",), ("tsdframe", ), ("tsgroup", "tsdframe")])
def test_save_load_layout_tsdframe_screenshots(apply_to, app__main_window__dock, color_by_kwargs, group_by_kwargs, sort_by_kwargs, tmp_path):
    app, main_window, dock, variables = app__main_window__dock
    # add widgets
    widget = None
    for varname in variables.keys():
        dock_widget = dock.add_dock_widget(varname)
        if varname in apply_to:
            widget = dock_widget.widget()
            apply_action(widget=widget, action_type="group_by" if group_by_kwargs is not None else None,
                         action_kwargs=group_by_kwargs, app=app)
            apply_action(widget=widget, action_type="sort_by" if sort_by_kwargs is not None else None,
                         action_kwargs=sort_by_kwargs, app=app)
            apply_action(widget=widget, action_type="color_by" if color_by_kwargs is not None else None,
                         action_kwargs=color_by_kwargs, app=app)
    # debug purposes, should not trigger.
    assert widget is not None, "widget not created."

    layout_path = tmp_path / "layout.json"
    dock.save_layout(layout_path)

    # Take screenshots
    orig_screenshots = {}
    for i, d in enumerate(dock.gui.findChildren(QDockWidget)):
        if d.objectName() != "MainDock":
            base_plot = d.widget().plot
            base_plot.renderer.render(base_plot.scene, base_plot.camera)
            orig_screenshots[i, base_plot.__class__.__name__] = base_plot.renderer.snapshot()

    # load a main window with the same configs.
    main_window_new = viz.qt.mainwindow.MainWindow()
    dock_new = viz.qt.mainwindow.MainDock(variables, main_window_new, layout_path=layout_path)
    # Take screenshots
    new_screenshots = {}
    for i, d in enumerate(dock_new.gui.findChildren(QDockWidget)):
        if d.objectName() != "MainDock":
            base_plot = d.widget().plot
            base_plot.renderer.render(base_plot.scene, base_plot.camera)
            new_screenshots[i, base_plot.__class__.__name__] = base_plot.renderer.snapshot()

    for k, img in orig_screenshots.items():
        np.testing.assert_allclose(img, new_screenshots[k], atol=1)
    # make sure there are no extra widgets
    assert len(orig_screenshots) == len(new_screenshots)
    # test layout
    main_orig = pixmap_to_array(main_window.grab())
    main_new = pixmap_to_array(main_window_new.grab())
    np.testing.assert_allclose(main_orig, main_new, atol=1)
