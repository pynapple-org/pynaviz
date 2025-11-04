from typing import Literal

import numpy as np
import pytest
from PySide6.QtWidgets import QDockWidget

import pynaviz as viz
from pynaviz import (
    IntervalSetWidget,
    TsdFrameWidget,
    TsdTensorWidget,
    TsdWidget,
    TsGroupWidget,
    TsWidget,
)
from pynaviz.qt.mainwindow import VariableDock


@pytest.fixture(scope="function", autouse=True)
def main_window__dock(nap_var, qtbot):
    """
    Set up a MainWindow and a MainDock.

    Set up a MainWindow and populate it with test data:
    - TsdFrame with metadata area, type and channel.
    - TsdTensor of white noise.
    - TsGroup for testing timestamp data
    - IntervalSet for testing intervals
    """
    main_window = viz.qt.mainwindow.MainWindow(nap_var)
    qtbot.addWidget(main_window)
    return main_window, nap_var


def apply_action(
        widget: VariableDock | TsdWidget | TsdFrameWidget | TsdTensorWidget | IntervalSetWidget | TsGroupWidget | TsWidget,
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
        qtbot,
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
    qtbot:
        The qtbot fixture from pytest-qt.
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
        qtbot.wait(1000)  # Wait 1 second

        widget._toggle_play()
        assert widget.playing is False, "not paused correctly"
        print("\ncurrent_time after playing: ", widget.ctrl_group.current_time)

    elif action_type == "stop":
        # Stop playback
        widget._stop()


@pytest.mark.parametrize(
    "group_by_kwargs", [
        None,
        dict(metadata_name="group")
    ]
)
@pytest.mark.parametrize(
    "sort_by_kwargs", [
        None,
        dict(metadata_name="channel")
    ]
)
@pytest.mark.parametrize(
    "color_by_kwargs", [
        None,
        dict(metadata_name="channel", cmap_name="rainbow", vmin=0, vmax=100)
    ]
)
@pytest.mark.parametrize(
    "apply_to", [
        ("tsgroup",),
        ("tsdframe",),
        ("tsgroup", "tsdframe")
    ]
)
def test_save_load_layout_tsdframe(apply_to, main_window__dock, color_by_kwargs, group_by_kwargs, sort_by_kwargs,
                                   tmp_path, qtbot):

    # print("\nTesting layout save/load with apply_to:", apply_to,
    #       "group_by:", group_by_kwargs,
    #       "sort_by:", sort_by_kwargs,
    #       "color_by:", color_by_kwargs)

    main_window, variables = main_window__dock
    # add widgets
    widget = None

    for varname in variables.keys():
        dock_widget = main_window.add_dock_widget(variables[varname], [varname])
        if varname in apply_to:
            widget = dock_widget.widget()
            apply_action(widget=widget, action_type="group_by" if group_by_kwargs is not None else None,
                         action_kwargs=group_by_kwargs, qtbot=qtbot)
            apply_action(widget=widget, action_type="sort_by" if sort_by_kwargs is not None else None,
                         action_kwargs=sort_by_kwargs, qtbot=qtbot)
            apply_action(widget=widget, action_type="color_by" if color_by_kwargs is not None else None,
                         action_kwargs=color_by_kwargs, qtbot=qtbot)

    # debug purposes, should not trigger.
    assert widget is not None, "widget not created."

    layout_path = tmp_path / "layout.json"
    layout_dict_orig = main_window._get_layout_dict()
    main_window._save_layout(layout_path)
    main_window.close()


    # load a main window with the same configs.
    # print(layout_path)
    main_window_new = viz.qt.mainwindow.MainWindow(variables=variables, layout_path=layout_path)
    qtbot.addWidget(main_window_new)

    layout_dict_new = main_window_new._get_layout_dict()
    # print("\nold", layout_dict_orig)
    # print("new", layout_dict_new)
    # discard geometry bytes, the initial window position may differ
    layout_dict_new.pop("geometry_b64")
    layout_dict_orig.pop("geometry_b64")
    # check dict
    assert layout_dict_orig == layout_dict_new

    main_window_new.close()



def verify_layout_structure(original_window, restored_window):
    """Verify that layouts have identical widget structure"""

    # Compare dock widget counts and types
    orig_docks = original_window.findChildren(QDockWidget)
    new_docks = restored_window.findChildren(QDockWidget)

    assert len(orig_docks) == len(new_docks), "Different number of dock widgets"

    # Compare widget types and object names
    orig_types = sorted([dock.objectName() for dock in orig_docks])
    new_types = sorted([dock.objectName() for dock in new_docks])
    assert orig_types == new_types, "Different widget types"

    # Compare dock areas (where widgets are positioned)
    for orig_dock in orig_docks:
        orig_area = original_window.dockWidgetArea(orig_dock)
        # Find corresponding dock in new window
        new_dock = restored_window.findChild(QDockWidget, orig_dock.objectName())
        new_area = restored_window.dockWidgetArea(new_dock)
        assert orig_area == new_area, f"Widget {orig_dock.objectName()} in wrong area"


def verify_layout_geometry(original_window, restored_window):
    """Verify relative positioning and sizing"""

    for orig_dock in original_window.findChildren(QDockWidget):

        if orig_dock.objectName() != "VariablesDock":

            new_dock = restored_window.findChild(QDockWidget, orig_dock.objectName())

            # Compare relative sizes (not absolute pixels)
            orig_size = orig_dock.size()
            new_size = new_dock.size()

            # Allow some tolerance for window manager differences
            assert abs(orig_size.width() - new_size.width()) < 10
            assert abs(orig_size.height() - new_size.height()) < 10

            # Compare dock widget properties
            assert orig_dock.isFloating() == new_dock.isFloating()
            assert orig_dock.isVisible() == new_dock.isVisible()


@pytest.mark.parametrize(
    "group_by_kwargs", [None, dict(metadata_name="group")]
)
@pytest.mark.parametrize(
    "sort_by_kwargs", [None, dict(metadata_name="channel")]
)
@pytest.mark.parametrize(
    "color_by_kwargs", [None, dict(metadata_name="channel", cmap_name="rainbow", vmin=5, vmax=80)]
)
@pytest.mark.parametrize("apply_to", [("tsgroup",), ("tsdframe",), ("tsgroup", "tsdframe")])
def test_save_load_layout_tsdframe_screenshots(apply_to, main_window__dock, color_by_kwargs, group_by_kwargs,
                                               sort_by_kwargs, tmp_path, qtbot):
    main_window, variables = main_window__dock
    # add widgets
    widget = None

    for varname in variables.keys():
        dock_widget = main_window.add_dock_widget(variables[varname], [varname])
        if varname in apply_to:
            widget = dock_widget.widget()
            apply_action(widget=widget, action_type="group_by" if group_by_kwargs is not None else None,
                         action_kwargs=group_by_kwargs, qtbot=qtbot)
            apply_action(widget=widget, action_type="sort_by" if sort_by_kwargs is not None else None,
                         action_kwargs=sort_by_kwargs, qtbot=qtbot)
            apply_action(widget=widget, action_type="color_by" if color_by_kwargs is not None else None,
                         action_kwargs=color_by_kwargs, qtbot=qtbot)
    # debug purposes, should not trigger.
    assert widget is not None, "widget not created."

    layout_path = tmp_path / "layout.json"
    main_window._save_layout(layout_path)


    # Take screenshots
    orig_screenshots = {}
    count = 0
    for d in main_window.findChildren(QDockWidget):
        if d.objectName() != "VariablesDock":
            base_plot = d.widget().plot
            base_plot.renderer.render(base_plot.scene, base_plot.camera)
            orig_screenshots[count, base_plot.__class__.__name__] = base_plot.renderer.snapshot()
            count += 1



    # load a main window with the same configs.
    main_window_new = viz.qt.mainwindow.MainWindow(variables, layout_path=layout_path)
    qtbot.addWidget(main_window_new)

    # Take screenshots
    new_screenshots = {}
    count = 0
    for d in main_window_new.findChildren(QDockWidget):
        if d.objectName() != "VariablesDock":
            base_plot = d.widget().plot
            base_plot.renderer.render(base_plot.scene, base_plot.camera)
            new_screenshots[count, base_plot.__class__.__name__] = base_plot.renderer.snapshot()
            count += 1

    for k, img in orig_screenshots.items():
        np.testing.assert_allclose(img, new_screenshots[k], atol=1)

    # make sure there are no extra widgets
    # assert len(orig_screenshots) == len(new_screenshots)

    # verify the qt layout struct
    verify_layout_structure(main_window, main_window_new)
    verify_layout_geometry(main_window, main_window_new)

    main_window.close()
    main_window_new.close()
