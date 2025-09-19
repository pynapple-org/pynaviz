"""
Test script
"""
import sys
from pathlib import Path

import imageio
import numpy as np
import pynapple as nap
import requests, os
from PyQt6.QtGui import QAction
from one.api import ONE

import pynaviz as viz
from PyQt6.QtCore import QTimer, Qt, QPoint
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication, QDockWidget, QMenu, QDialog
from pynaviz.qt.mainwindow import MainWindow, MainDock
from matplotlib.pyplot import *
from PIL import ImageGrab, Image
from utils import grab_window, click_on_item, add_dock_widget, move_and_resize_dock, save_gif


def main():
    # Load IBL session
    one = ONE()
    eid = "ebce500b-c530-47de-8cb1-963c552703ea"

    # Videos
    ibl_path = Path(os.path.expanduser("~/Downloads/ONE/"))
    if not ibl_path.exists():
        print("Please set the path to your IBL data directory in the variable `ibl_path`.")
        quit()
    vars = {}
    for label in ["left", "body", "right"]:
        video_path = (
                ibl_path
                / f"openalyx.internationalbrainlab.org/churchlandlab_ucla/Subjects/MFD_09/2023-10-19/001/raw_video_data/_iblrig_{label}Camera.raw.mp4"
        )
        if not video_path.exists():
            one.load_dataset(eid, f"*{label}Camera.raw*", collection="raw_video_data")
        times = one.load_object(eid, f"{label}Camera", collection="alf", attribute=["times*"])["times"]
        vars[label] = viz.VideoWidget(video_path, t=times)

    # Events and intervals
    timings = one.load_object(eid, "trials", collection="alf")
    licks = nap.Ts(one.load_object(eid, "licks", collection="alf")["times"])
    reactions = nap.Ts(timings["firstMovement_times"])
    trials = nap.IntervalSet(timings["intervals"])

    vars["licks"] = licks
    vars["trials"] = trials

    # Initialize the application and main window
    app = QApplication.instance() or QApplication(sys.argv)
    win = MainWindow()
    ctrl_dock = MainDock(vars, gui=win)
    win.show()

    # Make sure the window is shown and painted
    app.processEvents()
    QTest.qWaitForWindowExposed(win)    # waits until the window is mapped
    app.processEvents()
    QTest.qWait(1000)

    frames = []
    durations = []

    # --- Action 1: initial view ---
    frames.append(grab_window(win))
    durations.append(800)

    list_widget = ctrl_dock.listWidget

    # --- Add docks ---
    add_dock_widget(list_widget, win, app, frames, durations, item_number=4)
    add_dock_widget(list_widget, win, app, frames, durations, item_number=0)
    add_dock_widget(list_widget, win, app, frames, durations, item_number=3)


    # --- Resize and move the video dock ---
    move_and_resize_dock(win, app, frames, durations, move_offset=QPoint(500, 0), resize_width=500)

    # --- ACtion : jump to the first trial ---
    dock_widgets = win.findChildren(QDockWidget)
    iset_widget = dock_widgets[1].widget()
    button = iset_widget.button_container.right_jump_button

    orig_style = button.styleSheet()
    button.setStyleSheet("border: 2px solid red;")  # highlight
    app.processEvents()
    frames.append(grab_window(win))
    durations.append(800)
    QTest.qWait(1000)
    button.setStyleSheet(orig_style)
    app.processEvents()

    QTest.mouseClick(button, Qt.MouseButton.LeftButton)
    frames.append(grab_window(win))
    durations.append(800)
    app.processEvents()
    QTest.qWait(1000)
    frames.append(grab_window(win))
    durations.append(800)


    # # --- Action 4: play the animation ---
    duration_ms = 6000  # total recording time
    interval_ms = 50  # grab frame every 200 ms
    num_frames = duration_ms // interval_ms

    QTest.mouseClick(ctrl_dock.playPauseBtn, Qt.MouseButton.LeftButton)
    app.processEvents()

    for _ in range(num_frames):
        QTest.qWait(interval_ms)  # wait 25 ms
        frames.append(grab_window(win))  # grab frame
        durations.append(interval_ms)
        app.processEvents()

    # --- Pause the animation ---
    QTest.mouseClick(ctrl_dock.playPauseBtn, Qt.MouseButton.LeftButton)
    app.processEvents()
    frames.append(grab_window(win))  # grab frame
    durations.append(800)

    save_gif(frames, durations, "example_trials.gif")

    sys.exit(0)


if __name__ == "__main__":
    main()

