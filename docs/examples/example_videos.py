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
from pynaviz.qt.mainwindow import MainWindow
from matplotlib.pyplot import *
from PIL import ImageGrab
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
    videos = {}
    for label in ["left", "body", "right"]:
        video_path = (
                ibl_path
                / f"openalyx.internationalbrainlab.org/churchlandlab_ucla/Subjects/MFD_09/2023-10-19/001/raw_video_data/_iblrig_{label}Camera.raw.mp4"
        )
        if not video_path.exists():
            one.load_dataset(eid, f"*{label}Camera.raw*", collection="raw_video_data")
        times = one.load_object(eid, f"{label}Camera", collection="alf", attribute=["times*"])["times"]
        # The videos seem to start at 5 seconds. Removing artificially 5 seconds for the demo
        times = times - 5
        videos[label] = viz.VideoWidget(video_path, t=times)


    # Initialize the application and main window
    app = QApplication.instance() or QApplication(sys.argv)
    win = MainWindow(videos)
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

    tree_widget = win.variable_dock.treeWidget

    # --- Add docks ---
    add_dock_widget(tree_widget, win, app, frames, durations, item_number=0)
    add_dock_widget(tree_widget, win, app, frames, durations, item_number=1)
    add_dock_widget(tree_widget, win, app, frames, durations, item_number=2)

    # --- Resize and move the video dock ---
    move_and_resize_dock(win, app, frames, durations, move_offset=QPoint(500, 0), resize_width=500)


    # # --- Action 4: play the animation ---
    duration_ms = 5000  # total recording time
    interval_ms = 50  # grab frame every 200 ms
    num_frames = duration_ms // interval_ms

    QTest.mouseClick(win.playPauseBtn, Qt.MouseButton.LeftButton)
    app.processEvents()

    running_frames = []
    running_durations = []

    for _ in range(num_frames):
        QTest.qWait(interval_ms)  # wait 25 ms
        frames.append(grab_window(win))  # grab frame
        durations.append(interval_ms)
        app.processEvents()
        running_frames.append(frames[-1])
        running_durations.append(interval_ms)

    # --- Pause the animation ---
    QTest.mouseClick(win.playPauseBtn, Qt.MouseButton.LeftButton)
    app.processEvents()
    frames.append(grab_window(win))  # grab frame
    durations.append(800)

    save_gif(frames, durations, "example_videos.gif")

    save_gif(running_frames, running_durations, "example_videos_short.gif")

    sys.exit(0)

if __name__ == "__main__":
    main()

