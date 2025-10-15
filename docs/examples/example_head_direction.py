"""
Test script
"""
import sys

import imageio
import numpy as np
import pynapple as nap
import requests, os
from PyQt6.QtGui import QAction

import pynaviz as viz
from PyQt6.QtCore import QTimer, Qt, QPoint
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication, QDockWidget, QMenu, QDialog
from pynaviz.qt.mainwindow import MainWindow
from matplotlib.pyplot import *
from PIL import ImageGrab
from utils import grab_window, click_on_item, add_dock_widget, move_and_resize_dock, save_gif





def main():
    # Stream data
    nwb_file = "A5044-240404A_wake.nwb"
    avi_file = "A5044-240404A_wake.avi"

    files = os.listdir(".")
    if nwb_file not in files:
        url = "https://osf.io/um4nb/download"
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(nwb_file, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):  # Stream in chunks
                    f.write(chunk)
    if avi_file not in files:
        url = "https://osf.io/gyu2h/download"
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(avi_file, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):  # Stream in chunks
                    f.write(chunk)


    # Load data
    data = nap.load_file("A5044-240404A_wake.nwb")
    units = data["units"]
    manifold = data["manifold"]
    video = viz.VideoWidget("A5044-240404A_wake.avi")

    # Initialize the application and main window
    app = QApplication.instance() or QApplication(sys.argv)
    win = MainWindow({"units": units, "manifold": manifold, "video": video})
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
    add_dock_widget(tree_widget, win, app, frames, durations, item_number=0) # units
    add_dock_widget(tree_widget, win, app, frames, durations, item_number=1) # manifold
    add_dock_widget(tree_widget, win, app, frames, durations, item_number=2) # video

    # --- Resize and move the video dock ---
    move_and_resize_dock(win, app, frames, durations, move_offset=QPoint(500, 0), resize_width=400)

    # --- Toggle tree widget visibility ---
    QTest.mouseClick(win.variable_dock.handle, Qt.MouseButton.LeftButton)
    app.processEvents()
    frames.append(grab_window(win))  # grab frame
    durations.append(800)

    # --- Apply metadata actions ---
    dock_widgets = win.findChildren(QDockWidget)
    plots = {i:dock.widget().plot for i, dock in enumerate(dock_widgets[1:])}

    # Apply Sort by action
    # button = dock_widgets[0].button_container.action_button
    # QTimer.singleShot(1000, lambda: click_on_action("Sort by", win, frames, durations))
    # QTest.mouseClick(button, Qt.MouseButton.LeftButton)

    plots[0].sort_by("order")
    QTest.qWait(1000)
    frames.append(grab_window(win))
    durations.append(800)
    app.processEvents()

    plots[0].color_by("peak", "hsv")
    QTest.qWait(1000)
    frames.append(grab_window(win))
    durations.append(800)
    app.processEvents()

    plots[1].plot_x_vs_y(0, 1, markersize=20.0)
    QTest.qWait(1000)
    frames.append(grab_window(win))
    durations.append(800)
    app.processEvents()

    # # --- Action 4: play the animation ---
    duration_ms = 4000  # total recording time
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

    save_gif(frames, durations, "example_head_direction.gif")

    save_gif(running_frames, running_durations, "example_head_direction_short.gif")

    sys.exit(0)

if __name__ == "__main__":
    main()

