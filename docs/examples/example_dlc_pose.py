"""

"""
import sys
from pathlib import Path

import imageio
import numpy as np
import pandas as pd
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
    # Load data
    video_path = "m3v1mp4.mp4"

    df = pd.read_hdf("m3v1mp4DLC_Resnet50_openfieldOct30shuffle1_snapshot_best-70.h5")
    df.columns = [f"{bodypart}_{coord}" for _, bodypart, coord in df.columns]
    df = df[[c for c in df.columns if c.endswith(("_x", "_y"))]]
    y_col = [c for c in df.columns if c.endswith("_y")]
    df[y_col] = df[y_col]*-1 + 480 # Flipping y axis
    skeleton = nap.TsdFrame(t=df.index.values/30, d=df.values, columns=df.columns)

    # Initialize the application and main window
    app = QApplication.instance() or QApplication(sys.argv)
    win = MainWindow({"video":video_path, "pose":skeleton})
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

    # --- Resize and move the video dock ---
    move_and_resize_dock(win, app, frames, durations, move_offset=QPoint(500, 0), resize_width=500)

    # --- Toggle tree widget visibility ---
    QTest.mouseClick(win.variable_dock.handle, Qt.MouseButton.LeftButton)
    app.processEvents()
    frames.append(grab_window(win))  # grab frame
    durations.append(800)

    # --- Superimpose the skeleton on the video ---
    dock_widgets = win.findChildren(QDockWidget)
    # widget_menu = dock_widgets[0].widget().button_container
    plots = {i:dock.widget().plot for i, dock in enumerate(dock_widgets[1:])}
    plots[0].superpose_points(skeleton, color="red")
    app.processEvents()
    QTest.qWait(100)
    frames.append(grab_window(win))  # grab frame
    durations.append(800)

    # --- Plot the trajectory of the nose ---
    plots[1].plot_x_vs_y("snout_x", "snout_y", markersize=25)
    app.processEvents()
    QTest.qWait(100)
    frames.append(grab_window(win))  # grab frame
    durations.append(800)

    # # --- Action 4: play the animation ---
    duration_ms = 5000  # total recording time
    interval_ms = 50  # grab frame every 200 ms
    num_frames = duration_ms // interval_ms

    QTest.mouseClick(win.playPauseBtn, Qt.MouseButton.LeftButton)
    app.processEvents()

    running_frames = []
    running_durations = []

     # Grab frames while the animation is playing
    for _ in range(num_frames):
        QTest.qWait(interval_ms)  # wait 25 ms
        frames.append(grab_window(win))  # grab frame
        durations.append(interval_ms)
        app.processEvents()
        # adding for special gifs
        running_frames.append(frames[-1])
        running_durations.append(interval_ms)

    # --- Pause the animation ---
    QTest.mouseClick(win.playPauseBtn, Qt.MouseButton.LeftButton)
    app.processEvents()
    frames.append(grab_window(win))  # grab frame
    durations.append(800)

    save_gif(frames, durations, "example_dlc_pose.gif")

    # Save only the frames during the playing
    save_gif(running_frames, running_durations, "example_dlc_pose_short.gif")

    sys.exit(0)

if __name__ == "__main__":
    main()