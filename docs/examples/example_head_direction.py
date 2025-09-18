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
from pynaviz.qt.mainwindow import MainWindow, MainDock
from matplotlib.pyplot import *
from PIL import ImageGrab

def get_action_by_object_name(menu: QMenu, name: str) -> QAction | None:
    for action in menu.actions():
        if action.objectName() == name:
            return action
    return None

def grab_window(win):
    """Capture the visible pixels of the Qt window and return a PIL image."""
    rect = win.frameGeometry()
    top_left = rect.topLeft()
    x, y, w, h = top_left.x(), top_left.y(), rect.width(), rect.height()
    bbox = (x, y, x + w, y + h)
    return ImageGrab.grab(bbox=bbox)

def click_on_item(list_widget, item_number, win, app, frames, durations):
    item = list_widget.item(item_number)
    rect = list_widget.visualItemRect(item)
    pos = rect.center()
    QTest.mouseClick(list_widget.viewport(),
                     Qt.MouseButton.LeftButton,
                     pos=pos)
    QTest.qWait(1000)
    frames.append(grab_window(win))
    durations.append(800)
    return item

def add_dock_widget(list_widget, win, app, frames, durations, item_number=0):
    item = click_on_item(list_widget, item_number, win, app, frames, durations)
    app.processEvents()
    list_widget.itemDoubleClicked.emit(item)
    app.processEvents()
    QTest.qWait(1000)
    frames.append(grab_window(win))
    durations.append(800)

def flash_widget(widget, duration=200):
    orig_style = widget.styleSheet()
    widget.setStyleSheet("border: 2px solid red;")  # highlight
    QTimer.singleShot(duration, lambda: widget.setStyleSheet(orig_style))

def click_on_action(action_text, win, frames, durations):
    frames.append(grab_window(win))
    durations.append(800)
    menu = next((w for w in QApplication.topLevelWidgets()
                 if isinstance(w, QMenu) and w.isVisible()), None)
    # Find the QAction by visible text
    act = next((a for a in menu.actions() if a.text().replace("&", "") == action_text), None)
    if not act:
        raise ValueError(f"Menu item '{action_text}' not found")
    # Click inside the action rectangle
    rect = menu.actionGeometry(act)
    # Schedule dialog handling after a short delay
    QTimer.singleShot(1000, lambda: dialog_selection(win, frames, durations))
    QTest.mouseClick(menu, Qt.MouseButton.LeftButton, pos=rect.center())
    # Wait for dialog to appear and handle it



def dialog_selection(win, frames, durations):
    frames.append(grab_window(win))
    durations.append(800)

    dialog = next((w for w in QApplication.topLevelWidgets()
                   if isinstance(w, QDialog) and w.isVisible()), None)
    if dialog is None:
        raise RuntimeError("No dialog found")
    # Find the OK button
    ok_button = dialog.ok_button
    ok_button.setStyleSheet("border: 2px solid red;")
    frames.append(grab_window(win))
    durations.append(800)
    QTest.mouseClick(ok_button, Qt.MouseButton.LeftButton)


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
    win = MainWindow()
    ctrl_dock = MainDock({"units": units, "manifold": manifold, "video": video}, gui=win)
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
    add_dock_widget(list_widget, win, app, frames, durations, item_number=0) # units
    add_dock_widget(list_widget, win, app, frames, durations, item_number=1) # manifold
    add_dock_widget(list_widget, win, app, frames, durations, item_number=2) # video

    # --- Move docks ---
    dock_widgets = win.findChildren(QDockWidget)
    dock = dock_widgets[-1]  # last added dock (video)
    title_bar = dock.titleBarWidget()
    start = title_bar.mapToGlobal(title_bar.rect().center())
    end = start + QPoint(500, 0)
    QTest.mousePress(title_bar, Qt.MouseButton.LeftButton,
                     Qt.KeyboardModifier.NoModifier,
                     pos=title_bar.rect().center())

    # Process events so the dock enters drag mode
    QApplication.processEvents()

    # Move to target
    QTest.mouseMove(title_bar, pos=title_bar.mapFromGlobal(end))
    QApplication.processEvents()

    # Release to drop
    QTest.mouseRelease(title_bar, Qt.MouseButton.LeftButton,
                       Qt.KeyboardModifier.NoModifier,
                       pos=title_bar.mapFromGlobal(end))
    QTest.qWait(1000)
    app.processEvents()

    # Resize the dock
    win.resizeDocks([dock], [400], Qt.Orientation.Horizontal)
    QTest.qWait(1000)
    frames.append(grab_window(win))
    durations.append(800)
    app.processEvents()

    # --- Apply metadata actions ---
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


    frames[0].save(
        "example_head_direction.gif",
        save_all=True,
        append_images=frames[1:],
        duration=durations,     # ms per frame
        loop=0            # 0 = loop forever
    )

    sys.exit(0)

if __name__ == "__main__":
    main()

