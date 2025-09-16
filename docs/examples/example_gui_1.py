"""
Test script
"""
import sys

import imageio
import numpy as np
import pynapple as nap
from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication
from pynaviz.qt.mainwindow import MainWindow, MainDock
from matplotlib.pyplot import *
from PIL import ImageGrab

def get_vars():
    tsd1 = nap.Tsd(t=np.arange(1000), d=np.cos(np.arange(1000) * 0.1))
    tsd2 = nap.Tsd(t=np.arange(1000), d=np.cos(np.arange(1000) * 0.1))
    tsd3 = nap.Tsd(t=np.arange(1000), d=np.arange(1000))

    tsg = nap.TsGroup({
        i:nap.Ts(
            t=np.sort(np.random.uniform(0, 1000, 100*(  i+1)))
        ) for i in range(10)

    })
    tsdframe = nap.TsdFrame(
        t=np.arange(1000)/30,
        d=np.random.randn(1000, 10))
    tsdtensor = nap.TsdTensor(t=np.arange(10000)/30, d=np.random.randn(10000, 10, 10))

    return {"tsd1": tsd1, "tsd2": tsd2, "tsd3": tsd3, "tsg": tsg, "tsdframe": tsdframe, "tsdtensor": tsdtensor}

def grab_window(win):
    """Capture the visible pixels of the Qt window and return a PIL image."""
    rect = win.frameGeometry()
    top_left = rect.topLeft()
    x, y, w, h = top_left.x(), top_left.y(), rect.width(), rect.height()
    bbox = (x, y, x + w, y + h)
    return ImageGrab.grab(bbox=bbox)

def main():
    app = QApplication.instance() or QApplication(sys.argv)
    win = MainWindow()
    ctrl_dock = MainDock(get_vars(), gui=win)
    win.show()

    # Make sure the window is shown and painted
    app.processEvents()
    QTest.qWaitForWindowExposed(win)    # waits until the window is mapped
    app.processEvents()

    frames = []
    durations = []

    # --- Action 1: initial view ---
    frames.append(grab_window(win))
    durations.append(800)

    for n in [0, 3]:
        # # --- Action 2: double-click the first list item ---
        list_widget = ctrl_dock.listWidget
        item = list_widget.item(n)
        rect = list_widget.visualItemRect(item)
        pos = rect.center()
        QTest.mouseDClick(list_widget.viewport(),
                          Qt.MouseButton.LeftButton,
                          pos=pos)
        app.processEvents()
        QTest.qWait(1000)
        frames.append(grab_window(win))
        durations.append(800)

        # --- Action 3: emit the signal ---
        list_widget.itemDoubleClicked.emit(item)
        app.processEvents()
        QTest.qWait(1000)
        frames.append(grab_window(win))
        durations.append(800)

    # # --- Action 4: play the animation ---
    duration_ms = 2000  # total recording time
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
        "demo_actions.gif",
        save_all=True,
        append_images=frames[1:],
        duration=durations,     # ms per frame
        loop=0            # 0 = loop forever
    )

    sys.exit(0)

if __name__ == "__main__":
    main()

