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

def main():
    global app
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    win = MainWindow()
    ctrl_dock = MainDock(get_vars(), gui=win)
    win.show()

    frames = []
    timer = QTimer()

    def capture_frame():
        print("Capturing frame…")
        app.processEvents()
        pixmap = win.grab()
        img = pixmap.toImage()
        ptr = img.bits()
        ptr.setsize(img.sizeInBytes())
        arr = np.array(ptr).reshape(img.height(), img.width(), 4)
        frames.append(arr.copy())
        print(f"Captured {len(frames)} frames")

    # Capture at ~10 FPS
    timer.timeout.connect(capture_frame)
    timer.start(100)


    def run_demo():
        # Give time for the window to show
        QTest.qWait(500)

        # print("Clicking button1…")
        ctrl_dock.add_dock_widget("tsd1")

        list_widget = ctrl_dock.listWidget
        index = list_widget.model().index(0, 0)  # first row
        rect = list_widget.visualRect(index)
        pos = rect.center()
        QTest.mouseDClick(list_widget.viewport(), Qt.MouseButton.LeftButton, pos=pos)

        app.processEvents()
        QTest.qWait(5000)


        # Stop recording
        timer.stop()
        imageio.mimsave("demo.gif", frames, duration=0.2)
        print("Saved demo.gif")
        app.quit()

    QTimer.singleShot(5000, run_demo)

    print(len(frames))

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
