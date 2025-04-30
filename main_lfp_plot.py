"""
Test script
"""
import numpy as np
import os
import pynapple as nap
from PyQt6.QtWidgets import QApplication
import pynaviz as viz



tsdframe = nap.misc.load_eeg(
    os.path.expanduser("~/Dropbox/A2929-200711/A2929-200711.eeg"),
    n_channels=16,
    frequency=1250
    )
tsdframe.group = np.hstack((np.zeros(10), np.ones(6)))
tsdframe.channel = np.hstack((np.arange(10), np.arange(10,16)))

app = QApplication([])

v = viz.TsdFrameWidget(tsdframe)
# v.plot.sort_by("channel")
# v.plot.group_by("group")
# v.plot.plot_x_vs_y(0, 1)
# v.plot.plot_time_series()
v.show()

if __name__ == "__main__":
    app.exit(app.exec())
