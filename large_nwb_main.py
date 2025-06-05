"""
Test script
"""
import numpy as np
import pynapple as nap
from PyQt6.QtWidgets import QApplication
import pynaviz as viz
from PyQt6.QtCore import QTimer

path = "/Users/ebalzani/Downloads/Phase_0_exported.nwb"

nwb = nap.load_file(path)
data = nwb["ElectricalSeries"].restrict(nap.IntervalSet(0, 10))
data.channels = np.arange(1024)
print(nwb)



app = QApplication([])


v = viz.TsdFrameWidget(data)
v.plot.sort_by("channels")

v.show()

QTimer.singleShot(3000, app.quit)

if __name__ == "__main__":
    app.exit(app.exec())
