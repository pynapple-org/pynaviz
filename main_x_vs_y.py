"""
Test script
"""
import numpy as np
import pynapple as nap
import pynaviz as viz
from PyQt6.QtWidgets import QApplication
import qdarkstyle


tsdframe = nap.TsdFrame(
    t=np.arange(1000),
    d=np.stack((np.cos(np.arange(1000)*0.1),
                np.sin(np.arange(1000)*0.1))).T
                )




app = QApplication([])
# app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt6())

# viz.TsdWidget(tsd1).show()
# viz.TsdTensorWidget(tsdtensor).show()
# v = viz.TsGroupWidget(tsg)
# v.plot.controller.show_interval(0, 20)
# v.plot.color_by("label", 'jet')
# v.plot.sort_by("rate")
# v.show()
v = viz.TsdFrameWidget(tsdframe)
# v.plot.plot_x_vs_y(0, 1)
# v.plot.plot_time_series()
v.show()

if __name__ == "__main__":
    app.exit(app.exec())
