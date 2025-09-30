"""
Test script
"""
import numpy as np
import pynapple as nap
import pynaviz as viz
from PyQt6.QtWidgets import QApplication, QHBoxLayout, QWidget

from pynaviz import scope
from pynaviz.controller_group import ControllerGroup

tsdframe = nap.TsdFrame(
    t=np.arange(1000),
    d=np.stack((np.arange(1000)%10,
                np.arange(1000)%10)
                ).T,
    columns = ["x", "y"]
)

tsdtensor = nap.TsdTensor(t=np.arange(1000), d=np.random.randn(1000, 10, 10))


scope(globals())

# app = QApplication([])
# window = QWidget()
# window.setMinimumSize(1500, 800)
# layout = QHBoxLayout()
#
# viz1 = viz.TsdFrameWidget(tsdframe)
# viz1.plot.plot_x_vs_y(0, 1)
# viz2 = viz.TsdTensorWidget(tsdtensor)
# viz2.plot.superpose_time_series(tsdframe)
#
# ctrl_group = ControllerGroup()
# ctrl_group.add(viz1, 0)
# ctrl_group.add(viz2, 1)
#
# layout.addWidget(viz1)
# layout.addWidget(viz2)
# window.setLayout(layout)
# window.show()
#
# if __name__ == "__main__":
#     app.exit(app.exec())
