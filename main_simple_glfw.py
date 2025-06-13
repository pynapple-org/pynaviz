"""
Test script
"""
import numpy as np
import os
import pynapple as nap
from PyQt6.QtWidgets import QApplication
from wgpu.gui.auto import run
import pynaviz as viz
from pynaviz.base_plot import _BasePlot


tsd1 = nap.Tsd(t=np.arange(1000), d=np.sin(np.arange(1000) * 0.1))
tsg = nap.TsGroup({
    i:nap.Ts(
        t=np.sort(np.random.uniform(0, 1000, 100*(i+1)))
    ) for i in range(10)}, metadata={"label":np.random.randn(10)})
tsdframe = nap.TsdFrame(t=np.arange(1000),d=np.random.randn(1000, 10), metadata={"label":np.random.randn(10)})
tsdtensor = nap.TsdTensor(t=np.arange(1000), d=np.random.randn(1000, 10, 10))


app = QApplication([])
v = viz.PlotTsd(tsd1)
# v = _BasePlot(tsd1)
# v.canvas.request_draw(v.animate())
# v = viz.TsdTensorWidget(tsdtensor)
# v.show()
# v = viz.TsGroupWidget(tsg)
# v.show()
v1= viz.TsdFrameWidget(tsdframe)
from pynaviz.controller_group import ControllerGroup
ControllerGroup([v,v1])
v.show()
app.exit(app.exec())
