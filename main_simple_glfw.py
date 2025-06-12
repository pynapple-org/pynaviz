"""
Test script
"""
import numpy as np
import os, sys
import pynapple as nap
# from PyQt6.QtWidgets import QApplication
from wgpu.gui.auto import run
import pynaviz as viz
from pynaviz.base_plot import _BasePlot

import os

from pynaviz.controller_group import ControllerGroup

tsd1 = nap.Tsd(t=np.arange(1000), d=np.sin(np.arange(1000) * 0.1))
tsg = nap.TsGroup({
    i:nap.Ts(
        t=np.sort(np.random.uniform(0, 1000, 100*(i+1)))
    ) for i in range(10)}, metadata={"label":np.random.randn(10)})
tsdframe = nap.TsdFrame(t=np.arange(1000),d=np.random.randn(1000, 10), metadata={"label":np.random.randn(10)})
tsdtensor = nap.TsdTensor(t=np.arange(1000), d=np.random.randn(1000, 10, 10))

# video_path = "tests/test_video/numbered_video.mp4"
# video_path = "/Users/ebalzani/Downloads/20250404_171446-compressed.mp4"
video_path = "/Users/ebalzani/Downloads/WhatsApp Video 2025-06-11 at 11.08.16 PM.mp4"
# video_path = "/Users/ebalzani/Downloads/Take 2022-12-13 11.33.25 AM-Camera 5 (#378042) Sofia.avi"
# app = QApplication([])

# v = viz.PlotTsd(tsd1)
# v = _BasePlot(tsd1)
# v.canvas.request_draw(v.animate())
# v = viz.TsdTensorWidget(tsdtensor)
v = viz.base_plot.PlotVideo(video_path)
print(v.data.shape)

# v.show()
# v = viz.TsGroupWidget(tsg)
# v.show()
v2 = viz.PlotTsdFrame(tsdframe)
ControllerGroup([v, v2])
v.show()
v2.show()

# v.close()  # Ensure clean shutdown
