"""
This script is for generating screenshots than then will be compared to during the test
Naming of the file should equal the name of the test it corresponds to.

"""
import os

import numpy as np
import pynapple as nap
from PIL import Image

import pynaviz as viz

tsd1 = nap.Tsd(t=np.arange(1000), d=np.sin(np.arange(1000) * 0.1))
# tsg = nap.TsGroup({
#     i:nap.Ts(
#         t=np.sort(np.random.uniform(0, 1000, 100*(i+1)))
#     ) for i in range(10)}, metadata={"label":np.random.randn(10)})
# tsdframe = nap.TsdFrame(t=np.arange(1000),d=np.random.randn(1000, 10), metadata={"label":np.random.randn(10)})
# tsdtensor = nap.TsdTensor(t=np.arange(1000), d=np.random.randn(1000, 10, 10))



v = viz.PlotTsd(tsd1)
v.animate()
image_data = v.renderer.snapshot()
image = Image.fromarray(image_data, mode="RGBA")
image.save(
    os.path.expanduser("~/pynaviz/tests/screenshots/test_plot_tsd.png")
)
