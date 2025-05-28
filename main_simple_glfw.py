"""
Test script
"""
import numpy as np
import os
import pynapple as nap
from wgpu.gui.auto import run
import pynaviz as viz

tsd1 = nap.Tsd(t=np.arange(1000), d=np.sin(np.arange(1000) * 0.1))
tsg = nap.TsGroup({
    i:nap.Ts(
        t=np.sort(np.random.uniform(0, 1000, 100*(i+1)))
    ) for i in range(10)}, metadata={"label":np.random.randn(10)})
tsdframe = nap.TsdFrame(t=np.arange(1000),d=np.random.randn(1000, 10), metadata={"label":np.random.randn(10)})
tsdtensor = nap.TsdTensor(t=np.arange(1000), d=np.random.randn(1000, 10, 10))



v = viz.PlotTsd(tsd1)
# v = viz.TsdTensorWidget(tsdtensor)
# v.show()
# v = viz.TsGroupWidget(tsg)
# v.show()
# v = viz.TsdFrameWidget(tsdframe)
# v.show()

# if __name__ == "__main__":
#     run()