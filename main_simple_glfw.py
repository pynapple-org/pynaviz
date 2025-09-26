"""
Test script
"""

import numpy as np
import pynapple as nap

import pynaviz as viz

# from wgpu.gui.auto import run

tsd1 = nap.Tsd(t=np.arange(1000), d=np.sin(np.arange(1000) * 0.1))
tsg = nap.TsGroup({
    i: nap.Ts(
        t=np.sort(np.random.uniform(0, 1000, 100 * (i + 1)))
    ) for i in range(10)}, metadata={"label": np.random.randn(10)})
tsdframe = nap.TsdFrame(t=np.arange(10000), d=np.random.uniform(0, 10, size=(10000, 10)), metadata={"label": np.random.randn(10)})
tsdtensor = nap.TsdTensor(t=np.arange(1000), d=np.random.randn(1000, 10, 10))
ts = nap.Ts(t=np.arange(10))
iset = nap.IntervalSet(start=np.arange(0, 1000, 10), end=np.arange(5, 1005, 10))

v = viz.PlotTsdTensor(tsdtensor)
v.superpose_time_series(tsdframe[:,0:2])
v.show()

