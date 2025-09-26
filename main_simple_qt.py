"""
Test script
"""

import numpy as np
import pynapple as nap

import pynaviz as viz

tsd1 = nap.Tsd(t=np.arange(1000), d=np.sin(np.arange(1000) * 0.1))
tsg = nap.TsGroup({
    i:nap.Ts(
        t=np.sort(np.random.uniform(0, 1000, 100*(i+1)))
    ) for i in range(10)}, metadata={"label":np.random.randn(10)})
tsdtensor = nap.TsdTensor(t=np.arange(1000), d=np.random.randn(1000, 10, 10))
tsdframe = nap.TsdFrame(
    t=np.arange(1000),
    d=np.stack((np.cos(2*np.pi*np.arange(0, 100, 0.1)), np.sin(2*np.pi*np.arange(0, 100, 0.1)))).T,
    metadata={"label":np.random.randn(2)})

tsdframe = nap.TsdFrame(
    t=np.arange(1000),
    d=np.stack((np.cos(2*np.pi*np.arange(0, 100, 0.1)), np.sin(2*np.pi*np.arange(0, 100, 0.1)))).T
    )

ts = nap.Ts(t=np.arange(10))


# v = viz.TsdFrameWidget(tsdframe)
# # v = viz.TsdWidget(tsd1)
# v = viz.TsWidget(ts)

v = viz.TsdTensorWidget(tsdtensor, tsdframe=tsdframe)
v.show()

