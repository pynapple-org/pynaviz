"""
Test script
"""
import numpy as np
import pynapple as nap

from pynaviz import scope

tsd1 = nap.Tsd(t=np.arange(1000), d=np.cos(np.arange(1000) * 0.1))
tsd2 = nap.Tsd(t=np.arange(1000), d=np.cos(np.arange(1000) * 0.1))
tsd3 = nap.Tsd(t=np.arange(1000), d=np.arange(1000))

tsg = nap.TsGroup({
    i:nap.Ts(
        t=np.sort(np.random.uniform(0, 1000, 100*(  i+1)))
    ) for i in range(10)

})
tsdframe = nap.TsdFrame(
    t=np.arange(1000)/30,
    d=np.random.randn(1000, 10), metadata={"area": ["pfc"]*4 + ["ppc"]*6, "type": ["exc", "inh"]*5, "channel": np.arange(10)})
tsdtensor = nap.TsdTensor(t=np.arange(10000)/30, d=np.random.randn(10000, 10, 10))


iset = nap.IntervalSet(start=np.arange(0, 1000, 10), end=np.arange(5, 1005, 10))


scope(globals())
