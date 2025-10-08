"""
Test script
"""
import numpy as np
import pynapple as nap
import pynaviz as viz

from pynaviz import IntervalSetWidget

tsd1 = nap.Tsd(t=np.arange(1000), d=np.cos(np.arange(1000) * 0.1))
tsg = nap.TsGroup({
    i:nap.Ts(
        t=np.linspace(0, 100, 100*(10-i))
    ) for i in range(10)},
    metadata={
        "label":np.arange(10),
        "colors":["hotpink","lightpink","cyan","orange","lightcoral","lightsteelblue","lime","lightgreen","magenta","pink"],
        "test": ["hello", "word"]*5,
        }
    )
tsdframe = nap.TsdFrame(t=np.arange(1000),d=np.random.randn(1000, 10), metadata={"label":np.random.randn(10)})
tsdtensor = nap.TsdTensor(t=np.arange(1000), d=np.random.randn(1000, 10, 10))

iset = nap.IntervalSet(start=0, end=100, metadata={"label":["first"]})

data = nap.load_file("../pynapple/doc/examples/Mouse32-140822.nwb")

iset = data['position_time_support']

# scope({"iset":iset})
# scope("../pynapple/doc/examples/Mouse32-140822.nwb")
v = IntervalSetWidget(iset)
v.plot.color_by("tags", cmap_name="jet")
v.show()