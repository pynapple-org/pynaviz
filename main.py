"""
Test script
"""
import numpy as np
import pandas as pd
import pynapple as nap

from pynaviz import scope
import pynaviz as viz

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
    d=np.random.rand(1000, 10)*500,
    columns=[f"neuron_{i}" for i in range(10)],
    metadata={"area":
                  ["pfc"]*4 + ["ppc"]*6,
              "type": ["exc", "inh"]*5,
              "channel": np.arange(10)}
)

pose = nap.TsdFrame(
    t=np.arange(10000)/30,
    d=np.random.randn(10000, 10),
    columns=["nose_x", "nose_y", "ear_r_x", "ear_r_y", "ear_l_x", "ear_l_y", "tailbase_x", "tailbase_y", "speed", "acceleration"]
)

tsdtensor = nap.TsdTensor(t=np.arange(10000)/30, d=np.random.randn(10000, 10, 10))


iset = nap.IntervalSet(start=np.arange(0, 1000, 10), end=np.arange(5, 1005, 10))


video_path = "docs/examples/m3v1mp4.mp4"
v = viz.VideoHandler(video_path)


df = pd.read_hdf("docs/examples/m3v1mp4DLC_Resnet50_openfieldOct30shuffle1_snapshot_best-70.h5")
df.columns = [f"{bodypart}_{coord}" for _, bodypart, coord in df.columns]
df = df[[c for c in df.columns if c.endswith(("_x", "_y"))]]
y_col = [c for c in df.columns if c.endswith("_y")]
df[y_col] = df[y_col]*-1 + 480 # Flipping y axis
skeleton = nap.TsdFrame(t=df.index.values/30, d=df.values, columns=df.columns)

scope(globals())
