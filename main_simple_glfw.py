"""
Test script
"""

import numpy as np
import pynapple as nap
import pandas as pd
import pynaviz as viz

# from wgpu.gui.auto import run

tsd1 = nap.Tsd(t=np.arange(1000), d=np.sin(np.arange(1000) * 0.1))
tsg = nap.TsGroup({
    i: nap.Ts(
        t=np.sort(np.random.uniform(0, 1000, 100 * (i + 1)))
    ) for i in range(10)}, metadata={"label": np.random.randn(10)})
tsdframe = nap.TsdFrame(t=np.arange(10000), d=np.random.uniform(0, 100, size=(10000, 10)), metadata={"label": np.random.randn(10)})
tsdtensor = nap.TsdTensor(t=np.arange(1000), d=np.random.randn(1000, 10, 10))
ts = nap.Ts(t=np.arange(10))
iset = nap.IntervalSet(start=np.arange(0, 1000, 10), end=np.arange(5, 1005, 10))

skeleton = nap.TsdFrame(
    t=np.arange(10000)/30,
    d=np.random.randn(10000, 8)*480,
    columns=["nose_x", "nose_y", "ear_r_x", "ear_r_y", "ear_l_x", "ear_l_y", "tailbase_x", "tailbase_y"]
)

df = pd.read_hdf("/mnt/home/gviejo/pynaviz/docs/examples/m3v1mp4DLC_Resnet50_openfieldOct30shuffle1_snapshot_best-70.h5")
df.columns = [f"{bodypart}_{coord}" for _, bodypart, coord in df.columns]
df = df[[c for c in df.columns if c.endswith(("_x", "_y"))]]
y_col = [c for c in df.columns if c.endswith("_y")]
df[y_col] = df[y_col]*-1 + 480 # Flipping y axis


skeleton = nap.TsdFrame(t=df.index.values/30, d=df.values, columns=df.columns)

# v = viz.PlotVideo("/mnt/home/gviejo/pynaviz/docs/examples/m3v1mp4.mp4")
v = viz.VideoWidget("/mnt/home/gviejo/pynaviz/docs/examples/m3v1mp4.mp4")

v.plot.superpose_points(skeleton)
v.show()

