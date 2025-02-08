"""
Test script
"""
import os, requests, tqdm, math
import pynapple as nap
import numpy as np
import pynapple as nap
import pynaviz as viz
from pynaviz import scope

path = "A2929-200711.nwb"
# path = "Achilles_10252013_EEG.nwb"
# if path not in os.listdir("."):
#     r = requests.get(f"https://osf.io/2dfvp/download", stream=True)
#     block_size = 1024 * 1024
#     with open(path, "wb") as f:
#         for data in tqdm.tqdm(
#             r.iter_content(block_size),
#             unit="MB",
#             unit_scale=True,
#             total=math.ceil(int(r.headers.get("content-length", 0)) // block_size),
#         ):
#             f.write(data)


data = nap.load_file(path)
spikes = data['units']
# eeg = data['eeg'][:,0]
position = data['ry']

#
# tsd1 = nap.Tsd(t=np.arange(1000), d=np.cos(np.arange(1000) * 0.1))
# tsd2 = nap.Tsd(t=np.arange(1000), d=np.cos(np.arange(1000) * 0.1))
# tsd3 = nap.Tsd(t=np.arange(1000), d=np.arange(1000))
#
# tsg = nap.TsGroup({
#     i:nap.Ts(
#         t=np.sort(np.random.uniform(0, 1000, 100*(i+1)))
#     ) for i in range(10)
# })
#
scope(globals())