"""
Test script
"""

import pynapple as nap
from PyQt6.QtWidgets import QApplication
import pynaviz as viz
from tqdm import tqdm

from one.api import ONE
import ibllib.io.video as vidio
from brainbox.io.one import SpikeSortingLoader

app = QApplication([])

tsgroup = nap.load_file(
    "/home/wolf/Documents/spatial-manifolds/data/sessions/M25/D25/VR/sub-25_day-25_ses-VR_srt-kilosort4_clusters.npz"
)
v = viz.TsGroupWidget(tsgroup)
# v.plot.group_by("group")
# v.plot.sort_by("channel")#, order="descending")

# v.plot.sort_by("channel")
# v.plot.sort_by("channel")
# for i in range(4):
#     print(i)
#     print(v.plot._manager.data['offset'])
#     v.plot.sort_by("channel")
#     v.plot.group_by("group")

v.show()

one = ONE()
eid = "ebce500b-c530-47de-8cb1-963c552703ea"
labels = ["left", "right", "body"]

# Loading the videos
# for label in labels:
#    video_body = one.load_dataset(
#        eid, f"*{label}Camera.raw*", collection="raw_video_data"
#    )
#    print(video_body)

# Loading the spikesssl
# ssl = SpikeSortingLoader(eid=eid, one=one)
# spikes, clusters, channels = ssl.load_spike_sorting()
# clusters = ssl.merge_clusters(spikes, clusters, channels)
# waveforms = ssl.load_spike_sorting_object("waveforms")  # loads in the template waveforms
#
# tsgroup = nap.TsGroup(
#    {
#        cluster_id: nap.Ts(spikes["times"][spikes["clusters"] == cluster_id])
#        for cluster_id in tqdm(clusters.pop("cluster_id"))
#    },
#    metadata={metadata_key: metadata_values for metadata_key, metadata_values in clusters.items()},
# )
# print(tsgroup)

if __name__ == "__main__":
    app.exit(app.exec())
