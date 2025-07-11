"""
Test script
"""

import pynapple as nap
import pynaviz as viz
from brainbox.io.one import SpikeSortingLoader
from one.api import ONE
from PyQt6.QtWidgets import QApplication
from tqdm import tqdm

app = QApplication([])

one = ONE()
eid = "ebce500b-c530-47de-8cb1-963c552703ea"
labels = ["left", "right", "body"]

# Loading the spikes
ssl = SpikeSortingLoader(eid=eid, one=one)
spikes, clusters, channels = ssl.load_spike_sorting()
clusters = ssl.merge_clusters(spikes, clusters, channels)

tsgroup = nap.TsGroup(
    {
        cluster_id: nap.Ts(spikes["times"][spikes["clusters"] == cluster_id])
        for cluster_id in tqdm(clusters.pop("cluster_id")[:50])
    },
    metadata={metadata_key: metadata_values for metadata_key, metadata_values in clusters.items()},
)

v1 = viz.TsGroupWidget(tsgroup)
v1.show()

if __name__ == "__main__":
    app.exit(app.exec())
