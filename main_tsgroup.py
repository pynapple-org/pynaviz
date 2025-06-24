"""
Test script
"""

import pynapple as nap
from PyQt6.QtWidgets import QApplication
import pynaviz as viz
from tqdm import tqdm

from one.api import ONE
from brainbox.io.one import SpikeSortingLoader

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
        for cluster_id in tqdm(clusters.pop("cluster_id"))
    },
    metadata={metadata_key: metadata_values for metadata_key, metadata_values in clusters.items()},
)

viz.base_plot.PlotTsGroup(tsgroup).show()

if __name__ == "__main__":
    app.exit(app.exec())
