"""Stress test: load 125GB NWB file with almost 2H recording and > 1000 units"""

from pathlib import Path
import os

from one.api import ONE
from PyQt6.QtWidgets import QApplication

import uviz as viz

app = QApplication([])

# ------------------------------------------------------------------------------------
# Load IBL session
one = ONE()
eid = "ebce500b-c530-47de-8cb1-963c552703ea"

# Videos
ibl_path = Path(os.path.expanduser("~/Downloads"))
videos = {}
for label in ["left", "body", "right"]:
    video_path = (
        ibl_path
        / f"ONE/openalyx.internationalbrainlab.org/churchlandlab_ucla/Subjects/MFD_09/2023-10-19/001/raw_video_data/_iblrig_{label}Camera.raw.mp4"
    )
    if not video_path.exists():
        one.load_dataset(eid, f"*{label}Camera.raw*", collection="raw_video_data")
    times = one.load_object(eid, f"{label}Camera", collection="alf", attribute=["times*"])["times"]
    videos[label] = (times, video_path)

# Spikes
# ssl = SpikeSortingLoader(eid=eid, one=one)
# spikes, clusters, channels = ssl.load_spike_sorting()
# clusters = ssl.merge_clusters(spikes, clusters, channels)
# tsgroup = nap.TsGroup(
#    {
#        cluster_id: nap.Ts(spikes["times"][spikes["clusters"] == cluster_id])
#        for cluster_id in clusters.pop("cluster_id")[:20]
#    },
#    metadata={metadata_key: metadata_values for metadata_key, metadata_values in clusters.items()},
# )
# counts = tsgroup.count(0.02)

# ------------------------------------------------------------------------------------
# Visualize
videos = [
    viz.VideoWidget(video_path=video_path, t=times)
    for times, video_path in videos.values()
]
group = viz.controller_group.ControllerGroup([video.plot for video in videos])

for v in videos:
    v.show()

if __name__ == "__main__":
    app.exit(app.exec())
