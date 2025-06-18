"""Stress test: load 125GB NWB file with almost 2H recording and > 1000 units"""

from PyQt6.QtWidgets import QApplication

import pynaviz as viz

app = QApplication([])

# Load spikes
# one = ONE()
# eid = "ebce500b-c530-47de-8cb1-963c552703ea"
# ssl = SpikeSortingLoader(eid=eid, one=one)
# spikes, clusters, channels = ssl.load_spike_sorting()
# clusters = ssl.merge_clusters(spikes, clusters, channels)
#
# tsgroup = nap.TsGroup(
#    {
#        cluster_id: nap.Ts(spikes["times"][spikes["clusters"] == cluster_id])
#        for cluster_id in clusters.pop("cluster_id")[:20]
#    },
#    metadata={metadata_key: metadata_values for metadata_key, metadata_values in clusters.items()},
# )
# counts = tsgroup.count(0.02)
# v0 = viz.base_plot.PlotTsdFrame(counts)
#
## Load videos
# video_body = "/home/wolf/Downloads/ONE/openalyx.internationalbrainlab.org/churchlandlab_ucla/Subjects/MFD_09/2023-10-19/001/raw_video_data/_iblrig_bodyCamera.raw.mp4"
# v1 = viz.base_plot.PlotVideo(video_body)
#
left_body = "/home/wolf/Downloads/ONE/openalyx.internationalbrainlab.org/churchlandlab_ucla/Subjects/MFD_09/2023-10-19/001/raw_video_data/_iblrig_leftCamera.raw.mp4"
v2 = viz.base_plot.PlotVideo(left_body)
#
right_body = "/home/wolf/Downloads/ONE/openalyx.internationalbrainlab.org/churchlandlab_ucla/Subjects/MFD_09/2023-10-19/001/raw_video_data/_iblrig_rightCamera.raw.mp4"
v3 = viz.base_plot.PlotVideo(right_body)

# VISUALIZE
group = viz.controller_group.ControllerGroup([v2, v3])

if __name__ == "__main__":
    app.exit(app.exec())
