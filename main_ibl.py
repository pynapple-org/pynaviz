import os
import sys
from pathlib import Path

import numpy as np
import pynapple as nap

from one.api import ONE

# from brainbox.io.one import SpikeSortingLoader
from PyQt6.QtWidgets import QApplication, QHBoxLayout, QVBoxLayout, QWidget
from tqdm import tqdm

import pynaviz as viz

# app = QApplication([])

# ------------------------------------------------------------------------------------
# Load IBL session
one = ONE()
eid = "ebce500b-c530-47de-8cb1-963c552703ea"

# Videos
ibl_path = Path(os.path.expanduser("~/Downloads/ONE/"))
if not ibl_path.exists():
    print("Please set the path to your IBL data directory in the variable `ibl_path`.")
    quit()
videos = {}
for label in ["left", "body", "right"]:
    video_path = (
        ibl_path
        / f"openalyx.internationalbrainlab.org/churchlandlab_ucla/Subjects/MFD_09/2023-10-19/001/raw_video_data/_iblrig_{label}Camera.raw.mp4"
    )
    if not video_path.exists():
        one.load_dataset(eid, f"*{label}Camera.raw*", collection="raw_video_data")
    times = one.load_object(eid, f"{label}Camera", collection="alf", attribute=["times*"])["times"]
    licks = one.load_object(eid, f"{label}Camera", collection="alf", attribute=["times*"])["times"]
    videos[label] = (times, video_path)




# Events and intervals
timings = one.load_object(eid, "trials", collection="alf")
licks = nap.Ts(one.load_object(eid, "licks", collection="alf")["times"])
reactions = nap.Ts(timings["firstMovement_times"])
trials = nap.IntervalSet(timings["intervals"])

# # Spikes
# ssl = c(eid=eid, one=one)
# spikes, clusters, channels = ssl.load_spike_sorting()
# clusters = ssl.merge_clusters(spikes, clusters, channels)
# clusters = nap.TsGroup(
#     {
#         cluster_id: nap.Ts(spikes["times"][spikes["clusters"] == cluster_id])
#         for cluster_id in tqdm(
#             clusters.pop("cluster_id")[:200], unit="cluster", desc="Loading spikes"
#         )
#     },
#     metadata={metadata_key: metadata_values for metadata_key, metadata_values in clusters.items()},
# )
# subset = nap.IntervalSet(start=0, end=100)
# counts = clusters.count(0.001, ep=subset)

# Modulation
# for label, tref, minmax in [
#    ("licks", licks, (-0.1, 0.1)),
#    ("reactions", licks, (-0.2, 0.4)),
#    ("trials", nap.Ts(trials.start), (-0.2, 0.6)),
# ]:
#    psths = nap.compute_perievent_continuous(
#        timeseries=counts,
#        tref=tref,
#        minmax=minmax,
#    ).mean(axis=1)
#    before = (psths.index >= minmax[0]) & (psths.index < 0.0)
#    after = (psths.index >= 0.0) & (psths.index <= minmax[1])
#    clusters[f"{label}_modulation"] = (
#        psths.values[after].mean(axis=0) - psths.values[before].mean(axis=0)
#    ) / (psths.values[after].mean(axis=0) + psths.values[before].mean(axis=0))
#    clusters[f"{label}_peak"] = np.argmax(
#        psths.values[(psths.index >= minmax[0]) & (psths.index < minmax[1])], axis=0
#    )
# clusters = clusters[
#    ~np.isnan(clusters["licks_modulation"])
#    & ~np.isnan(clusters["trials_modulation"])
#    & ~np.isnan(clusters["reactions_modulation"])
#    & (np.abs(clusters["reactions_modulation"]) > 0.1)
# ]


# ------------------------------------------------------------------------------------
# Visualize

# spikes
# raster = viz.TsGroupWidget(clusters)
# raster.show()
# # intervals
# raster.plot.add_interval_sets(trials, colors="white", alpha=0.2)
# videos

videos = [
    viz.VideoWidget(video_path=video_path, t=times)
    for times, video_path in videos.values()
]
vars = {name:var for name, var in zip(["left_video", "body_video", "right_video"], videos)}
vars["trials"] = trials
vars["licks"] = licks



from pynaviz import scope


# videos['licks'] = licks
# videos['trials'] = trials

scope(vars)


# # link
# viz.controller_group.ControllerGroup([video.plot for video in videos])# + [raster.plot])
#
# # Main window
# window = QWidget()
#
# # Top row: horizontal layout for videos
# top_layout = QHBoxLayout()
# for video in videos:
#     top_layout.addWidget(video)
#
# # Overall layout: vertical, stacking videos on top of raster
# main_layout = QVBoxLayout()
# main_layout.addLayout(top_layout)  # Add videos at the top
# # main_layout.addWidget(raster)  # Raster goes at the bottom
#
# window.setLayout(main_layout)
# window.show()
#
# if __name__ == "__main__":
#     app.exit(app.exec())

