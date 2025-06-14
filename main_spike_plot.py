"""
Test script
"""

import numpy as np
import os
import pynapple as nap
from PyQt6.QtWidgets import QApplication
import pynaviz as viz


tsgroup = nap.load_file("/home/wolf/sub-25_day-25_ses-OF1_srt-kilosort4_clusters.npz")

app = QApplication([])

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

if __name__ == "__main__":
    app.exit(app.exec())
