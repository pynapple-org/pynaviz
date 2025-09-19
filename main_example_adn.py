import numpy as np
import os
import pynapple as nap
from PyQt6.QtWidgets import QApplication, QWidget, QHBoxLayout, QVBoxLayout
from sklearn.manifold import Isomap

import pynaviz as viz
import sys
from scipy.ndimage import gaussian_filter1d
from sklearn.decomposition import KernelPCA, PCA
from matplotlib.pyplot import *

from pynaviz import scope

data = nap.load_file(os.path.expanduser("~/Dropbox/A5044-240404A/A5044-240404A.nwb"))

units = data["units"]
units = units[(units.location == "adn") | (units.location == "lmn")]
# units = units[units.location == "adn"]
units = units[units.rate > 1]
ry = data['ry']
tc = nap.compute_1d_tuning_curves(units, ry, 120, ry.time_support)
tc = tc.apply(gaussian_filter1d, axis=0, sigma=1)
units.peak = tc.idxmax()
SI = nap.compute_1d_mutual_info(tc, ry, ry.time_support.loc[[0]], minmax=(0, 2 * np.pi))
units.set_info(SI)
units = units[units.SI > 0.3]
units.order = np.argsort(units.peak.sort_values().index.values)

count = units.count(0.1, ry.time_support).smooth(0.2)
# imap = KernelPCA(n_components=2, kernel='cosine').fit_transform(count)
# imap = PCA(n_components=2).fit_transform(count)
imap = Isomap(n_components=2).fit_transform(count)
imap = nap.TsdFrame(t=count.t, d=imap)
# units.order = np.sort(units.peak)

# Resetting everything to t = 0
ep = ry.time_support

units2 = {}
for n in units.keys():
    t = units[n].t
    new_t = t - ep.start[0]
    units2[n] = nap.Ts(t=new_t[new_t>0])
units2 = nap.TsGroup(units2, metadata=units.metadata.drop(['rate'], axis=1))

imap2 = nap.TsdFrame(t=imap.t - ep.start[0], d=imap.d, metadata={"imap":[0, 1]})

###############
# VISUALIZATION
###############

# app = QApplication([])
#
# window = QWidget()
# window.setMinimumSize(1500, 800)
#
# layout = QVBoxLayout()
#
# hlayout = QHBoxLayout()
#
# # Raster
# viz1 = viz.TsGroupWidget(units2, set_parent=True)
# viz1.plot.sort_by("order")
# viz1.plot.color_by("order", cmap_name="hsv")
#
# # Manifold
# viz2 = viz.TsdFrameWidget(imap2, set_parent=True)
# viz2.plot.plot_x_vs_y(0, 1, markersize=20.0)
#
# # Video
viz3 = viz.VideoWidget(os.path.expanduser("~/Dropbox/A5044-240404A/Video4th.avi"))
#
# # Controller
# ctrl_group = viz.controller_group.ControllerGroup([viz1.plot, viz2.plot, viz3.plot])
#
# # Layout
# hlayout.addWidget(viz2)
# hlayout.addWidget(viz3)
# layout.addLayout(hlayout)
# layout.addWidget(viz1)
#
# window.setLayout(layout)
# window.show()

if __name__ == "__main__":
    scope({"units": units2, "manifold": imap2, "video": viz3},
          layout_path = "layout_2025-09-15_16-41.json")

    # app.exit(app.exec())