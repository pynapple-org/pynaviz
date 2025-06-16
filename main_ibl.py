"""Stress test: load 125GB NWB file with almost 2H recording and > 1000 units"""

import pynapple as nap
from PyQt6.QtWidgets import QApplication
from pynaviz import scope
import pynaviz as viz
import numpy as np

app = QApplication([])

video_body = "/Users/gviejo/Downloads/ONE/openalyx.internationalbrainlab.org/churchlandlab_ucla/Subjects/MFD_09/2023-10-19/001/raw_video_data/_iblrig_bodyCamera.raw.mp4"
v1 = viz.base_plot.PlotVideo(video_body)

# left_body = "/Users/gviejo/Downloads/ONE/openalyx.internationalbrainlab.org/churchlandlab_ucla/Subjects/MFD_09/2023-10-19/001/raw_video_data/_iblrig_leftCamera.raw.mp4"
# v2 = viz.base_plot.PlotVideo(left_body)
#
# right_body = "/Users/gviejo/Downloads/ONE/openalyx.internationalbrainlab.org/churchlandlab_ucla/Subjects/MFD_09/2023-10-19/001/raw_video_data/_iblrig_rightCamera.raw.mp4"
# v3 = viz.base_plot.PlotVideo(right_body)



v4 = viz.PlotTsd(nap.Tsd(t=np.arange(0, 7000, 0.01), d=np.arange(0, 7000, 0.01)))

# group = viz.controller_group.ControllerGroup([v1, v2, v3, v4])
group = viz.controller_group.ControllerGroup([v1, v4])

v1.controller.set_frame(10.0)

# v1.show()
# v2.show()
# v3.show()
# v4.show()
#
# io = nmo.fetch.download_dandi_data(
#     "000409",
#     "sub-CSH-ZAD-022/sub-CSH-ZAD-022_ses-41431f53-69fd-4e3b-80ce-ea62e03bf9c7_behavior+ecephys+image.nwb",
# )
# data = nap.NWBFile(io.read(), lazy_loading=True)
# print("loaded data")
# #units = data["units"]
# print("extract unit")
#
#
#
# app = QApplication([])
# # app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt6())
#
# # viz.TsdWidget(tsd1).show()
# v = viz.TsdFrameWidget(data["ElectricalSeriesLf00"].get(0, 100))
# # v = viz.TsGroupWidget(units)
# v.plot.controller.show_interval(0, 20)
# # v.plot.color_by("label", 'jet')
# # v.plot.sort_by("rate")
# v.show()
# # v = viz.TsdFrameWidget(tsdframe)
# # v.show()
#
if __name__ == "__main__":
    app.exit(app.exec())