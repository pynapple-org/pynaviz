"""Stress test: load 125GB NWB file with almost 2H recording and > 1000 units"""

import time
import pynapple as nap
from PyQt6.QtWidgets import QApplication
from pynaviz import scope
import pynaviz as viz
import numpy as np
from pynaviz.video_handling import VideoHandler

left_body = "/home/wolf/Downloads/ONE/openalyx.internationalbrainlab.org/churchlandlab_ucla/Subjects/MFD_09/2023-10-19/001/raw_video_data/_iblrig_leftCamera.raw.mp4"
v = VideoHandler(left_body)
start_time = time.time()
print(v[0].shape)
print(f"Loading frame took {time.time() - start_time:.2f} seconds")

start_time = time.time()
print(v[1].shape)
print(f"Loading 1 took {time.time() - start_time:.2f} seconds")

start_time = time.time()
print(v[2].shape)
print(f"Loading 2 took {time.time() - start_time:.2f} seconds")

start_time = time.time()
print(v[3].shape)
print(f"Loading 3 took {time.time() - start_time:.2f} seconds")

start_time = time.time()
print(v[10].shape)
print(f"Loading 10 took {time.time() - start_time:.2f} seconds")

start_time = time.time()
print(v[50].shape)
print(f"Loading 50 took {time.time() - start_time:.2f} seconds")

start_time = time.time()
print(v[51].shape)
print(f"Loading 51 took {time.time() - start_time:.2f} seconds")

start_time = time.time()
print(v[52].shape)
print(f"Loading 52 took {time.time() - start_time:.2f} seconds")

start_time = time.time()
print(v[1000].shape)
print(f"Loading 1000 took {time.time() - start_time:.2f} seconds")

start_time = time.time()
print(v[1001].shape)
print(f"Loading 1001 took {time.time() - start_time:.2f} seconds")

start_time = time.time()
print(v[1002].shape)
print(f"Loading 1002 took {time.time() - start_time:.2f} seconds")
quit()


app = QApplication([])

video_body = "/home/wolf/Downloads/ONE/openalyx.internationalbrainlab.org/churchlandlab_ucla/Subjects/MFD_09/2023-10-19/001/raw_video_data/_iblrig_bodyCamera.raw.mp4"
v1 = viz.base_plot.PlotVideo(left_body)

# v2 = viz.base_plot.PlotVideo(left_body)
# print(v2.data.shape)
#
# right_body = "/home/wolf/Downloads/ONE/openalyx.internationalbrainlab.org/churchlandlab_ucla/Subjects/MFD_09/2023-10-19/001/raw_video_data/_iblrig_rightCamera.raw.mp4"
# v3 = viz.base_plot.PlotVideo(left_body)

v4 = viz.PlotTsd(nap.Tsd(t=np.arange(0, 7000, 0.01), d=np.arange(0, 7000, 0.01)))

group = viz.controller_group.ControllerGroup([v1, v4])
# group = viz.controller_group.ControllerGroup([v3, v4])

# v1.controller.set_frame(10.0)

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
