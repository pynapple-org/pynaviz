"""Stress test: load 125GB NWB file with almost 2H recording and > 1000 units"""

try:
    import nemos as nmo
except ImportError:
    raise ImportError("Install nemos to run this.")
import pynapple as nap
from pynaviz import scope
import pynaviz as viz
from PyQt6.QtWidgets import QApplication

io = nmo.fetch.download_dandi_data(
    "000409",
    "sub-CSH-ZAD-022/sub-CSH-ZAD-022_ses-41431f53-69fd-4e3b-80ce-ea62e03bf9c7_behavior+ecephys+image.nwb",
)
data = nap.NWBFile(io.read(), lazy_loading=True)
print("loaded data")
#units = data["units"]
print("extract unit")

#scope(units)


app = QApplication([])
# app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt6())

# viz.TsdWidget(tsd1).show()
v = viz.TsdFrameWidget(data["ElectricalSeriesLf00"].get(0, 100))
# v = viz.TsGroupWidget(units)
v.plot.controller.show_interval(0, 20)
# v.plot.color_by("label", 'jet')
# v.plot.sort_by("rate")
v.show()
# v = viz.TsdFrameWidget(tsdframe)
# v.show()


app.exit(app.exec())
