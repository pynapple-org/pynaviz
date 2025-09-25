import pynaviz as viz
import pytest
import pynapple as nap
import numpy as np
import sys
from PyQt6.QtWidgets import QApplication, QDockWidget


@pytest.fixture
def setup_main_gui():
    tsdframe = nap.TsdFrame(
        t=np.arange(1000) / 30,
        d=np.random.randn(1000, 10),
        metadata={"area": ["pfc"] * 4 + ["ppc"] * 6, "type": ["exc", "inh"] * 5, "channel": np.arange(10)})
    tsdtensor = nap.TsdTensor(t=np.arange(10000) / 30, d=np.random.randn(10000, 10, 10))
    app = QApplication(sys.argv)
    gui = viz.qt.mainwindow.MainWindow()

def test_save_load_layout():
    pass