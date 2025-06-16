"""
Test script
"""
import numpy as np
import os
import pynapple as nap
from PyQt6.QtWidgets import QApplication
import pynaviz as viz
from pynaviz.controller_group import ControllerGroup


def main():
    tsd1 = nap.Tsd(t=np.arange(1000), d=np.sin(np.arange(1000) * 0.1))
    tsg = nap.TsGroup({
        i: nap.Ts(
            t=np.sort(np.random.uniform(0, 1000, 100 * (i + 1)))
        ) for i in range(10)}, metadata={"label": np.random.randn(10)})
    tsdframe = nap.TsdFrame(t=np.arange(1000), d=np.random.randn(1000, 10), metadata={"label": np.random.randn(10)})
    tsdtensor = nap.TsdTensor(t=np.arange(1000), d=np.random.randn(1000, 10, 10))

    video_path = "/Users/ebalzani/Downloads/WhatsApp Video 2025-06-11 at 11.08.16 PM.mp4"

    # Your app setup
    v = viz.base_plot.PlotVideo(video_path)
    v2 = viz.PlotTsdFrame(tsdframe)
    ControllerGroup([v, v2])
    app.exec()  # Start the Qt event loop


if __name__ == "__main__":
    import multiprocessing as mp
    mp.freeze_support()  # Needed on Windows, harmless on macOS
    app = QApplication([])
    main()
