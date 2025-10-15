"""
Test script
"""
import sys

import imageio
import numpy as np
import pynapple as nap
import requests, os
from PyQt6.QtGui import QAction

import pynaviz as viz
from PyQt6.QtCore import QTimer, Qt, QPoint
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication, QDockWidget, QMenu, QDialog
from pynaviz.qt.mainwindow import MainWindow
from matplotlib.pyplot import *
from PIL import ImageGrab
from utils import grab_window, click_on_item, add_dock_widget, move_and_resize_dock, save_gif



def loadXML(path):
    listdir = os.listdir(path)
    xmlfiles = [f for f in listdir if f.endswith('.xml')]
    new_path = os.path.join(path, xmlfiles[0])

    from xml.dom import minidom
    xmldoc = minidom.parse(new_path)
    nChannels = xmldoc.getElementsByTagName('acquisitionSystem')[0].getElementsByTagName('nChannels')[0].firstChild.data
    fs_dat = xmldoc.getElementsByTagName('acquisitionSystem')[0].getElementsByTagName('samplingRate')[0].firstChild.data
    fs_eeg = xmldoc.getElementsByTagName('fieldPotentials')[0].getElementsByTagName('lfpSamplingRate')[
        0].firstChild.data
    if os.path.splitext(xmlfiles[0])[0] + '.dat' in listdir:
        fs = fs_dat
    elif os.path.splitext(xmlfiles[0])[0] + '.eeg' in listdir:
        fs = fs_eeg
    else:
        fs = fs_eeg
    shank_to_channel = {}
    shank_to_keep = {}
    groups = xmldoc.getElementsByTagName('anatomicalDescription')[0].getElementsByTagName('channelGroups')[
        0].getElementsByTagName('group')
    for i in range(len(groups)):
        shank_to_channel[i] = []
        shank_to_keep[i] = []
        for child in groups[i].getElementsByTagName('channel'):
            shank_to_channel[i].append(int(child.firstChild.data))
            tmp = child.toprettyxml()
            shank_to_keep[i].append(int(tmp[15]))

        # shank_to_channel[i] = np.array([int(child.firstChild.data) for child in groups[i].getElementsByTagName('channel')])
        shank_to_channel[i] = np.array(shank_to_channel[i])
        shank_to_keep[i] = np.array(shank_to_keep[i])
        shank_to_keep[i] = shank_to_keep[i] == 0  # ugly
    return int(nChannels), int(fs), shank_to_channel, shank_to_keep


def main():
    # Load data
    path = "A2929-200711.dat"
    num_channels, fs, shank_to_channel, shank_to_keep = loadXML(".")
    tsdframe = nap.misc.load_eeg(
        path,
        n_channels=num_channels,
        frequency=fs
    )
    ch = np.hstack([shank_to_channel[i] for i in shank_to_channel])
    tsdframe.channel = np.argsort(ch)
    gr = np.zeros(num_channels)
    for i in shank_to_channel:
        gr[shank_to_channel[i]] = i
    tsdframe.group = gr

    # Initialize the application and main window
    app = QApplication.instance() or QApplication(sys.argv)
    win = MainWindow({"lfp": tsdframe})
    win.show()

    # Make sure the window is shown and painted
    app.processEvents()
    QTest.qWaitForWindowExposed(win)    # waits until the window is mapped
    app.processEvents()
    QTest.qWait(1000)

    frames = []
    durations = []

    # --- Action 1: initial view ---
    frames.append(grab_window(win))
    durations.append(800)

    tree_widget = win.variable_dock.treeWidget

    # --- Add docks ---
    add_dock_widget(tree_widget, win, app, frames, durations, item_number=0) # eeg

    # --- Toggle tree widget visibility ---
    QTest.mouseClick(win.variable_dock.handle, Qt.MouseButton.LeftButton)
    app.processEvents()
    frames.append(grab_window(win))  # grab frame
    durations.append(800)

    # --- Apply metadata actions ---
    dock_widgets = win.findChildren(QDockWidget)
    plots = {i:dock.widget().plot for i, dock in enumerate(dock_widgets[1:])}

    # Apply Sort by action
    # button = dock_widgets[0].button_container.action_button
    # QTimer.singleShot(1000, lambda: click_on_action("Sort by", win, frames, durations))
    # QTest.mouseClick(button, Qt.MouseButton.LeftButton)

    plots[0].sort_by("channel")
    QTest.qWait(1000)
    frames.append(grab_window(win))
    durations.append(800)
    app.processEvents()

    plots[0].group_by("group")
    QTest.qWait(1000)
    frames.append(grab_window(win))
    durations.append(800)
    app.processEvents()

    plots[0].color_by("group", "RdYlGn")
    QTest.qWait(1000)
    frames.append(grab_window(win))
    durations.append(800)
    app.processEvents()

    QTest.keyClick(dock_widgets[1].widget(), Qt.Key.Key_I)
    app.processEvents()
    QTest.qWait(1000)
    frames.append(grab_window(win))
    durations.append(800)

    QTest.keyClick(dock_widgets[1].widget(), Qt.Key.Key_I)
    app.processEvents()
    QTest.qWait(1000)
    frames.append(grab_window(win))
    durations.append(800)

    # # --- Action 4: play the animation ---
    duration_ms = 3000  # total recording time
    interval_ms = 50  # grab frame every 200 ms
    num_frames = duration_ms // interval_ms

    QTest.mouseClick(win.playPauseBtn, Qt.MouseButton.LeftButton)
    app.processEvents()

    running_frames = []
    running_durations = []

    for _ in range(num_frames):
        QTest.qWait(interval_ms)  # wait 25 ms
        frames.append(grab_window(win))  # grab frame
        durations.append(interval_ms)
        app.processEvents()
        running_frames.append(frames[-1])
        running_durations.append(interval_ms)

    # --- Pause the animation ---
    QTest.mouseClick(win.playPauseBtn, Qt.MouseButton.LeftButton)
    app.processEvents()
    frames.append(grab_window(win))  # grab frame
    durations.append(800)

    save_gif(frames, durations, "example_lfp.gif")

    save_gif(running_frames, running_durations, "example_lfp_short.gif")

    sys.exit(0)

if __name__ == "__main__":
    main()

