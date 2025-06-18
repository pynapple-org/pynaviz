"""
Test script
"""
import numpy as np
import os
import pynapple as nap
from PyQt6.QtWidgets import QApplication
import pynaviz as viz
import sys

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

tsdframe = nap.misc.load_eeg(
    os.path.expanduser("~/Dropbox/A2929-200711/A2929-200711.dat"),
    n_channels=16,
    frequency=20000
    )
tsdframe.group = np.hstack((np.zeros(10), np.ones(6)))
tsdframe.channel = np.arange(0, 16)
tsdframe.random = np.random.randn(16)


# path = "/mnt/ceph/users/gviejo/LMN-ADN/A5011/A5011-201014A/A5011-201014A.dat"
# num_channels, fs, shank_to_channel, shank_to_keep = loadXML(os.path.dirname(path))
# tsdframe = nap.misc.load_eeg(
#     path,
#     n_channels=num_channels,
#     frequency=20000
#     )
# ch = np.hstack([shank_to_channel[i] for i in shank_to_channel])
# tsdframe.channel = np.argsort(ch)
# gr = np.zeros(num_channels)
# for i in shank_to_channel:
#     gr[shank_to_channel[i]] = i
# tsdframe.group = gr


app = QApplication([])

v = viz.TsdFrameWidget(tsdframe)
v.plot.sort_by("channel")#, mode="descending")
v.plot.group_by("group")
v.plot.color_by("channel")
v.show()

if __name__ == "__main__":
    app.exit(app.exec())
