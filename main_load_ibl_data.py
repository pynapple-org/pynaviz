import pynapple as nap
from pynaviz import scope
import pynaviz as viz
from PyQt6.QtWidgets import QApplication
from one.api import ONE
import ibllib.io.video as vidio
from brainbox.io.one import SpikeSortingLoader

one = ONE()

eid = 'ebce500b-c530-47de-8cb1-963c552703ea'
labels = ['left', 'right', 'body']

# Loading the videos
for label in labels:
    video_body = one.load_dataset(eid, f'*{label}Camera.raw*', collection='raw_video_data')

# # Loading the spikes
# ssl = SpikeSortingLoader(eid=eid, one=one)
# spikes, clusters, channels = ssl.load_spike_sorting()
# clusters = ssl.merge_clusters(spikes, clusters, channels)
# waveforms = ssl.load_spike_sorting_object('waveforms')  # loads in the template waveforms
