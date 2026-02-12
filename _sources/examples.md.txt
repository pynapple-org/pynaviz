Examples gallery
=================


Spikes & 2-d projection & video
-------------------------------

In this example, head-direction neurons are being recorded during wake. 
The video shows the animal's movement in an open field.
The manifold widget shows a projection of the neural activity onto a 2D space using Isomap.




![example_head_direction](/examples/example_head_direction.gif)


:::{dropdown} See the example code
:color: info
:icon: info

```{code} ipython
import os
import requests
import pynapple as nap
from pynaviz import scope
import pynaviz as viz

nwb_file = "A5044-240404A_wake.nwb"
avi_file = "A5044-240404A_wake.avi"

files = os.listdir(".")
if nwb_file not in files:
    url = "https://osf.io/um4nb/download"
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(nwb_file, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):  # Stream in chunks
                f.write(chunk)
if avi_file not in files:
    url = "https://osf.io/gyu2h/download"
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(avi_file, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):  # Stream in chunks
                f.write(chunk)

# Load data
data = nap.load_file("A5044-240404A_wake.nwb")
units = data["units"]
manifold = data["manifold"]
video = viz.VideoWidget("A5044-240404A_wake.avi")

# Open the GUI
scope({"units": units, "manifold": manifold, "video": video})

```
:::

***

Local Field Potential (LFP)
---------------------------

In this example, a Local Field Potential (LFP) signal is being recorded during sleep in mice.

![example_head_direction](/examples/example_lfp.gif)

:::{dropdown} See the example code
:color: info
:icon: info

```{code} ipython
import pynapple as nap
import numpy as np
from pynaviz import scope

path_to_dat_file = "lfp.dat"
fs = 20000 # Sampling frequency in Hz
num_channels = 16 # Number of channels in the recording
tsdframe = nap.misc.load_eeg(
    path_to_dat_file,
    n_channels=num_channels,
    frequency=fs
)
tsdframe.channel = np.arange(0, num_channels) # Add channel metadata
tsdframe.group = np.hstack([[0]*10, [1]*6]) # Add group metadata


# Open the GUI
scope({"lfp":tsdframe})

```
:::

***

DLC pose estimation
-----------------------

This example shows pose estimation data obtained using DeepLabCut (DLC).
The data are from the DeepLabCut GitHub repository. 


![example_dlc](/examples/example_dlc_pose.gif)


:::{dropdown} See the example code
:color: info
:icon: info

```{code} ipython
import pynapple as nap
import numpy as np
from pynaviz import scope
from pathlib import Path


# Path to the video
video_path = "docs/examples/m3v1mp4.mp4"

# Output of deeplabcut
df = pd.read_hdf("docs/examples/m3v1mp4DLC_Resnet50_openfieldOct30shuffle1_snapshot_best-70.h5")
df.columns = [f"{bodypart}_{coord}" for _, bodypart, coord in df.columns]
df = df[[c for c in df.columns if c.endswith(("_x", "_y"))]]
y_col = [c for c in df.columns if c.endswith("_y")]
df[y_col] = df[y_col]*-1 + 480 # Flipping y axis
skeleton = nap.TsdFrame(t=df.index.values/30, d=df.values, columns=df.columns)


# Open the GUI
scope({"skeleton": skeleton, "video": video_path})

```
:::


***

Multiple videos
-----------------------

This example shows multiple videos being recorded simultaneously from different angles.
The data are from the International Brain Lab (IBL).

![example_videos](/examples/example_videos.gif)


:::{dropdown} See the example code
:color: info
:icon: info

```{code} ipython
import pynapple as nap
import numpy as np
from pynaviz import scope
from pathlib import Path
from one.api import ONE

# Load IBL session
one = ONE()
eid = "ebce500b-c530-47de-8cb1-963c552703ea"

# Videos
ibl_path = Path(os.path.expanduser("~/Downloads/ONE/"))
videos = {}
for label in ["left", "body", "right"]:
    video_path = (
            ibl_path
            / f"openalyx.internationalbrainlab.org/churchlandlab_ucla/Subjects/MFD_09/2023-10-19/001/raw_video_data/_iblrig_{label}Camera.raw.mp4"
    )
    if not video_path.exists():
        one.load_dataset(eid, f"*{label}Camera.raw*", collection="raw_video_data")
    times = one.load_object(eid, f"{label}Camera", collection="alf", attribute=["times*"])["times"]
    # The videos seem to start at 5 seconds. Removing artificially 5 seconds for the demo
    times = times - 5
    videos[label] = viz.VideoWidget(video_path, t=times)


# Open the GUI
scope(videos)

```
:::


***

Video & Event & Trial intervals
---------------------------------

This example shows spikes being recorded during a trial-based task. 
The data are from the International Brain Lab (IBL).

![example_trials](/examples/example_trials.gif)


:::{dropdown} See the example code
:color: info
:icon: info

```{code} ipython
import pynapple as nap
import numpy as np
from pynaviz import scope
from pathlib import Path
from one.api import ONE

# Load IBL session
one = ONE()
eid = "ebce500b-c530-47de-8cb1-963c552703ea"

# Videos
ibl_path = Path(os.path.expanduser("~/Downloads/ONE/"))
vars = {}
for label in ["left", "body", "right"]:
    video_path = (
            ibl_path
            / f"openalyx.internationalbrainlab.org/churchlandlab_ucla/Subjects/MFD_09/2023-10-19/001/raw_video_data/_iblrig_{label}Camera.raw.mp4"
    )
    if not video_path.exists():
        one.load_dataset(eid, f"*{label}Camera.raw*", collection="raw_video_data")
    times = one.load_object(eid, f"{label}Camera", collection="alf", attribute=["times*"])["times"]
    # The videos seem to start at 5 seconds. Removing artificially 5 seconds for the demo
    times = times - 5
    vars[label] = viz.VideoWidget(video_path, t=times)

timings = one.load_object(eid, "trials", collection="alf")
licks = nap.Ts(one.load_object(eid, "licks", collection="alf")["times"])
trials = nap.IntervalSet(timings["intervals"])

vars["trials"] = trials
vars["licks"] = licks

# Open the GUI
scope(videos)

```
:::


