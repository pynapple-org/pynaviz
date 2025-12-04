# Video

## Type of videos supported

This library supports common video formats such as MP4, AVI, MOV, and MKV. For best compatibility, use MP4 (H.264 codec).

## Reading video

The class `VideoHandler` can read video files and provide frame-by-frame access. It supports various formats and allows synchronization with time series data.

```python
import pynaviz as viz

handler = viz.VideoHandler("path_to_video.mp4")

handler.get(10.0) # Returns the frame closest to 10 seconds
```

You can specify the timestamps for each frame if needed:

```python
import numpy as np
import pynaviz as viz
timestamps = np.arange(0, 100, 0.033)  # Example timestamps for 30 FPS video
handler = viz.VideoHandler("path_to_video.mp4", time=timestamps)
```

## Simple video display

```python
import pynaviz as viz

handler = viz.VideoHandler("path_to_video.mp4")

v = viz.PlotVideo(handler)
v.show()
```

Videos can be display by providing the path directly:

```python
import pynaviz as viz

v = viz.PlotVideo("path_to_video.mp4")
v.show()
```


## Video display with pyqt6

```python
import pynaviz as viz

handler = viz.VideoHandler("path_to_video.mp4")

v = viz.VideoWidget(handler)
v.show()
```

Videos can be display by providing the path directly:

```python
import pynaviz as viz

v = viz.VideoWidget("path_to_video.mp4")
v.show()
```