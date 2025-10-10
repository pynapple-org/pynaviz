User guide
==========


Graphical User Interface (GUI)
-----------------------------

The GUI is designed to visualize and interact with time series data stored in pynapple objects. It provides a user-friendly interface to explore the data.

To launch the GUI, you can use the `scope` function from the `pynaviz` package. This function takes a dictionary of pynapple objects and opens the GUI.

```python
import pynapple as nap
import numpy as np
from pynaviz import scope
# Create some example data
tsd = nap.Tsd(t=np.arange(1000), d=np.sin(np.arange(1000) * 0.1))
tsdframe = nap.TsdFrame(
    t=np.arange(1000),
    d=np.stack((
        np.cos(2 * np.pi * np.arange(0, 100, 0.1)),
        np.sin(2 * np.pi * np.arange(0, 100, 0.1))
    )).T,
    metadata={"label": np.random.randn(2)}
)
# Launch the GUI
scope({"tsd": tsd, "tsdframe": tsdframe})
```

You can also pass a path to a video file to visualize it alongside the time series data.

```python
scope({"tsd": tsd, "video": "path_to_video.mp4"})
```

The GUI supports various pynapple objects, including `Tsd`, `TsdFrame`, `TsGroup`, `IntervalSet`, `Ts`, and `TsdTensor`. Each object type has its own visualization and interaction capabilities.
It also support reading NWB files.

```python
scope("path_to_file.nwb")
```

To visualize NWB files and videos together, you can pass them together in a dictionary or a list.

```python
scope({"nwb": "path_to_file.nwb", "video": "path_to_video.mp4"})
```
or
```python
scope(["path_to_file.nwb", "path_to_video.mp4"])
```

Layout of the GUI can be saved. To reload a saved layout, you can pass the path to the layout file using the `layout` parameter.

```python
scope({"tsd": tsd, "tsdframe": tsdframe}, layout_path="path_to_layout.json")
``` 

The same can be done at the shell level using the `pynaviz` command.

```bash
$ pynaviz path_to_file.nwb --layout_path path_to_layout.json
```

With multiples files:

```bash
$ pynaviz path_to_file.nwb path_to_video.mp4 --layout_path path_to_layout.json
``` 


Simple visuals
--------------

Each pynapple object is mapped into a simple visuals using pygfx.

::::{grid} 1 2 2 2

:::{grid-item-card}

**Tsd**

 <a href="/user_guide/tsd.html">
    <img src="_static/screenshots/test_plot_tsd.png"
         alt="Tsd Image"
         style="width: 250px;" />
  </a>

:::

:::{grid-item-card}

**TsdFrame**

 <a href="/user_guide/tsdframe.html">
    <img src="_static/screenshots/test_plot_tsdframe.png"
         alt="TsdFrame Image"
         style="width: 250px;" />
  </a>

:::


:::{grid-item-card}

**TsGroup**

 <a href="/user_guide/tsgroup.html">
    <img src="_static/screenshots/test_plot_tsgroup.png"
         alt="TsGroup Image"
         style="width: 250px;" />
  </a>


:::


:::{grid-item-card}

**IntervalSet**

 <a href="/user_guide/intervalset.html">
    <img src="_static/screenshots/test_plot_intervalset.png"
         alt="IntervalSet Image"
         style="width: 250px;" />
  </a>


:::

:::{grid-item-card}

**Ts**

 <a href="/user_guide/ts.html">
    <img src="_static/screenshots/test_plot_ts.png"
         alt="Ts Image"
         style="width: 250px;" />
  </a>


:::

:::{grid-item-card}

**TsdTensor**

 <a href="/user_guide/tsdtensor.html">
    <img src="_static/screenshots/test_plot_tsdtensor.png"
         alt="TsdTensor Image"
         style="width: 250px;" />
  </a>

:::



::::


Video handlers
--------------

A video handler is provided to display videos synchronized to time series. 

<a href="/user_guide/video.html">
    <img src="_static/screenshots/test_plot_videohandler.png"
        alt="PlotVideo Image"
        style="width: 250px;" />
</a>


```{toctree}
Displaying Videos <user_guide/video>
```




Qt visuals
----------

Qt widgets wraps the simple visuals with additional interactive functions.


```{toctree}
Qt Widgets <user_guide/qt>
```


Time syncing
------------

```{toctree}
Time Series Synchronization <user_guide/syncing>
```






