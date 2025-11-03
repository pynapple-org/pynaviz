# Pynaviz

**Python Neural Analysis Visualization**

<table>
  <tr>
    <td><img src="docs/examples/example_dlc_pose_short.gif" width="100%"></td>
    <td><img src="docs/examples/example_head_direction_short.gif" width="100%"></td>
  </tr>
  <tr>
    <td><img src="docs/examples/example_lfp_short.gif" width="100%"></td>
    <td><img src="docs/examples/example_videos_short.gif" width="100%"></td>
  </tr>
</table>


**Pynaviz** provides interactive, high-performance visualizations designed to work seamlessly with [Pynapple](https://github.com/pynapple-org/pynapple) time series and video data. It allows synchronized exploration of neural signals and behavioral recordings.


---

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/pynapple-org/pynaviz/blob/main/LICENSE)
[![CI](https://github.com/pynapple-org/pynaviz/actions/workflows/ci.yml/badge.svg)](https://github.com/pynapple-org/pynaviz/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/pynapple-org/pynaviz/graph/badge.svg?token=852A4EHI1Q)](https://codecov.io/gh/pynapple-org/pynaviz)


## Installation

We recommend using the **Qt-based interface** for the best interactive experience:

```bash
pip install pynaviz[qt]
```



To check if the installation was successful with qt, try running:

```bash
pynaviz
````

If Qt is not available on your system, you can still use the fallback rendering engine (via PyGFX):

```bash
pip install pynaviz
```

## Basic usage

Once installed (and if Qt installation worked), you can explore Pynapple data interactively using the `scope` interface:

```python
import pynapple as nap
import numpy as np
from pynaviz import scope

# Create some example time series
tsd = nap.Tsd(t=np.arange(100), d=np.random.randn(100))

# Create a TsdFrame with metadata
tsdframe = nap.TsdFrame(
    t=np.arange(10000),
    d=np.random.randn(10000, 10),
    metadata={"label": np.random.randn(10)}
)

# Launch the visualization GUI
scope(globals())

```

This will launch an interactive viewer where you can inspect time series, event data, and video tracks in a synchronized environment.
