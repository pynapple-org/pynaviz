# Pynaviz

**Python Neural Analysis Visualization**

**Pynaviz** provides interactive, high-performance visualizations designed to work seamlessly with [Pynapple](https://github.com/pynapple-org/pynapple) time series and video data. It allows synchronized exploration of neural signals and behavioral recordings.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/pynapple-org/pynaviz/blob/main/LICENSE)

<p align="center">
  <img src="in_construction.png" alt="In Construction" width="300"/>
</p>

> ⚠️ **Note**  
> This package is under active development. Interfaces and features may change.

---

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