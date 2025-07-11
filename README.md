# Pynaviz

Python Neural Analysis Visualization

Interactive visualizations that keep **pynapple time series** and/or **video frames** in sync.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/pynapple-org/pynaviz/blob/main/LICENSE)


<img src="in_construction.png" alt="image" width="300"/>

> **Warning**
> ⚠️ This package is still in construction!

## Installation

The best way to use pynaviz is through the Qt Application.

```bash
git clone https://github.com/pynapple-org/pynaviz.git
cd pynaviz
pip install -e '.[qt]'
```

If Qt does not work, it is still possible to use the default of pygfx.

```bash
git clone https://github.com/pynapple-org/pynaviz.git
cd pynaviz
pip install -e .
```

## Basic usage

If Qt is working, interactive visualizations can be constructed through the GUI application :

```python
import pynapple as nap
import numpy as np
from pynaviz import scope

# Load some data
tsd = nap.Tsd(t=np.arange(100), d=np.random.randn(100))
tsdframe = nap.TsdFrame(t=np.arange(10000), d=np.random.randn(10000, 10), metadata={"label": np.random.randn(10)})

scope(globals())

```
