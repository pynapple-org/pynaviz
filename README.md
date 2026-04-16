# Pynaviz

**Python Neural Analysis Visualization**

<table>
  <tr>
    <td><img src="docs/examples/example_dlc_pose_short.gif" width="100%"></td>
    <td><img src="docs/examples/example_head_direction_short.gif" width="100%"></td>
    <td><img src="docs/examples/example_lfp_short.gif" width="100%"></td>
    <td><img src="docs/examples/example_videos_short.gif" width="100%"></td>
  </tr>
</table>

**Pynaviz** provides interactive, high-performance visualizations designed to work seamlessly with [Pynapple](https://github.com/pynapple-org/pynapple) time series and video data. It allows synchronized exploration of neural signals and behavioral recordings. It is built on top of [pygfx](https://pygfx.org/), a modern GPU-based rendering engine.

---

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/pynapple-org/pynaviz/blob/main/LICENSE)
[![CI](https://github.com/pynapple-org/pynaviz/actions/workflows/ci.yml/badge.svg)](https://github.com/pynapple-org/pynaviz/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/pynapple-org/pynaviz/graph/badge.svg?token=852A4EHI1Q)](https://codecov.io/gh/pynapple-org/pynaviz)


## Installation

We recommend using the **Qt-based interface** for the best interactive experience:

```bash
pip install pynaviz[qt]
```

If Qt is not available on your system, you can still use the fallback rendering engine (via PyGFX):

```bash
pip install pynaviz
```

## Quick start

### From the command line

```bash
$ pynaviz [files ...] [-l layout.json] [-f FORMAT]
```

| Example | Argument | Notes |
|---|---|---|
| `$ pynaviz` | *(no arguments)* | Opens an empty viewer |
| `$ pynaviz data.nwb` | `files` | One or more `.nwb` files; objects unpacked individually |
| `$ pynaviz data.npz` | `files` | One or more `.npz` files; must contain a single pynapple object each |
| `$ pynaviz recording.mp4` | `files` | One or more video files (`.mp4`, `.avi`, `.mov`, `.mkv`) |
| `$ pynaviz data.nwb recording.mp4` | `files` | Multiple files of different types can be mixed |
| `$ pynaviz recording.plx` | `files` | Ephys file; format auto-detected via `nap.EphysReader` |
| `$ pynaviz rec/` | `files` | Directory; auto-detected as NeuroScopeIO if `.dat` + `.xml` are present |
| `$ pynaviz rec/ -f NeuroScopeIO` | `files` + `-f` | Directory with explicit Neo IO format |
| `$ pynaviz recording.plx -f PlexonIO` | `files` + `-f` | Ephys file with explicit format |
| `$ pynaviz -l layout.json` | `-l` / `--layout` | Restore a previously saved layout (`.json`) |
| `$ pynaviz data.nwb -l layout.json` | `files` + `-l` | Load files and restore layout simultaneously |

### From a Python script

```python
from pynaviz import scope
```

The `scope` function accepts many input types:

| Example | Input type | Notes |
|---|---|---|
| `scope({"lfp": tsdframe, "spikes": tsgroup})` | `dict` | Keys become display names in the variable panel |
| `scope([tsdframe, tsgroup, interval_set])` | `list` / `tuple` | Names inferred from class (`TsdFrame`, `TsGroup`, …) |
| `scope(tsgroup)` | `nap.TsGroup` | Collection of spike trains. Same for all pynapple objects (`Tsd`, `TsdFrame`, …) |
| `scope(nap.load_file("data.nwb"))` | `nap.NWBFile` | All contained objects unpacked individually |
| `scope(nap.EphysReader("rec/", format="NeuroScopeIO"))` | `nap.EphysReader` | All contained objects unpacked individually |
| `scope("data.nwb")` | `str` / `pathlib.Path` — `.nwb` | Loaded via pynapple, objects unpacked |
| `scope("data.npz")` | `str` / `pathlib.Path` — `.npz` | Must contain a single pynapple object |
| `scope("recording.mp4")` | `str` / `pathlib.Path` — video | `.mp4`, `.avi`, `.mov`, `.mkv` supported |
| `scope("recording.plx")` | `str` / `pathlib.Path` — ephys file | Loaded via `nap.EphysReader`; format auto-detected |
| `scope("rec/")` | `str` / `pathlib.Path` — directory | Directory passed to `nap.EphysReader`; auto-detects NeuroScopeIO |

See the [User Guide](https://pynaviz.readthedocs.io) for more details.


## Keyboard shortcuts

**Global**

| Shortcut | Action |
|---|---|
| `Space` | Play / pause |
| `Ctrl+S` | Save layout |
| `Ctrl+O` | Load layout |

**Per-dock** (active when the mouse is over the canvas)

| Shortcut | Action |
|---|---|
| `r` | Reset view |
| `← / →` | Pan left / right by one page |
| `y` | Lock / unlock y-axis |
| `x` | Lock / unlock x-axis |
| `Ctrl+← / Ctrl+→` | Jump to previous / next superposed epoch (requires an IntervalSet overlay) |
| `i` / `d` | Increase / decrease contrast (TsdFrame) or marker size (TsGroup) |
| `n` / `p` | Jump to next / previous interval or timestamp (IntervalSet & Ts) |


## Basic usage

This example demonstrates how to create some example time series and launch the visualization GUI:

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