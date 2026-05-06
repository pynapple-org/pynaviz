GUI reference
=============

## Command line

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

## Python API — `scope()`

```python
from pynaviz import scope
```

| Example | Input type | Notes |
|---|---|---|
| `scope({"lfp": tsdframe, "spikes": tsgroup})` | `dict` | Keys become display names in the variable panel |
| `scope([tsdframe, tsgroup, interval_set])` | `list` / `tuple` | Names inferred from class (`TsdFrame`, `TsGroup`, …) |
| `scope(tsgroup)` | `nap.TsGroup` | Same for all pynapple objects (`Tsd`, `TsdFrame`, `TsdTensor`, `IntervalSet`, `Ts`) |
| `scope(nap.load_file("data.nwb"))` | `nap.NWBFile` | All contained objects unpacked individually |
| `scope(nap.EphysReader("rec/", format="NeuroScopeIO"))` | `nap.EphysReader` | All contained objects unpacked individually |
| `scope("data.nwb")` | `str` / `pathlib.Path` — `.nwb` | Loaded via pynapple; objects unpacked |
| `scope("data.npz")` | `str` / `pathlib.Path` — `.npz` | Must contain a single pynapple object |
| `scope("recording.mp4")` | `str` / `pathlib.Path` — video | `.mp4`, `.avi`, `.mov`, `.mkv` supported |
| `scope("recording.plx")` | `str` / `pathlib.Path` — ephys file | Loaded via `nap.EphysReader`; format auto-detected |
| `scope("rec/")` | `str` / `pathlib.Path` — directory | Passed to `nap.EphysReader`; auto-detects NeuroScopeIO |

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
| `←` / `→` | Pan left / right by one page |
| `y` | Lock / unlock y-axis |
| `x` | Lock / unlock x-axis |
| `Ctrl+←` / `Ctrl+→` | Jump to previous / next superposed epoch (requires an `IntervalSet` overlay) |
| `i` / `d` | Increase / decrease contrast (TsdFrame) or marker size (TsGroup) |
| `n` / `p` | Jump to next / previous interval or timestamp (IntervalSet & Ts) |