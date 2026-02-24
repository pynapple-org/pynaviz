"""
Test for PlotTsdFrame
"""
import pathlib
import sys
from types import SimpleNamespace

import numpy as np
import pygfx as gfx
import pynapple as nap
import pytest
from PIL import Image

import pynaviz as viz

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from config import TsdFrameConfig


def test_plot_tsdframe_init(dummy_tsdframe):
    v = viz.PlotTsdFrame(dummy_tsdframe)

    assert isinstance(v.data, nap.TsdFrame)
    assert v.cmap == "viridis"
    assert isinstance(v.renderer, gfx.Renderer)
    assert isinstance(v.scene, gfx.Scene)
    assert isinstance(v.ruler_x, gfx.Ruler)
    assert isinstance(v.ruler_y, gfx.Ruler)

    assert isinstance(v.controller, viz.controller.SpanController)
    assert isinstance(v.graphic, gfx.Line)
    v.close()

    assert hasattr(v._modes["lines"], "stream")

def test_plot_tsdframe_flush(dummy_tsdframe):
    v = viz.PlotTsdFrame(dummy_tsdframe)
    mode = v._modes["lines"]
    max_n = mode.stream._max_n

    # Check that the init flushed the data
    pos = v.graphic.geometry.positions.data
    for i, c in enumerate(dummy_tsdframe.columns):
        sl = mode._buffer_slices[c]
        np.testing.assert_almost_equal(
            pos[sl, 0], dummy_tsdframe.t[0:max_n].astype("float32")
        )
        np.testing.assert_almost_equal(
            pos[sl, 1], dummy_tsdframe.d[0:max_n, i].astype("float32")
        )

    # Flush the same range
    v._flush(start=0, end=1)
    for i, c in enumerate(dummy_tsdframe.columns):
        sl = mode._buffer_slices[c]
        np.testing.assert_almost_equal(
            pos[sl, 0], dummy_tsdframe.t[0:max_n].astype("float32")
        )
        np.testing.assert_almost_equal(
            pos[sl, 1], dummy_tsdframe.d[0:max_n, i].astype("float32")
        )

    # Flush a different range (all data fits in memory, buffer unchanged)
    v._flush(start=1, end=1.5)
    for i, c in enumerate(dummy_tsdframe.columns):
        sl = mode._buffer_slices[c]
        np.testing.assert_almost_equal(
            pos[sl, 0], dummy_tsdframe.t[0:max_n].astype("float32")
        )
        np.testing.assert_almost_equal(
            pos[sl, 1], dummy_tsdframe.d[0:max_n, i].astype("float32")
        )

    v.close()

@pytest.mark.parametrize(
    "window", [
        (0, 6.1),
        (9, 11)
    ]
)
def test_plot_tsdframe_large(dummy_tsdframe, window):
    v = viz.PlotTsdFrame(dummy_tsdframe, window_size=2.0)
    mode = v._modes["lines"]

    # ws = 2 second -> max_n = 201
    assert mode.stream._max_n == 201
    win_start, win_end = window

    v._flush(start=win_start, end=win_end)
    sl = mode.stream.get_slice(win_start, win_end)

    pos = v.graphic.geometry.positions.data
    for i, c in enumerate(dummy_tsdframe.columns):
        bsl = mode._buffer_slices[c]
        x = pos[bsl, 0]
        np.testing.assert_almost_equal(
            x[~np.isnan(x)], dummy_tsdframe.t[sl].astype("float32")
        )
        y = pos[bsl, 1]
        np.testing.assert_almost_equal(
            y[~np.isnan(y)], dummy_tsdframe.d[sl, i].astype("float32")
        )

    v.close()


def test_plot_tsdframe_min_max(tmp_path, dummy_tsdframe):
    path = tmp_path / "test.dat"
    mmap = np.memmap(path, mode="w+", shape=dummy_tsdframe.d.shape, dtype=dummy_tsdframe.d.dtype)
    mmap[:] = dummy_tsdframe.d[:]
    mmap.flush()
    tsdframe = nap.TsdFrame(t=dummy_tsdframe.t, d=mmap, columns=dummy_tsdframe.columns)

    v = viz.PlotTsdFrame(tsdframe, window_size=2.0)
    minmax = v._get_min_max()
    np.testing.assert_almost_equal(minmax[:,0], np.min(tsdframe.get(0, 2), 0))
    np.testing.assert_almost_equal(minmax[:, 1], np.max(tsdframe.get(0, 2), 0))

    v.close()


@pytest.mark.parametrize(
    "func, kwargs",
    TsdFrameConfig.parameters,
)
def test_plot_tsdframe_action(dummy_tsdframe, func, kwargs):
    v = viz.PlotTsdFrame(dummy_tsdframe)
    if func is not None:
        if isinstance(func, (list, tuple)):
            for n, k in zip(func, kwargs):
                getattr(v, n)(**k)
        else:
            getattr(v, func)(**kwargs)
    v.animate()
    image_data = v.renderer.snapshot()
    filename = TsdFrameConfig._build_filename(func, kwargs)
    image = Image.open(
        pathlib.Path(__file__).parent / "screenshots" / filename
    ).convert("RGBA")
    np.allclose(np.array(image), image_data)
    v.close()


# ── LinesMode tests ──────────────────────────────────────────────────────────

def test_lines_mode_buffer_layout(dummy_tsdframe):
    v = viz.PlotTsdFrame(dummy_tsdframe)
    mode = v._modes["lines"]
    n_cols = dummy_tsdframe.shape[1]
    max_n = mode.stream._max_n

    # Buffer has (max_n + 1) * n_channels rows, 3 columns (x, y, z)
    assert mode.buffer.shape == ((max_n + 1) * n_cols, 3)

    # Each channel slice has exactly max_n entries
    assert len(mode._buffer_slices) == n_cols
    for c in dummy_tsdframe.columns:
        sl = mode._buffer_slices[c]
        assert sl.stop - sl.start == max_n

    # NaN gaps between channels (the row right after each slice)
    for c in dummy_tsdframe.columns:
        sl = mode._buffer_slices[c]
        assert np.isnan(mode.buffer[sl.stop, 0])

    v.close()


def test_lines_mode_get_callbacks(dummy_tsdframe):
    # All data fits — no streaming callback needed
    v = viz.PlotTsdFrame(dummy_tsdframe)
    mode = v._modes["lines"]
    assert mode.stream._max_n >= dummy_tsdframe.shape[0]
    assert mode.get_callbacks() == []
    v.close()

    # Streaming needed
    v2 = viz.PlotTsdFrame(dummy_tsdframe, window_size=2.0)
    mode2 = v2._modes["lines"]
    assert mode2.stream._max_n < dummy_tsdframe.shape[0]
    callbacks = mode2.get_callbacks()
    assert len(callbacks) == 1
    assert callbacks[0] == mode2.stream.stream
    v2.close()


def test_lines_mode_rescale(dummy_tsdframe):
    v = viz.PlotTsdFrame(dummy_tsdframe)
    mode = v._modes["lines"]

    # Before sorting: rescale should return False
    assert mode.rescale("i") is False

    # Sort, then rescale
    v.sort_by("channel")
    buf_before = mode.buffer[:, 1].copy()
    result = mode.rescale("i")
    assert result is True
    # Buffer y-values should have changed
    assert not np.allclose(mode.buffer[:, 1], buf_before, equal_nan=True)

    v.close()


def test_lines_mode_update_visibility(dummy_tsdframe):
    v = viz.PlotTsdFrame(dummy_tsdframe)
    mode = v._modes["lines"]

    # All channels visible initially
    colors = mode.graphic.geometry.colors.data
    for _, sl in mode._buffer_slices.items():
        assert np.all(colors[sl, -1] == 1.0)

    # Hide first channel
    vis = v._manager.visible.copy()
    vis[0] = False
    v._manager.visible = vis
    mode.update_visibility()

    colors = mode.graphic.geometry.colors.data
    first_col = dummy_tsdframe.columns[0]
    sl = mode._buffer_slices[first_col]
    assert np.all(colors[sl, -1] == 0.0)

    # Other channels still visible
    for c in dummy_tsdframe.columns[1:]:
        sl = mode._buffer_slices[c]
        assert np.all(colors[sl, -1] == 1.0)

    v.close()


# ── ImageMode tests ──────────────────────────────────────────────────────────

def test_image_mode_init(dummy_tsdframe):
    v = viz.PlotTsdFrame(dummy_tsdframe, display_mode="image")
    mode = v._modes["image"]

    assert isinstance(v.graphic, gfx.Image)
    assert mode.buffer.shape[0] == dummy_tsdframe.shape[1]
    assert hasattr(mode, "texture")
    assert mode._n_visible == dummy_tsdframe.shape[1]
    assert isinstance(v.controller, viz.controller.SpanYLockController)

    v.close()


def test_image_mode_flush(dummy_tsdframe):
    v = viz.PlotTsdFrame(dummy_tsdframe, display_mode="image")
    mode = v._modes["image"]

    # Buffer should have data (not all zeros after flush)
    assert not np.all(mode.buffer == 0)

    # Time alignment
    assert mode.graphic.local.x == pytest.approx(float(dummy_tsdframe.t[0]))

    # Buffer rows match data channels (all-data path, no reorder since unsorted)
    n = min(dummy_tsdframe.shape[0], mode.stream._max_n)
    for i in range(dummy_tsdframe.shape[1]):
        np.testing.assert_almost_equal(
            mode.buffer[i, :n],
            dummy_tsdframe.d[:n, i].astype("float32"),
        )

    v.close()


def test_image_mode_rescale(dummy_tsdframe):
    v = viz.PlotTsdFrame(dummy_tsdframe, display_mode="image")
    mode = v._modes["image"]

    mode.graphic.material.clim = (-1, 1)

    # Narrow (increase)
    mode.rescale("i")
    lo, hi = mode.graphic.material.clim
    assert lo > -1 and hi < 1

    # Reset and widen (decrease)
    mode.graphic.material.clim = (-1, 1)
    mode.rescale("d")
    lo, hi = mode.graphic.material.clim
    assert lo < -1 and hi > 1

    v.close()


def test_image_mode_color_by(dummy_tsdframe):
    v = viz.PlotTsdFrame(dummy_tsdframe, display_mode="image")
    mode = v._modes["image"]

    mode.color_by("plasma", "test", 0, 10, lambda vals: None, {})
    assert mode.graphic.material.map == gfx.cm.plasma
    lo, hi = mode.graphic.material.clim
    assert lo == 0 and hi == 10

    v.close()


def test_image_mode_visibility(dummy_tsdframe):
    v = viz.PlotTsdFrame(dummy_tsdframe, display_mode="image")
    mode = v._modes["image"]
    n_channels = dummy_tsdframe.shape[1]

    assert mode._n_visible == n_channels

    # Hide first channel
    vis = v._manager.visible.copy()
    vis[0] = False
    v._manager.visible = vis
    mode.update_visibility()

    assert mode._n_visible == n_channels - 1
    assert mode.texture.data.shape[0] == n_channels - 1

    v.close()


def test_image_mode_reorder_rows(dummy_tsdframe):
    v = viz.PlotTsdFrame(dummy_tsdframe, display_mode="image")
    mode = v._modes["image"]
    n_cols = dummy_tsdframe.shape[1]

    # No reorder when not sorted
    raw = np.arange(n_cols * 10, dtype="float32").reshape(n_cols, 10)
    result = mode._reorder_image_rows(raw)
    np.testing.assert_array_equal(result, raw)

    # Sort, then reorder should permute rows
    v.sort_by("channel")
    result = mode._reorder_image_rows(raw)
    offsets = np.array([
        v._manager.data.loc[c]["offset"] for c in dummy_tsdframe.columns
    ])
    order = np.argsort(offsets)
    np.testing.assert_array_equal(result, raw[order])

    v.close()


def test_image_mode_get_visible_buffer(dummy_tsdframe):
    v = viz.PlotTsdFrame(dummy_tsdframe, display_mode="image")
    mode = v._modes["image"]

    # All visible: returns full buffer
    visible_buf = mode._get_visible_buffer()
    assert visible_buf.shape[0] == dummy_tsdframe.shape[1]

    # Hide one channel
    vis = v._manager.visible.copy()
    vis[0] = False
    v._manager.visible = vis
    visible_buf = mode._get_visible_buffer()
    assert visible_buf.shape[0] == dummy_tsdframe.shape[1] - 1

    v.close()


# ── XvsYMode tests ───────────────────────────────────────────────────────────

def test_xvsy_mode_init(dummy_tsdframe):
    v = viz.PlotTsdFrame(dummy_tsdframe)
    v.plot_x_vs_y(x_col=0, y_col=1)
    mode = v._modes["x_vs_y"]

    assert isinstance(mode.graphic, gfx.Line)
    assert isinstance(mode.time_point, gfx.Points)
    assert mode.buffer.shape == (len(dummy_tsdframe), 3)

    # Buffer x/y match the two selected columns
    np.testing.assert_almost_equal(
        mode.buffer[:, 0], dummy_tsdframe.d[:, 0].astype("float32")
    )
    np.testing.assert_almost_equal(
        mode.buffer[:, 1], dummy_tsdframe.d[:, 1].astype("float32")
    )

    # Time point rendered above the line (z = 1)
    assert mode.time_point.geometry.positions.data[0, 2] == 1.0

    v.close()


def test_xvsy_mode_update_parameters(dummy_tsdframe):
    v = viz.PlotTsdFrame(dummy_tsdframe)
    mode = v._modes["x_vs_y"]

    mode.update_parameters(x_col=2, y_col=3, color="blue", thickness=3.0, markersize=20.0)
    assert mode.x_col == 2
    assert mode.y_col == 3
    assert mode.color == "blue"
    assert mode.thickness == 3.0
    assert mode.markersize == 20.0

    v.close()


def test_xvsy_mode_update_buffer(dummy_tsdframe):
    v = viz.PlotTsdFrame(dummy_tsdframe)
    v.plot_x_vs_y(x_col=0, y_col=1)
    mode = v._modes["x_vs_y"]

    # Move to frame 50
    mode._update_buffer(50)
    marker = mode.time_point.geometry.positions.data[0]
    line = mode.graphic.geometry.positions.data[50]
    np.testing.assert_almost_equal(marker[0:2], line[0:2])

    # Move to frame 100
    mode._update_buffer(100)
    marker = mode.time_point.geometry.positions.data[0]
    line = mode.graphic.geometry.positions.data[100]
    np.testing.assert_almost_equal(marker[0:2], line[0:2])

    v.close()


def test_xvsy_mode_get_callbacks(dummy_tsdframe):
    v = viz.PlotTsdFrame(dummy_tsdframe)
    mode = v._modes["x_vs_y"]
    callbacks = mode.get_callbacks()
    assert len(callbacks) == 1
    assert callbacks[0] == mode._update_buffer

    v.close()


# ── Mode switching ───────────────────────────────────────────────────────────

def test_toggle_display_mode(dummy_tsdframe):
    v = viz.PlotTsdFrame(dummy_tsdframe)

    # Starts in lines mode
    assert v._display_mode == "lines"
    assert isinstance(v.graphic, gfx.Line)

    # Toggle to image
    v._toggle_display_mode(SimpleNamespace(type="key_down", key="m"))
    assert v._display_mode == "image"
    assert isinstance(v.graphic, gfx.Image)
    assert isinstance(v.controller, viz.controller.SpanYLockController)

    # Toggle back to lines
    v._toggle_display_mode(SimpleNamespace(type="key_down", key="m"))
    assert v._display_mode == "lines"
    assert isinstance(v.graphic, gfx.Line)
    assert not isinstance(v.controller, viz.controller.SpanYLockController)
    assert isinstance(v.controller, viz.controller.SpanController)

    v.close()
