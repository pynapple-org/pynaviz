"""
Test for PlotTsdFrame
"""
import pathlib
import sys

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

    assert hasattr(v, "_stream")

def test_plot_tsdframe_flush(dummy_tsdframe):
    v = viz.PlotTsdFrame(dummy_tsdframe)

    # Check that the init flushed the data
    pos = v.graphic.geometry.positions.data
    for col, start in zip(range(5), range(0, 5000, 1000)):
        sl = slice(start + col, start + col + 1000)
        np.testing.assert_almost_equal(
            pos[sl, 0], dummy_tsdframe.t[0:1000].astype("float32")
        )
        np.testing.assert_almost_equal(
            pos[sl, 1], dummy_tsdframe.d[0:1000, col].astype("float32")
        )

    # Flush the same slice
    v._flush()
    for col, start in zip(range(5), range(0, 5000, 1000)):
        sl = slice(start + col, start + col + 1000)
        np.testing.assert_almost_equal(
            pos[sl, 0], dummy_tsdframe.t[0:1000].astype("float32")
        )
        np.testing.assert_almost_equal(
            pos[sl, 1], dummy_tsdframe.d[0:1000, col].astype("float32")
        )

    # Flush a different slice
    v._flush(dummy_tsdframe.get_slice(1, 1.5))
    for col, start in zip(range(5), range(0, 5000, 1000)):
        sl = slice(start + col, start + col + 1000)
        np.testing.assert_almost_equal(
            pos[sl, 0], dummy_tsdframe.t[0:1000].astype("float32")
        )
        np.testing.assert_almost_equal(
            pos[sl, 1], dummy_tsdframe.d[0:1000, col].astype("float32")
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

    # ws = 2 second -> max_n = 201
    assert v._stream._max_n == 201
    start, end = window

    sl = dummy_tsdframe._get_slice(start, end, n_points=int(v._stream._max_n))
    v._flush(sl)

    pos = v.graphic.geometry.positions.data
    for col, start in zip(range(5), range(0, 1000, 200)):
        sl2 = slice(start + 2*col, start + 2*col + 200 + 1)
        x = pos[sl2, 0]
        np.testing.assert_almost_equal(
            x[~np.isnan(x)], dummy_tsdframe.t[sl].astype("float32")
        )
        y = pos[sl2, 1]
        np.testing.assert_almost_equal(
            y[~np.isnan(y)], dummy_tsdframe.d[sl, col].astype("float32")
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
