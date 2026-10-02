"""
Test for viz.PlotTsGroup.
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
from config import TsGroupConfig


def test_plot_tsgroup_init(dummy_tsgroup):
    v = viz.viz.PlotTsGroup(dummy_tsgroup)

    assert isinstance(v.controller, viz.controller.SpanController)
    v.close()


@pytest.mark.parametrize(
    "func, kwargs",
    TsGroupConfig.parameters,
)
def test_plot_tsgroup_action(dummy_tsgroup, func, kwargs):
    v = viz.viz.PlotTsGroup(dummy_tsgroup)
    if func is not None:
        if isinstance(func, (list, tuple)):
            for n, k in zip(func, kwargs):
                getattr(v, n)(**k)
        else:
            getattr(v, func)(**kwargs)
    v.animate()
    image_data = v.renderer.snapshot()
    filename = TsGroupConfig._build_filename(func, kwargs)
    image = Image.open(pathlib.Path(__file__).parent / "screenshots" / filename).convert("RGBA")
    np.allclose(np.array(image), image_data)
    v.close()


def test_tsgroup_ts_entries_remain_spike_rasters():
    data = nap.TsGroup(
        {
            0: nap.Ts(t=[0.1, 0.4]),
            1: nap.Ts(t=[0.2, 0.8]),
        }
    )
    plot = viz.PlotTsGroup(data)
    assert plot._entry_kind == "ts"
    assert not plot._continuous
    assert all(isinstance(graphic, gfx.Points) for graphic in plot.graphic.values())
    np.testing.assert_allclose(
        plot.graphic[0].geometry.positions.data[:, 0],
        data[0].t,
    )
    plot.close()


def test_tsgroup_tsd_entries_use_independent_timestamps():
    data = nap.TsGroup(
        {
            0: nap.Tsd(
                t=[0.0, 0.5, 1.0],
                d=[1.0, 2.0, 3.0],
            ),
            1: nap.Tsd(
                t=[0.2, 0.7],
                d=[4.0, 5.0],
            ),
        }
    )
    plot = viz.PlotTsGroup(data)
    assert plot._entry_kind == "tsd"
    assert plot._continuous
    assert all(isinstance(graphic, gfx.Line) for graphic in plot.graphic.values())
    np.testing.assert_allclose(
        plot.graphic[0].geometry.positions.data[:, 0],
        data[0].t,
    )
    np.testing.assert_allclose(
        plot.graphic[1].geometry.positions.data[:, 0],
        data[1].t,
    )
    np.testing.assert_allclose(
        plot.graphic[1].geometry.positions.data[:, 1],
        data[1].d,
    )
    plot.close()


def test_tsgroup_tsdframe_creates_one_line_per_column():
    data = nap.TsGroup(
        {
            0: nap.TsdFrame(
                t=[0.0, 0.5, 1.0],
                d=np.array(
                    [
                        [1.0, 4.0],
                        [2.0, 5.0],
                        [3.0, 6.0],
                    ]
                ),
                columns=["a", "b"],
            ),
            1: nap.TsdFrame(
                t=[0.2, 0.7],
                d=np.array(
                    [
                        [7.0, 9.0],
                        [8.0, 10.0],
                    ]
                ),
                columns=["a", "b"],
            ),
        }
    )
    plot = viz.PlotTsGroup(data)
    assert plot._entry_kind == "tsd_frame"
    assert plot._continuous
    assert len(plot._entry_graphics[0]) == 2
    assert len(plot._entry_graphics[1]) == 2
    assert all(
        isinstance(graphic, gfx.Line)
        for graphics in plot._entry_graphics.values()
        for graphic in graphics
    )
    for graphic in plot._entry_graphics[1]:
        np.testing.assert_allclose(
            graphic.geometry.positions.data[:, 0],
            data[1].t,
        )
    np.testing.assert_allclose(
        plot._entry_graphics[0][0].geometry.positions.data[:, 1],
        data[0].d[:, 0],
    )
    np.testing.assert_allclose(
        plot._entry_graphics[0][1].geometry.positions.data[:, 1],
        data[0].d[:, 1],
    )
    plot.close()


def test_tsgroup_rejects_mixed_entry_types():
    data = nap.TsGroup(
        {
            0: nap.Ts(t=[0.1, 0.2]),
            1: nap.Tsd(
                t=[0.1, 0.2],
                d=[1.0, 2.0],
            ),
        }
    )
    with pytest.raises(
        TypeError,
        match="All TsGroup entries must have the same type",
    ):
        viz.PlotTsGroup(data)


def test_continuous_tsgroup_rescales_line_thickness():
    data = nap.TsGroup(
        {
            0: nap.Tsd(
                t=[0.0, 0.5],
                d=[1.0, 2.0],
            )
        }
    )
    plot = viz.PlotTsGroup(data)
    line = plot.graphic[0]
    initial_thickness = line.material.thickness
    plot._rescale(
        SimpleNamespace(
            type="key_down",
            key="i",
        )
    )
    assert line.material.thickness > initial_thickness
    plot.close()


def test_continuous_tsgroup_uses_data_ylim():
    data = nap.TsGroup(
        {
            0: nap.Tsd(
                t=[0.0, 0.5],
                d=[-2.0, 3.0],
            ),
            1: nap.Tsd(
                t=[0.2, 0.7],
                d=[4.0, 8.0],
            ),
        }
    )
    plot = viz.PlotTsGroup(data)
    ymin, ymax = plot.controller.get_ylim()
    assert ymin < -2.0
    assert ymax > 8.0
    plot.close()
