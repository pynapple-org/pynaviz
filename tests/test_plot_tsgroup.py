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
from pynaviz.base_plot import PlotTsGroup

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from config import TsGroupConfig


def test_plot_tsgroup_init(dummy_tsgroup):
    v = viz.PlotTsGroup(dummy_tsgroup)

    assert isinstance(v.controller, viz.controller.SpanController)
    v.close()



@pytest.mark.parametrize(
    "func, kwargs",
    TsGroupConfig.parameters,
)
def test_plot_tsgroup_action(dummy_tsgroup, func, kwargs):
    v = viz.PlotTsGroup(dummy_tsgroup)
    if func is not None:
        if isinstance(func, (list, tuple)):
            for n, k in zip(func, kwargs, strict=False):
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


def test_tsgroup_x_vs_y_shared_columns(tsgroup_tsdframes):
    plot = PlotTsGroup(tsgroup_tsdframes)

    try:
        assert plot.x_vs_y_columns == ["x", "y"]
        assert plot.supports_x_vs_y
        assert plot.supports_assigned_x_vs_y_colors
    finally:
        plot.close()

def test_tsgroup_x_vs_y_requires_two_shared_columns():
    first = nap.TsdFrame(
        t=np.array([0.0, 1.0]),
        d=np.array([[1.0, 2.0], [3.0, 4.0]]),
        columns=["shared", "first_only"],
    )
    second = nap.TsdFrame(
        t=np.array([0.0, 1.0]),
        d=np.array([[5.0, 6.0], [7.0, 8.0]]),
        columns=["shared", "second_only"],
    )
    data = nap.TsGroup({0: first, 1: second})
    plot = PlotTsGroup(data)

    try:
        assert plot.x_vs_y_columns == ["shared"]
        assert not plot.supports_x_vs_y
        assert "get" not in plot._controllers
    finally:
        plot.close()

def test_tsd_tsgroup_does_not_support_x_vs_y():
    data = nap.TsGroup(
        {
            0: nap.Tsd(t=np.array([0.0, 1.0]), d=np.array([1.0, 2.0])),
            1: nap.Tsd(t=np.array([0.2, 1.2]), d=np.array([3.0, 4.0])),
        }
    )
    plot = PlotTsGroup(data)

    try:
        assert not plot.supports_x_vs_y
        assert plot.x_vs_y_columns == []
        assert "get" not in plot._controllers
    finally:
        plot.close()

def test_tsgroup_x_vs_y_has_one_trajectory_per_entry(
    tsgroup_tsdframes,
):
    plot = PlotTsGroup(tsgroup_tsdframes)

    try:
        plot.plot_x_vs_y("x", "y")

        assert plot._display_mode == "x_vs_y"
        assert plot._active_controller_key == "get"
        assert len(plot._xy_mode.graphics) == len(tsgroup_tsdframes)
        assert set(plot._xy_mode.graphics) == set(
            tsgroup_tsdframes.keys()
        )

        assert not hasattr(plot._xy_mode, "time_point")
        assert plot.ruler_ref_time not in plot.scene.children
    finally:
        plot.close()

def test_tsgroup_x_vs_y_rejects_non_shared_column(
    tsgroup_tsdframes,
):
    plot = PlotTsGroup(tsgroup_tsdframes)

    try:
        with pytest.raises(ValueError, match="must exist in every entry"):
            plot.plot_x_vs_y("x", "first_only")
    finally:
        plot.close()

def test_tsgroup_can_return_from_x_vs_y(
    tsgroup_tsdframes,
):
    plot = PlotTsGroup(tsgroup_tsdframes)

    try:
        plot.plot_x_vs_y("x", "y")
        plot._set_time_mode()

        assert plot._display_mode == "time"
        assert plot._active_controller_key == "span"
        assert plot.graphic is plot._time_graphic
        assert plot._entry_graphics is plot._time_entry_graphics
    finally:
        plot.close()
def test_tsgroup_x_vs_y_retains_assigned_colors(
    tsgroup_tsdframes,
):
    plot = PlotTsGroup(tsgroup_tsdframes)

    try:
        expected = {
            0: gfx.Color("red"),
            1: gfx.Color("blue"),
        }

        for key, color in expected.items():
            for graphic in plot._time_entry_graphics[key]:
                graphic.material.color = color

        plot.plot_x_vs_y(
            "x",
            "y",
            color="white",
            color_mode="assigned",
        )

        for key, graphic in plot._xy_mode.graphics.items():
            actual = graphic.geometry.colors.data[0]
            assert np.allclose(actual, expected[key].rgba)
    finally:
        plot.close()

def test_tsgroup_x_vs_y_single_color(
    tsgroup_tsdframes,
):
    plot = PlotTsGroup(tsgroup_tsdframes)

    try:
        plot.plot_x_vs_y(
            "x",
            "y",
            color="orange",
            color_mode="single",
        )

        expected = gfx.Color("orange")
        expected_rgba = (
            expected.r,
            expected.g,
            expected.b,
            expected.a,
        )

        for graphic in plot._xy_mode.graphics.values():
            assert np.allclose(
                graphic.geometry.colors.data[0],
                expected_rgba,
            )
    finally:
        plot.close()
def test_tsgroup_x_vs_y_respects_visibility(
    tsgroup_tsdframes,
):
    plot = PlotTsGroup(tsgroup_tsdframes)

    try:
        plot.plot_x_vs_y("x", "y")

        plot._manager.visible = np.array([True, False])
        plot._update("toggle_visibility")

        assert plot._xy_mode.graphics[0].visible
        assert not plot._xy_mode.graphics[1].visible

        plot._set_time_mode()

        assert plot._time_graphic[0].visible
        assert not plot._time_graphic[1].visible
    finally:
        plot.close()

def test_tsgroup_x_vs_y_state_roundtrip(
    tsgroup_tsdframes,
):
    source = PlotTsGroup(tsgroup_tsdframes)
    restored = PlotTsGroup(tsgroup_tsdframes)

    try:
        source.plot_x_vs_y(
            "x",
            "y",
            color="orange",
            style="scatter",
            range_mode="custom",
            window_before=0.5,
            window_after=1.5,
            color_mode="assigned",
            markersize=6.0,
        )
        state = source.get_plot_state()

        restored.set_plot_state(state)

        assert restored._display_mode == "x_vs_y"
        assert restored._xy_mode.get_state() == state["x_vs_y"]
        assert not hasattr(restored._xy_mode, "time_point")
    finally:
        source.close()
        restored.close()


