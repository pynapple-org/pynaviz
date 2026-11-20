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


@pytest.mark.parametrize(
    "data",
    [
        nap.TsGroup(
            {
                0: nap.Ts(t=[0.1, 0.2]),
                1: nap.Tsd(
                    t=[0.3, 0.4],
                    d=[1.0, 2.0],
                ),
            }
        ),
        nap.TsGroup(
            {
                0: nap.TsdTensor(
                    t=[0.1, 0.2],
                    d=np.zeros((2, 2, 2)),
                ),
                1: nap.TsdTensor(
                    t=[0.3, 0.4],
                    d=np.zeros((2, 2, 2)),
                ),
            }
        ),
    ],
)
def test_tsgroup_mixed_or_unsupported_entries_fall_back_to_raster(data):
    with pytest.warns(
        UserWarning,
        match="Plotting the timestamps as a spike raster",
    ):
        plot = viz.PlotTsGroup(data)
    assert plot._entry_kind == "ts"
    assert not plot._continuous
    assert all(isinstance(graphic, gfx.Points) for graphic in plot.graphic.values())
    np.testing.assert_allclose(
        plot.graphic[1].geometry.positions.data[:, 0],
        data[1].t,
    )
    np.testing.assert_allclose(
        plot.graphic[1].geometry.positions.data[:, 1],
        1.0,
    )
    plot.close()


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

def test_tsgroup_x_vs_y_playhead_covers_all_entries():
    # Entry 1 runs well past the end of entry 0.
    group = nap.TsGroup(
        {
            0: nap.TsdFrame(
                t=np.linspace(0.0, 10.0, 11),
                d=np.random.randn(11, 2),
                columns=["x", "y"],
            ),
            1: nap.TsdFrame(
                t=np.linspace(5.0, 20.0, 16),
                d=np.random.randn(16, 2),
                columns=["x", "y"],
            ),
        }
    )
    plot = PlotTsGroup(group)

    try:
        plot.plot_x_vs_y("x", "y")
        plot.controller.set_frame(15.0)
        assert plot.controller._current_time == pytest.approx(15.0)

        plot.controller.set_frame(100.0)
        assert plot.controller._current_time == pytest.approx(20.0)
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




def test_tsgroup_x_vs_y_epoch_jump_moves_playhead(tsgroup_tsdframes):
    plot = PlotTsGroup(tsgroup_tsdframes)

    try:
        plot.add_interval_sets(nap.IntervalSet(start=[0.6, 1.4], end=[0.9, 1.8]))
        plot.plot_x_vs_y("x", "y")
        plot.controller.set_frame(0.0)

        plot.jump_to_next_epoch()
        assert plot.controller._current_time == pytest.approx(0.6)
        plot.jump_to_next_epoch()
        assert plot.controller._current_time == pytest.approx(1.4)
        plot.jump_to_previous_epoch()
        assert plot.controller._current_time == pytest.approx(0.6)
    finally:
        plot.close()


@pytest.mark.parametrize(
    ("keys", "expected"),
    [((), "span"), (("y",), "span_ylock"), (("x",), "span_xlock"), (("x", "y"), "span_xylock")],
)
def test_tsgroup_x_vs_y_restores_axis_lock(tsgroup_tsdframes, keys, expected):
    plot = PlotTsGroup(tsgroup_tsdframes)

    try:
        for key in keys:
            event = SimpleNamespace(type="key_down", key=key, modifiers=())
            plot._toggle_y_lock(event)
            plot._toggle_x_lock(event)
        assert plot._active_controller_key == expected

        plot.plot_x_vs_y("x", "y")
        assert plot._active_controller_key == "get"
        plot._set_time_mode()
        assert plot._active_controller_key == expected

        # One more 'y' press toggles the lock (no extra press needed).
        y_locked = plot._y_locked
        plot._toggle_y_lock(SimpleNamespace(type="key_down", key="y", modifiers=()))
        assert plot._y_locked is (not y_locked)
        assert plot._active_controller_key == plot._get_span_lock_key()
    finally:
        plot.close()


def test_tsgroup_x_vs_y_same_column(tsgroup_tsdframes):
    plot = PlotTsGroup(tsgroup_tsdframes)

    try:
        plot.plot_x_vs_y("x", "x")
        assert plot._display_mode == "x_vs_y"
        for key in tsgroup_tsdframes.keys():
            positions = plot._xy_mode.graphics[key].geometry.positions.data
            np.testing.assert_array_equal(positions[:, 0], positions[:, 1])
    finally:
        plot.close()


def _xvy_group_with_metadata():
    t = np.linspace(0.0, 10.0, 101)
    return nap.TsGroup(
        {
            i: nap.TsdFrame(
                t=t,
                d=np.column_stack([np.cos(t + i), np.sin(t + i)]),
                columns=["x", "y"],
            )
            for i in range(2)
        },
        metadata={"label": [1.0, 2.0]},
    )


def test_tsgroup_x_vs_y_color_by_keeps_custom_fade():
    plot = PlotTsGroup(_xvy_group_with_metadata())

    try:
        plot.plot_x_vs_y(
            "x",
            "y",
            color="white",
            color_mode="assigned",
            range_mode="custom",
            window_before=2.0,
            window_after=2.0,
        )
        plot.controller.set_frame(5.0)
        before = {
            key: graphic.geometry.colors.data.copy()
            for key, graphic in plot._xy_mode.graphics.items()
        }
        assert any((c[:, 3] < 1.0).any() for c in before.values())

        plot.color_mapping_thread.wait_until_done(timeout=5)
        plot.color_by("label", cmap_name="viridis", vmin=0.0, vmax=100.0)

        for key, graphic in plot._xy_mode.graphics.items():
            after = graphic.geometry.colors.data
            np.testing.assert_array_equal(after[:, 3], before[key][:, 3])
            assert not np.allclose(after[:, :3], before[key][:, :3])
    finally:
        plot.close()


@pytest.mark.parametrize(("style", "attr"), [("lines", "thickness"), ("scatter", "markersize")])
def test_tsgroup_x_vs_y_rescale_updates_state(tsgroup_tsdframes, style, attr):
    plot = PlotTsGroup(tsgroup_tsdframes)

    try:
        plot.plot_x_vs_y("x", "y", style=style)
        initial = getattr(plot._xy_mode, attr)

        plot._rescale(SimpleNamespace(type="key_down", key="i", modifiers=()))

        assert getattr(plot._xy_mode, attr) == pytest.approx(initial * 1.2)
        assert plot.get_plot_state()["x_vs_y"][attr] == pytest.approx(initial * 1.2)
    finally:
        plot.close()


@pytest.mark.parametrize(
    "kwargs",
    [{"style": "bad"}, {"range_mode": "bad"}, {"color_mode": "bad"}, {"window_before": -1.0}],
)
def test_tsgroup_x_vs_y_invalid_parameters_leave_plot_untouched(tsgroup_tsdframes, kwargs):
    plot = PlotTsGroup(tsgroup_tsdframes)

    try:
        with pytest.raises(ValueError):
            plot.plot_x_vs_y("x", "y", **kwargs)
        assert plot._display_mode != "x_vs_y"
        assert plot._active_controller_key == "span"
        assert plot.graphic is plot._time_graphic
    finally:
        plot.close()
