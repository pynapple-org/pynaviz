
import numpy as np
import pygfx as gfx
import pynapple as nap
import pytest

from pynaviz.base_plot import PlotTsGroup
from pynaviz.display_modes import MultiXvsYMode, XvsYMode


@pytest.fixture
def xy_data():
    return nap.TsdFrame(
        t=np.arange(5, dtype=float),
        d=np.column_stack(
            (
                np.arange(5, dtype=float),
                np.arange(5, dtype=float) ** 2,
            )
        ),
        columns=["x", "y"],
    )


def make_mode(xy_data, **kwargs):
    mode = XvsYMode(
        xy_data,
        manager=None,
        default_color="white",
    )
    mode.update_parameters(
        x_col="x",
        y_col="y",
        **kwargs,
    )
    mode.initialize_graphic()
    return mode


@pytest.mark.parametrize(
    ("style", "graphic_type"),
    [
        ("lines", gfx.Line),
        ("scatter", gfx.Points),
    ],
)
def test_x_vs_y_style(xy_data, style, graphic_type):
    mode = make_mode(xy_data, style=style)

    assert isinstance(mode.graphic, graphic_type)
    assert isinstance(mode.time_point, gfx.Points)


@pytest.mark.parametrize(
    ("range_mode", "expected"),
    [
        ("full", (0, 5)),
        ("history", (0, 3)),
        ("future", (2, 3)),
    ],
)
def test_x_vs_y_ranges(xy_data, range_mode, expected):
    mode = make_mode(
        xy_data,
        range_mode=range_mode,
    )

    mode._update_buffer(2)

    assert mode.graphic.geometry.positions.draw_range == expected


def test_x_vs_y_custom_range(xy_data):
    mode = make_mode(
        xy_data,
        range_mode="custom",
        window_before=1.0,
        window_after=1.0,
    )

    mode._update_buffer(2)

    assert mode.graphic.geometry.positions.draw_range == (1, 3)


def test_x_vs_y_custom_alpha_is_maximal_at_current_time(xy_data):
    mode = make_mode(
        xy_data,
        range_mode="custom",
        window_before=2.0,
        window_after=2.0,
    )

    mode._update_buffer(2)
    alpha = mode.graphic.geometry.colors.data[:, 3]

    assert alpha[2] == pytest.approx(1.0)
    assert alpha[0] == pytest.approx(mode.MIN_ALPHA)
    assert alpha[4] == pytest.approx(mode.MIN_ALPHA)
    assert alpha[1] == pytest.approx(alpha[3])


def test_x_vs_y_line_rescale(xy_data):
    mode = make_mode(xy_data, style="lines")
    initial_thickness = mode.graphic.material.thickness
    initial_marker_size = mode.time_point.material.size

    mode.rescale("i")

    assert mode.graphic.material.thickness > initial_thickness
    assert mode.time_point.material.size > initial_marker_size


def test_x_vs_y_scatter_rescale(xy_data):
    mode = make_mode(xy_data, style="scatter")
    initial_size = mode.graphic.material.size

    mode.rescale("i")

    assert mode.graphic.material.size > initial_size
    assert mode.time_point.material.size > initial_size


def test_x_vs_y_state_contains_customization(xy_data):
    mode = make_mode(
        xy_data,
        style="scatter",
        range_mode="custom",
        window_before=2.0,
        window_after=3.0,
    )

    state = mode.get_state()

    assert state["style"] == "scatter"
    assert state["range_mode"] == "custom"
    assert state["window_before"] == 2.0
    assert state["window_after"] == 3.0

def test_multi_x_vs_y_custom_ranges_use_entry_timestamps(
    tsgroup_tsdframes,
):
    plot = PlotTsGroup(tsgroup_tsdframes)

    try:
        mode = MultiXvsYMode(tsgroup_tsdframes, plot._manager)
        mode.update_parameters(
            x_col="x",
            y_col="y",
            range_mode="custom",
            window_before=0.4,
            window_after=0.6,
        )
        mode.initialize_graphics()
        mode.update_time(1.0)

        for key, graphic in mode.graphics.items():
            timestamps = np.asarray(tsgroup_tsdframes[key].t)
            start = int(
                np.searchsorted(
                    timestamps,
                    0.6,
                    side="left",
                )
            )
            end = int(
                np.searchsorted(
                    timestamps,
                    1.6,
                    side="right",
                )
            )

            assert graphic.geometry.positions.draw_range == (
                start,
                end - start,
            )
    finally:
        plot.close()
def test_multi_x_vs_y_history_range(tsgroup_tsdframes):
    plot = PlotTsGroup(tsgroup_tsdframes)

    try:
        mode = MultiXvsYMode(tsgroup_tsdframes, plot._manager)
        mode.update_parameters(
            x_col="x",
            y_col="y",
            range_mode="history",
        )
        mode.initialize_graphics()
        mode.update_time(1.0)

        for key, graphic in mode.graphics.items():
            timestamps = np.asarray(tsgroup_tsdframes[key].t)
            end = int(
                np.searchsorted(
                    timestamps,
                    1.0,
                    side="right",
                )
            )

            assert graphic.geometry.positions.draw_range == (0, end)
    finally:
        plot.close()

def test_multi_x_vs_y_future_range(tsgroup_tsdframes):
    plot = PlotTsGroup(tsgroup_tsdframes)

    try:
        mode = MultiXvsYMode(tsgroup_tsdframes, plot._manager)
        mode.update_parameters(
            x_col="x",
            y_col="y",
            range_mode="future",
        )
        mode.initialize_graphics()
        mode.update_time(1.0)

        for key, graphic in mode.graphics.items():
            timestamps = np.asarray(tsgroup_tsdframes[key].t)
            start = int(
                np.searchsorted(
                    timestamps,
                    1.0,
                    side="left",
                )
            )

            assert graphic.geometry.positions.draw_range == (
                start,
                len(timestamps) - start,
            )
    finally:
        plot.close()

def test_multi_x_vs_y_bounds_include_all_entries(
    tsgroup_tsdframes,
):
    plot = PlotTsGroup(tsgroup_tsdframes)

    try:
        mode = MultiXvsYMode(tsgroup_tsdframes, plot._manager)
        mode.update_parameters("x", "y")
        mode.initialize_graphics()

        xmin, xmax, ymin, ymax = mode.get_bounds()

        all_x = np.concatenate(
            [
                entry.values[:, list(entry.columns).index("x")]
                for entry in tsgroup_tsdframes.values()
            ]
        )
        all_y = np.concatenate(
            [
                entry.values[:, list(entry.columns).index("y")]
                for entry in tsgroup_tsdframes.values()
            ]
        )

        assert xmin == pytest.approx(np.min(all_x))
        assert xmax == pytest.approx(np.max(all_x))
        assert ymin == pytest.approx(np.min(all_y))
        assert ymax == pytest.approx(np.max(all_y))
    finally:
        plot.close()

