"""Tests for the time bar: _TimeSlider widget and MainWindow scrubber methods."""

import pytest
from PySide6.QtCore import QPoint, Qt
from PySide6.QtGui import QWheelEvent

import pynaviz as viz
from pynaviz.qt.mainwindow import _TIMEBAR_RESOLUTION, _TimeSlider

# ---------------------------------------------------------------------------
# _TimeSlider
# ---------------------------------------------------------------------------


@pytest.fixture
def slider(qtbot):
    s = _TimeSlider(Qt.Orientation.Horizontal)
    s.setRange(0, _TIMEBAR_RESOLUTION)
    qtbot.addWidget(s)
    s.show()
    return s


def test_slider_initial_view_window(slider):
    assert slider._view_start_frac == 0.0
    assert slider._view_end_frac == 0.0


def test_set_view_window_stores_fractions(slider):
    slider.set_view_window(0.2, 0.8)
    assert slider._view_start_frac == pytest.approx(0.2)
    assert slider._view_end_frac == pytest.approx(0.8)


def test_set_view_window_clamps_below_zero(slider):
    slider.set_view_window(-0.5, 0.3)
    assert slider._view_start_frac == 0.0
    assert slider._view_end_frac == pytest.approx(0.3)


def test_set_view_window_clamps_above_one(slider):
    slider.set_view_window(0.7, 1.5)
    assert slider._view_start_frac == pytest.approx(0.7)
    assert slider._view_end_frac == 1.0


def test_set_view_window_clamps_both_ends(slider):
    slider.set_view_window(-1.0, 2.0)
    assert slider._view_start_frac == 0.0
    assert slider._view_end_frac == 1.0


def test_wheel_event_increases_value(slider, qtbot):
    slider.setValue(500_000)
    event = QWheelEvent(
        slider.rect().center().toPointF(),
        slider.mapToGlobal(slider.rect().center()).toPointF(),
        QPoint(0, 0),
        QPoint(0, 120),  # one positive tick
        Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.NoModifier,
        Qt.ScrollPhase.NoScrollPhase,
        False,
    )
    initial = slider.value()
    slider.wheelEvent(event)
    assert slider.value() > initial


def test_wheel_event_decreases_value(slider, qtbot):
    slider.setValue(500_000)
    event = QWheelEvent(
        slider.rect().center().toPointF(),
        slider.mapToGlobal(slider.rect().center()).toPointF(),
        QPoint(0, 0),
        QPoint(0, -120),  # one negative tick
        Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.NoModifier,
        Qt.ScrollPhase.NoScrollPhase,
        False,
    )
    initial = slider.value()
    slider.wheelEvent(event)
    assert slider.value() < initial


def test_wheel_event_step_size(slider):
    """One scroll tick should move the value by 1 % of the full range."""
    slider.setValue(500_000)
    event = QWheelEvent(
        slider.rect().center().toPointF(),
        slider.mapToGlobal(slider.rect().center()).toPointF(),
        QPoint(0, 0),
        QPoint(0, 120),
        Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.NoModifier,
        Qt.ScrollPhase.NoScrollPhase,
        False,
    )
    slider.wheelEvent(event)
    expected_step = int(1 * _TIMEBAR_RESOLUTION / 100)
    assert slider.value() == 500_000 + expected_step


# ---------------------------------------------------------------------------
# MainWindow time bar integration
# ---------------------------------------------------------------------------


@pytest.fixture
def main_window(nap_var, qtbot):
    mw = viz.qt.mainwindow.MainWindow(nap_var)
    qtbot.addWidget(mw)
    for name, var in nap_var.items():
        mw.add_dock_widget(var, [name])
    return mw


def test_timebar_range(main_window):
    assert main_window.time_slider.minimum() == 0
    assert main_window.time_slider.maximum() == _TIMEBAR_RESOLUTION


def test_update_timebar_sets_slider_position(main_window):
    """Slider moves to the correct fractional position."""
    min_t, max_t = 0.0, 10.0
    current_t = 5.0  # midpoint → slider should be at 50 %
    main_window._update_timebar(min_t, max_t, current_t)
    expected = int(0.5 * _TIMEBAR_RESOLUTION)
    assert main_window.time_slider.value() == expected


def test_update_timebar_at_start(main_window):
    main_window._update_timebar(0.0, 10.0, 0.0)
    assert main_window.time_slider.value() == 0


def test_update_timebar_at_end(main_window):
    main_window._update_timebar(0.0, 10.0, 10.0)
    assert main_window.time_slider.value() == _TIMEBAR_RESOLUTION


def test_update_timebar_clamps_out_of_range(main_window):
    """Values outside [min, max] should be clamped to the slider bounds."""
    main_window._update_timebar(0.0, 10.0, -5.0)
    assert main_window.time_slider.value() == 0

    main_window._update_timebar(0.0, 10.0, 20.0)
    assert main_window.time_slider.value() == _TIMEBAR_RESOLUTION


def test_update_timebar_noop_when_range_invalid(main_window):
    """_update_timebar with max <= min should not change the slider value."""
    main_window.time_slider.setValue(12345)
    main_window._update_timebar(5.0, 5.0, 5.0)
    assert main_window.time_slider.value() == 12345


def test_on_timebar_scrub_sets_time(main_window):
    """Scrubbing to mid-slider should seek to the midpoint of the data range."""
    min_t, max_t = main_window._get_max_min_time()
    mid_value = _TIMEBAR_RESOLUTION // 2
    main_window._on_timebar_scrub(mid_value)
    expected_t = min_t + 0.5 * (max_t - min_t)
    assert main_window.ctrl_group.current_time == pytest.approx(expected_t, abs=1e-3)


def test_unit_change_updates_spinbox(main_window):
    """Switching from seconds to milliseconds should multiply the displayed value."""
    main_window.ctrl_group.set_interval(1.0, None)
    # Switch to ms unit first, then update the label (which refreshes min/max + value).
    main_window.time_unit_combo.setCurrentIndex(1)  # ms → multiplier 1e3
    main_window._update_time_label(1.0)
    displayed = main_window.time_spin_box.value()
    assert displayed == pytest.approx(1000.0, rel=1e-3)


def test_spinbox_change_seeks_time(main_window):
    """Typing a value in the spinbox (in seconds) should update the controller time."""
    main_window.time_unit_combo.setCurrentIndex(2)  # seconds
    main_window._on_spinbox_changed(3.0)
    assert main_window.ctrl_group.current_time == pytest.approx(3.0, abs=1e-6)
