"""Tests for same-level IntervalSet resolution in nested variable trees.

Covers _get_same_level_interval_sets and the key_path plumbing through
add_dock_widget -> _create_widget_for_variable for flat, nested-dict,
EphysReference, and NWBReference variable structures.
"""

import numpy as np
import pynapple as nap
import pytest

import pynaviz.qt.mainwindow as mw
from pynaviz.qt.references import EphysReference, NWBReference

# ---------------------------------------------------------------------------
# Helpers / shared data
# ---------------------------------------------------------------------------

def _make_tsdframe():
    t = np.arange(0, 1, 0.01)
    return nap.TsdFrame(t=t, d=np.random.randn(len(t), 3))


def _make_intervalset():
    return nap.IntervalSet(start=[0.1, 0.5], end=[0.2, 0.6])


class _MockReader:
    """Minimal stand-in for nap.EphysReader / nap.NWBFile."""
    def __init__(self, mapping):
        self._mapping = mapping

    def __getitem__(self, key):
        return self._mapping[key]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def app(qtbot):
    """Ensure a QApplication exists (provided by qtbot)."""
    return


@pytest.fixture
def flat_variables():
    """Top-level dict with TsdFrame and two IntervalSets."""
    return {
        "lfp": _make_tsdframe(),
        "sleep_ep": _make_intervalset(),
        "wake_ep": _make_intervalset(),
    }


@pytest.fixture
def nested_variables():
    """One level of nesting: session -> {lfp, sleep_ep, wake_ep}."""
    return {
        "session": {
            "lfp": _make_tsdframe(),
            "sleep_ep": _make_intervalset(),
            "wake_ep": _make_intervalset(),
        }
    }


@pytest.fixture
def ephys_variables():
    """Nested dict where leaves are EphysReference objects."""
    tsdframe = _make_tsdframe()
    iset = _make_intervalset()
    reader = _MockReader({"lfp": tsdframe, "sleep_ep": iset})
    return {
        "recording": {
            "lfp": EphysReference(ephys_reader=reader, key="lfp"),
            "sleep_ep": EphysReference(ephys_reader=reader, key="sleep_ep"),
        }
    }


@pytest.fixture
def nwb_variables():
    """Nested dict where leaves are NWBReference objects."""
    tsdframe = _make_tsdframe()
    iset = _make_intervalset()
    nwb_file = _MockReader({"lfp": tsdframe, "sleep_ep": iset})
    return {
        "recording": {
            "lfp": NWBReference(nwb_file=nwb_file, key="lfp"),
            "sleep_ep": NWBReference(nwb_file=nwb_file, key="sleep_ep"),
        }
    }


def _make_window(variables, qtbot):
    window = mw.MainWindow(variables)
    qtbot.addWidget(window)
    return window


# ---------------------------------------------------------------------------
# Unit tests: _get_same_level_interval_sets
# ---------------------------------------------------------------------------

class TestGetSameLevelIntervalSets:

    def test_flat_returns_sibling_interval_sets(self, flat_variables, qtbot):
        window = _make_window(flat_variables, qtbot)
        result = window._get_same_level_interval_sets(["lfp"])
        assert set(result.keys()) == {"sleep_ep", "wake_ep"}
        assert all(isinstance(v, nap.IntervalSet) for v in result.values())

    def test_flat_excludes_self(self, flat_variables, qtbot):
        window = _make_window(flat_variables, qtbot)
        result = window._get_same_level_interval_sets(["sleep_ep"])
        assert "sleep_ep" not in result
        assert "wake_ep" in result

    def test_flat_no_sibling_interval_sets_returns_empty(self, qtbot):
        variables = {"lfp": _make_tsdframe(), "spikes": _make_tsdframe()}
        window = _make_window(variables, qtbot)
        result = window._get_same_level_interval_sets(["lfp"])
        assert result == {}

    def test_nested_returns_sibling_interval_sets(self, nested_variables, qtbot):
        window = _make_window(nested_variables, qtbot)
        result = window._get_same_level_interval_sets(["session", "lfp"])
        assert set(result.keys()) == {"session/sleep_ep", "session/wake_ep"}
        assert all(isinstance(v, nap.IntervalSet) for v in result.values())

    def test_nested_label_uses_full_path(self, nested_variables, qtbot):
        window = _make_window(nested_variables, qtbot)
        result = window._get_same_level_interval_sets(["session", "lfp"])
        assert "session/sleep_ep" in result
        assert "session/wake_ep" in result

    def test_nested_excludes_self(self, nested_variables, qtbot):
        window = _make_window(nested_variables, qtbot)
        result = window._get_same_level_interval_sets(["session", "sleep_ep"])
        assert "session/sleep_ep" not in result
        assert "session/wake_ep" in result

    def test_ephys_reference_interval_set_resolved(self, ephys_variables, qtbot):
        window = _make_window(ephys_variables, qtbot)
        result = window._get_same_level_interval_sets(["recording", "lfp"])
        assert "recording/sleep_ep" in result
        assert isinstance(result["recording/sleep_ep"], nap.IntervalSet)

    def test_ephys_reference_non_interval_set_excluded(self, ephys_variables, qtbot):
        window = _make_window(ephys_variables, qtbot)
        result = window._get_same_level_interval_sets(["recording", "lfp"])
        # 'lfp' sibling is TsdFrame, should not appear
        assert "recording/lfp" not in result

    def test_nwb_reference_interval_set_resolved(self, nwb_variables, qtbot):
        window = _make_window(nwb_variables, qtbot)
        result = window._get_same_level_interval_sets(["recording", "lfp"])
        assert "recording/sleep_ep" in result
        assert isinstance(result["recording/sleep_ep"], nap.IntervalSet)

    def test_nwb_reference_non_interval_set_excluded(self, nwb_variables, qtbot):
        window = _make_window(nwb_variables, qtbot)
        result = window._get_same_level_interval_sets(["recording", "lfp"])
        assert "recording/lfp" not in result

    def test_cross_branch_interval_sets_excluded(self, qtbot):
        """IntervalSets under a different top-level key must not appear."""
        variables = {
            "session_a": {
                "lfp": _make_tsdframe(),
                "sleep_ep": _make_intervalset(),
            },
            "session_b": {
                "lfp": _make_tsdframe(),
                "wake_ep": _make_intervalset(),
            },
        }
        window = _make_window(variables, qtbot)
        result = window._get_same_level_interval_sets(["session_a", "lfp"])
        assert "session_a/sleep_ep" in result
        # session_b's interval set must not bleed into session_a
        assert "session_b/wake_ep" not in result


# ---------------------------------------------------------------------------
# Integration tests: add_dock_widget passes interval_sets to the widget
# ---------------------------------------------------------------------------

def _user_interval_sets(widget):
    """Return the interval_sets registered on the widget, minus 'Time Support'."""
    isets = widget.button_container._interval_sets or {}
    return {k: v for k, v in isets.items() if k != "Time Support"}


class TestAddDockWidgetIntervalSets:

    def test_flat_tsdframe_receives_sibling_interval_sets(self, flat_variables, qtbot):
        window = _make_window(flat_variables, qtbot)
        dock = window.add_dock_widget(flat_variables["lfp"], ["lfp"])
        isets = _user_interval_sets(dock.widget())
        assert set(isets.keys()) == {"sleep_ep", "wake_ep"}

    def test_nested_tsdframe_receives_sibling_interval_sets(self, nested_variables, qtbot):
        window = _make_window(nested_variables, qtbot)
        var = nested_variables["session"]["lfp"]
        dock = window.add_dock_widget(var, ["session", "lfp"])
        isets = _user_interval_sets(dock.widget())
        assert set(isets.keys()) == {"session/sleep_ep", "session/wake_ep"}

    def test_nested_tsdframe_no_cross_contamination(self, qtbot):
        variables = {
            "session_a": {"lfp": _make_tsdframe(), "sleep_ep": _make_intervalset()},
            "session_b": {"lfp": _make_tsdframe(), "wake_ep": _make_intervalset()},
        }
        window = _make_window(variables, qtbot)
        dock = window.add_dock_widget(variables["session_a"]["lfp"], ["session_a", "lfp"])
        isets = _user_interval_sets(dock.widget())
        assert "session_a/sleep_ep" in isets
        assert "session_b/wake_ep" not in isets

    def test_ephys_reference_tsdframe_receives_sibling_interval_sets(self, ephys_variables, qtbot):
        window = _make_window(ephys_variables, qtbot)
        ref = ephys_variables["recording"]["lfp"]
        dock = window.add_dock_widget(ref, ["recording", "lfp"])
        isets = _user_interval_sets(dock.widget())
        assert "recording/sleep_ep" in isets

    def test_nwb_reference_tsdframe_receives_sibling_interval_sets(self, nwb_variables, qtbot):
        window = _make_window(nwb_variables, qtbot)
        ref = nwb_variables["recording"]["lfp"]
        dock = window.add_dock_widget(ref, ["recording", "lfp"])
        isets = _user_interval_sets(dock.widget())
        assert "recording/sleep_ep" in isets

    def test_flat_variable_with_no_interval_sets_gets_only_time_support(self, qtbot):
        variables = {"lfp": _make_tsdframe(), "spikes": _make_tsdframe()}
        window = _make_window(variables, qtbot)
        dock = window.add_dock_widget(variables["lfp"], ["lfp"])
        isets = _user_interval_sets(dock.widget())
        assert isets == {}
