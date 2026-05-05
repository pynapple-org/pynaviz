"""
Tests for format_profiles and the associated helpers in mainwindow and variable_loader.
"""
from unittest.mock import MagicMock

from pynaviz.format_profiles import FORMAT_PROFILES, FormatProfile
from pynaviz.qt.mainwindow import _apply_format_profile
from pynaviz.qt.variable_loader import _infer_ephys_format

# ---------------------------------------------------------------------------
# FormatProfile dataclass
# ---------------------------------------------------------------------------


def test_format_profile_defaults():
    p = FormatProfile()
    assert p.group_by is None
    assert p.sort_by is None
    assert p.color_by is None
    assert p.sort_mode == "descending"
    assert p.cmap_name is None


def test_format_profile_fields():
    p = FormatProfile(group_by="group", sort_by="anatomy", color_by="group", cmap_name="tab10")
    assert p.group_by == "group"
    assert p.sort_by == "anatomy"
    assert p.color_by == "group"
    assert p.cmap_name == "tab10"


# ---------------------------------------------------------------------------
# FORMAT_PROFILES registry
# ---------------------------------------------------------------------------


def test_neuroscopeio_registered():
    assert "NeuroScopeIO" in FORMAT_PROFILES


def test_neuroscopeio_has_tsdframe_entry():
    profiles = FORMAT_PROFILES["NeuroScopeIO"]
    assert "TsdFrameWidget" in profiles


def test_neuroscopeio_tsdframe_profile_values():
    p = FORMAT_PROFILES["NeuroScopeIO"]["TsdFrameWidget"]
    assert p.group_by == "group"
    assert p.sort_by == "anatomy"
    assert p.color_by == "group"
    assert p.cmap_name is not None


# ---------------------------------------------------------------------------
# _infer_ephys_format
# ---------------------------------------------------------------------------


def test_infer_format_with_neo_reader():
    """When _reader is set, return its class name."""
    class FakeIO:
        pass

    reader = MagicMock()
    reader._reader = FakeIO()
    assert _infer_ephys_format(reader) == "FakeIO"


def test_infer_format_without_reader_is_neuroscope():
    """When _reader is absent (Neurosuite path), return 'NeuroScopeIO'."""
    reader = MagicMock(spec=[])  # no attributes at all
    assert _infer_ephys_format(reader) == "NeuroScopeIO"


# ---------------------------------------------------------------------------
# _apply_format_profile
# ---------------------------------------------------------------------------


def _make_widget(widget_class_name: str, columns: list[str]):
    """Build a minimal widget mock with the given class name and metadata columns."""
    metadata = MagicMock()
    metadata.columns = columns

    button_container = MagicMock()
    button_container.metadata = metadata

    plot = MagicMock()

    widget = MagicMock()
    widget.__class__.__name__ = widget_class_name
    widget.plot = plot
    widget.button_container = button_container
    return widget


def test_apply_profile_calls_group_sort_color():
    """All three actions are called when their columns are present."""
    widget = _make_widget("TsdFrameWidget", ["group", "anatomy"])
    profiles = FORMAT_PROFILES["NeuroScopeIO"]

    _apply_format_profile(widget, profiles)

    widget.plot.group_by.assert_called_once_with("group")
    widget.plot.sort_by.assert_called_once_with("anatomy", mode="descending")
    widget.plot.color_by.assert_called_once_with("group", cmap_name="tab10")


def test_apply_profile_skips_missing_columns():
    """Actions whose metadata column is absent are silently skipped."""
    widget = _make_widget("TsdFrameWidget", ["group"])  # no "anatomy"
    profiles = FORMAT_PROFILES["NeuroScopeIO"]

    _apply_format_profile(widget, profiles)

    widget.plot.group_by.assert_called_once_with("group")
    widget.plot.sort_by.assert_not_called()
    widget.plot.color_by.assert_called_once()


def test_apply_profile_no_entry_for_widget_type():
    """No actions are called when the widget type has no profile entry."""
    widget = _make_widget("TsGroupWidget", ["group", "anatomy"])
    profiles = FORMAT_PROFILES["NeuroScopeIO"]

    _apply_format_profile(widget, profiles)

    widget.plot.group_by.assert_not_called()
    widget.plot.sort_by.assert_not_called()
    widget.plot.color_by.assert_not_called()


def test_apply_profile_no_plot_attribute():
    """Widgets without a plot attribute are silently ignored."""
    profiles = FORMAT_PROFILES["NeuroScopeIO"]
    widget = MagicMock(spec=[])  # no 'plot'
    widget.__class__.__name__ = "TsdFrameWidget"
    # Should not raise
    _apply_format_profile(widget, profiles)


def test_apply_profile_no_metadata():
    """Widgets without metadata (no button_container) are silently ignored."""
    profiles = FORMAT_PROFILES["NeuroScopeIO"]
    widget = MagicMock()
    widget.__class__.__name__ = "TsdFrameWidget"
    widget.plot = MagicMock()
    widget.button_container.metadata = None

    _apply_format_profile(widget, profiles)

    widget.plot.group_by.assert_not_called()


def test_apply_profile_empty_format_profiles():
    """An empty profiles dict is a no-op."""
    widget = _make_widget("TsdFrameWidget", ["group", "anatomy"])
    _apply_format_profile(widget, {})
    widget.plot.group_by.assert_not_called()
