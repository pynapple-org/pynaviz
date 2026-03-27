"""Tests for DropdownDialog layout, widget_factory icon/group support, and icon factories."""
from collections import OrderedDict

import pytest
from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QComboBox, QDoubleSpinBox

from pynaviz.qt.drop_down_dict_builder import _CMAP_GROUPS, _cmap_icon, _color_icon
from pynaviz.qt.widget_menu import DropdownDialog, widget_factory


# ---------------------------------------------------------------------------
# widget_factory — flat combobox
# ---------------------------------------------------------------------------

def test_widget_factory_combobox_basic(qtbot):
    widget = widget_factory({
        "type": QComboBox,
        "items": ["A", "B", "C"],
        "current_index": 1,
    })
    assert isinstance(widget, QComboBox)
    assert widget.count() == 3
    assert widget.currentIndex() == 1


def test_widget_factory_spinbox(qtbot):
    widget = widget_factory({
        "type": QDoubleSpinBox,
        "value": 3.14,
    })
    assert isinstance(widget, QDoubleSpinBox)
    assert abs(widget.value() - 3.14) < 1e-6


def test_widget_factory_unknown_type_raises(qtbot):
    with pytest.raises(ValueError):
        widget_factory({"type": object})


# ---------------------------------------------------------------------------
# widget_factory — icon_factory (flat, color swatches)
# ---------------------------------------------------------------------------

def test_widget_factory_icon_factory_sets_icons(qtbot):
    """icon_factory is called for each item and icons are set."""
    icons_generated = []

    def fake_icon(name):
        icons_generated.append(name)
        px = QPixmap(8, 8)
        return px

    widget = widget_factory({
        "type": QComboBox,
        "items": ["red", "blue"],
        "icon_factory": fake_icon,
        "icon_size": QSize(8, 8),
        "clear_text": False,
    })
    assert icons_generated == ["red", "blue"]
    # Icons are set, text still visible
    assert widget.itemText(0) == "red"
    assert widget.itemText(1) == "blue"


def test_widget_factory_clear_text_removes_label_and_sets_tooltip(qtbot):
    """clear_text=True clears item text and stores name as tooltip and UserRole data."""
    widget = widget_factory({
        "type": QComboBox,
        "items": ["navy", "crimson"],
        "icon_factory": _color_icon,
        "icon_size": QSize(32, 16),
        "clear_text": True,
    })
    for i in range(widget.count()):
        assert widget.itemText(i) == ""
        assert widget.itemData(i) is not None           # UserRole — color name
        assert widget.itemData(i, Qt.ItemDataRole.ToolTipRole) == widget.itemData(i)


def test_widget_factory_clear_text_data_survives(qtbot):
    """After clearing text, currentData() still returns the original color name."""
    widget = widget_factory({
        "type": QComboBox,
        "items": ["navy", "crimson"],
        "icon_factory": _color_icon,
        "icon_size": QSize(32, 16),
        "clear_text": True,
    })
    widget.setCurrentIndex(0)
    assert widget.currentData() == "navy"
    widget.setCurrentIndex(1)
    assert widget.currentData() == "crimson"


# ---------------------------------------------------------------------------
# widget_factory — grouped combobox
# ---------------------------------------------------------------------------

def test_widget_factory_groups_builds_headers(qtbot):
    """Group headers are non-selectable; separators exist between groups."""
    groups = OrderedDict([
        ("GroupA", ["a1", "a2"]),
        ("GroupB", ["b1"]),
    ])
    widget = widget_factory({
        "type": QComboBox,
        "groups": groups,
        "current_value": "b1",
    })
    # Total items: header + 2 items + separator + header + 1 item = 6
    assert widget.count() == 6

    # First item is the "GroupA" header — not selectable
    header_item = widget.model().item(0)
    assert not (header_item.flags() & Qt.ItemFlag.ItemIsSelectable)
    assert header_item.font().bold()

    # current_value="b1" lands on the correct index
    assert widget.currentText() == "b1"


def test_widget_factory_groups_current_value_default(qtbot):
    """When current_value is not found, index defaults to 0 (first header)."""
    groups = OrderedDict([("G", ["x", "y"])])
    widget = widget_factory({
        "type": QComboBox,
        "groups": groups,
        "current_value": "nonexistent",
    })
    assert widget.currentIndex() == 0


def test_widget_factory_groups_with_icon_factory(qtbot):
    """Icons and tooltips are set for group items; headers have no icon."""
    called = []

    def fake_icon(name):
        called.append(name)
        return QPixmap(4, 4)

    groups = OrderedDict([("G", ["red", "blue"])])
    widget = widget_factory({
        "type": QComboBox,
        "groups": groups,
        "icon_factory": fake_icon,
        "icon_size": QSize(4, 4),
        "clear_text": True,
        "current_value": "red",
    })
    assert set(called) == {"red", "blue"}
    # Items have data (UserRole) and tooltip set
    # Index 0 = header "G", index 1 = "red", index 2 = "blue"
    assert widget.itemData(1) == "red"
    assert widget.itemData(1, Qt.ItemDataRole.ToolTipRole) == "red"
    assert widget.itemData(2) == "blue"


# ---------------------------------------------------------------------------
# _color_icon
# ---------------------------------------------------------------------------

def test_color_icon_returns_pixmap(qtbot):
    px = _color_icon("navy")
    assert isinstance(px, QPixmap)
    assert not px.isNull()
    assert px.width() == 32 and px.height() == 16


def test_color_icon_fill(qtbot):
    """The pixmap is filled with the requested colour (sample centre pixel)."""
    from PySide6.QtGui import QColor
    px = _color_icon("red")
    img = px.toImage()
    center = img.pixelColor(16, 8)
    assert center.red() > 200
    assert center.blue() < 50


# ---------------------------------------------------------------------------
# _cmap_icon
# ---------------------------------------------------------------------------

def test_cmap_icon_returns_pixmap(qtbot):
    px = _cmap_icon("viridis")
    assert isinstance(px, QPixmap)
    assert not px.isNull()
    assert px.width() == 64 and px.height() == 12


def test_cmap_icon_unknown_cmap_returns_gray(qtbot):
    """Unknown cmap name returns a gray fallback pixmap."""
    from PySide6.QtGui import QColor
    px = _cmap_icon("__not_a_real_cmap__")
    assert not px.isNull()
    img = px.toImage()
    c = img.pixelColor(32, 6)
    assert c == QColor("gray")


def test_cmap_icon_gradient_varies(qtbot):
    """Left and right ends of a gradient pixmap have different colours."""
    px = _cmap_icon("viridis")
    img = px.toImage()
    left = img.pixelColor(2, 6)
    right = img.pixelColor(61, 6)
    assert left != right


# ---------------------------------------------------------------------------
# _CMAP_GROUPS
# ---------------------------------------------------------------------------

def test_cmap_groups_no_duplicates():
    all_names = [n for names in _CMAP_GROUPS.values() for n in names]
    assert len(all_names) == len(set(all_names)), "Duplicate cmap names in _CMAP_GROUPS"


def test_cmap_groups_known_categories():
    assert "Perceptually Uniform" in _CMAP_GROUPS
    assert "Sequential" in _CMAP_GROUPS
    assert "Diverging" in _CMAP_GROUPS
    assert "Qualitative" in _CMAP_GROUPS
    assert "Miscellaneous" in _CMAP_GROUPS


def test_cmap_groups_key_members():
    assert "viridis" in _CMAP_GROUPS["Perceptually Uniform"]
    assert "coolwarm" in _CMAP_GROUPS["Diverging"]
    assert "tab10" in _CMAP_GROUPS["Qualitative"]


# ---------------------------------------------------------------------------
# DropdownDialog — layout and behaviour
# ---------------------------------------------------------------------------

def test_dropdown_dialog_no_fixed_size(qtbot):
    """Dialog must not have a fixed size — both dimensions should be flexible."""
    dialog = DropdownDialog(
        "Test", OrderedDict(Val={"type": QDoubleSpinBox, "value": 1.0}),
        func=lambda v: None,
    )
    qtbot.addWidget(dialog)
    # setFixedSize pins min == max; the new layout must not do this
    assert dialog.minimumWidth() != dialog.maximumWidth() or dialog.maximumWidth() == 16777215
    assert dialog.minimumHeight() != dialog.maximumHeight() or dialog.maximumHeight() == 16777215


def test_dropdown_dialog_get_selections_combobox(qtbot):
    called = {}
    dialog = DropdownDialog(
        "T",
        OrderedDict(Color={
            "type": QComboBox,
            "items": ["navy", "crimson"],
            "icon_factory": _color_icon,
            "icon_size": QSize(32, 16),
            "clear_text": True,
        }),
        func=lambda v: called.update(value=v),
    )
    qtbot.addWidget(dialog)
    dialog.widgets[0].setCurrentIndex(1)
    assert dialog.get_selections() == ["crimson"]


def test_dropdown_dialog_get_selections_spinbox(qtbot):
    out = {}
    dialog = DropdownDialog(
        "T",
        OrderedDict(Val={"type": QDoubleSpinBox, "value": 0.0}),
        func=lambda v: out.update(v=v),
    )
    qtbot.addWidget(dialog)
    dialog.widgets[0].setValue(7.5)
    assert dialog.get_selections() == [7.5]


def test_dropdown_dialog_ok_calls_func(qtbot):
    results = []
    dialog = DropdownDialog(
        "T",
        OrderedDict(Val={"type": QDoubleSpinBox, "value": 2.0}),
        func=lambda v: results.append(v),
        ok_cancel_button=True,
    )
    qtbot.addWidget(dialog)
    dialog.widgets[0].setValue(9.9)
    dialog.accept()
    assert len(results) == 1
    assert abs(results[0] - 9.9) < 1e-6


def test_dropdown_dialog_cancel_does_not_call_func(qtbot):
    results = []
    dialog = DropdownDialog(
        "T",
        OrderedDict(Val={"type": QDoubleSpinBox, "value": 0.0}),
        func=lambda v: results.append(v),
        ok_cancel_button=True,
    )
    qtbot.addWidget(dialog)
    dialog.reject()
    assert results == []


def test_dropdown_dialog_immediate_update_on_change(qtbot):
    """Without ok_cancel_button, item_changed fires on activated — including same-index re-selection."""
    results = []
    dialog = DropdownDialog(
        "T",
        OrderedDict(Color={
            "type": QComboBox,
            "items": ["a", "b"],
        }),
        func=lambda v: results.append(v),
        ok_cancel_button=False,
    )
    qtbot.addWidget(dialog)
    # activated fires even when the user picks the already-selected item
    dialog.widgets[0].activated.emit(0)
    assert len(results) == 1


def test_dropdown_dialog_multiple_widgets(qtbot):
    """Multiple widgets are laid out and get_selections returns all values."""
    out = {}
    dialog = DropdownDialog(
        "T",
        OrderedDict(
            A={"type": QComboBox, "items": ["x", "y"], "current_index": 1},
            B={"type": QDoubleSpinBox, "value": 3.0},
        ),
        func=lambda a, b: out.update(a=a, b=b),
        ok_cancel_button=True,
    )
    qtbot.addWidget(dialog)
    assert dialog.get_selections() == ["y", 3.0]
    dialog.accept()
    assert out == {"a": "y", "b": 3.0}
