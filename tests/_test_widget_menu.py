import sys
from collections import OrderedDict
from unittest.mock import MagicMock

import pytest
from PySide6.QtWidgets import QApplication, QComboBox, QDoubleSpinBox

from pynaviz.qt.widget_list_selection import ChannelListModel
from pynaviz.qt.widget_menu import ChannelList, DropdownDialog, MenuWidget, widget_factory


# Initialize QApplication once per test session
@pytest.fixture(scope="session")
def app():
    return QApplication(sys.argv)

# Dummy plot class with minimal attributes
class DummyPlot:
    def __init__(self):
        self.material = [MagicMock(opacity=1.0) for _ in range(3)]
        self.canvas = MagicMock()
        self.animate = MagicMock()

@pytest.fixture
def dummy_plot():
    return DummyPlot()

@pytest.fixture
def dummy_model(dummy_plot):
    return ChannelListModel(dummy_plot)

def test_widget_factory_combobox():
    params = {
        "type": QComboBox,
        "name": "combo",
        "items": ["A", "B"],
        "values": [1, 2],
        "current_index": 0
    }
    widget = widget_factory(params)
    assert isinstance(widget, QComboBox)
    assert widget.count() == 2
    assert widget.currentIndex() == 0

def test_widget_factory_spinbox():
    params = {
        "type": QDoubleSpinBox,
        "name": "spin",
        "value": 5.0
    }
    widget = widget_factory(params)
    assert isinstance(widget, QDoubleSpinBox)
    assert widget.value() == 5.0

def test_dropdown_dialog_updates(app):
    called = {}
    def dummy_func(val):
        called['value'] = val

    widgets = OrderedDict({
        "Test": {
            "type": QDoubleSpinBox,
            "value": 3.14
        }
    })
    dialog = DropdownDialog("Title", widgets, dummy_func, ok_cancel_button=True)
    dialog.widgets[0].setValue(2.71)
    dialog.accept()  # Simulate pressing OK
    assert 'value' in called
    assert abs(called['value'] - 2.71) < 1e-3

def test_channel_list_shows_model(app, dummy_model):
    dialog = ChannelList(dummy_model)
    assert dialog.view.model() is dummy_model

def test_menu_widget_initializes(dummy_plot):
    menu = MenuWidget(metadata={}, plot=dummy_plot)
    assert menu.channel_model.plot is dummy_plot
    assert menu.layout() is not None
