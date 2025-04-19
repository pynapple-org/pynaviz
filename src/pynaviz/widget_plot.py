"""
Plotting class for each pynapple object using Qt Widget.
Create a unique Qt widget for each class.
Classes hold the specific interactive methods for each pynapple object.
"""

from importlib.metadata import metadata
from typing import List
import bisect
import numpy as np
from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QLabel,
    QPushButton,
    QHBoxLayout,
    QStyle,
    QMenu,
    QDialog,
    QComboBox,
    QListView,
    QSizePolicy,
)
from PyQt6.QtCore import Qt, QSize, QPoint
from pynaviz import PlotTs, PlotTsd, PlotTsdFrame, PlotTsdTensor, PlotTsGroup
import matplotlib.pyplot as plt
from pynaviz.qt_item_models import ChannelListModel
import pynapple as nap
import numpy as np


class DropdownDialog(QDialog):
    def __init__(
        self,
        title: str,
        meta_names: List[str],
        other_combo: List[str],
        initial_idx_other: int,
        parent=None,
    ):
        """
        Parameters
        ----------
        meta_names:
            The metadata column names.
        other_combo:
            Key value pair for the combo box. Keys are used for element to list, values as the content.
        parent:
            The parent widget.
        """
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setWindowModality(Qt.WindowModality.ApplicationModal)
        self.setFixedSize(300, 150)

        # Create dropdown menus
        self.combo_meta = QComboBox()
        self.combo_meta.addItems(list(meta_names))

        self.combo_other = QComboBox()
        self.combo_other.addItems(list(other_combo))
        self.combo_other.setCurrentIndex(initial_idx_other)

        # Layout setup
        layout = QHBoxLayout()
        layout.addWidget(self.combo_meta)
        layout.addWidget(self.combo_other)
        self.setLayout(layout)

        self.combo_meta.currentIndexChanged.connect(self.combo_changed)
        self.combo_other.currentIndexChanged.connect(self.combo_changed)
        self._parent = parent

    def get_current_cmap(self):
        plot = getattr(self._parent, "plot", None)
        if plot:
            return plot.cmap
        return None

    def get_selections(self):
        return self.combo_meta.currentText(), self.combo_other.currentText()

    def combo_changed(self):
        plot = getattr(self._parent, "plot", None)
        if plot:
            meta, cmap = self.get_selections()
            plot.color_by(meta, cmap)


class ChannelList(QDialog):
    def __init__(self, model, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Channel List")
        self.setWindowModality(Qt.WindowModality.ApplicationModal)
        self.setFixedSize(300, 150)

        self.view = QListView(self)

        # Set the model for the list view
        # self.plot = plot
        self.model = model
        self.view.setModel(self.model)

        # Set the selection mode
        self.view.setSelectionMode(self.view.SelectionMode.MultiSelection)

        self.view.clicked.connect(self.handle_click)

        layout = QVBoxLayout()
        layout.addWidget(self.view)
        self.setLayout(layout)

    def handle_click(self, index):
        # Get current state of clicked item
        state = self.model.data(index, Qt.ItemDataRole.CheckStateRole)
        new_state = (
            Qt.CheckState.Unchecked
            if state == Qt.CheckState.Checked
            else Qt.CheckState.Checked
        )

        # Get selected indexes
        selected = self.view.selectionModel().selectedIndexes()

        # Update all selected indexes
        for idx in selected:
            self.model.setData(idx, new_state, Qt.ItemDataRole.CheckStateRole)


class MenuWidget(QWidget):

    def __init__(self, metadata, plot):
        """

        Parameters
        ----------
        metadata: pd.DataFrame or dict
            The list of metadata column names
        plot: _BasePlot
            All the possible base plot
        """
        super().__init__()
        self.metadata = metadata
        self.plot = plot
        self.channel_model = ChannelListModel(self.plot)

        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)  # No border padding
        layout.setSpacing(0)

        # Select button
        icon_size = 18
        self.select_button = self._make_button(
            menu_to_show=self.show_select_menu,
            icon_name="SP_DialogApplyButton",
            icon_size=icon_size,
        )
        layout.addWidget(self.select_button)

        # Action button
        self.action_button = self._make_button(
            menu_to_show=self.show_action_menu,
            icon_name="SP_FileDialogDetailedView",
            icon_size=icon_size,
        )
        layout.addWidget(self.action_button)
        layout.addStretch()

        # Action menu
        self._action_menu()

        # Set layout to the container widget
        self.setLayout(layout)
        self.setFixedHeight(icon_size)

    def _make_button(self, menu_to_show, icon_name, icon_size=20):
        button = QPushButton()
        button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        pixmapi = getattr(QStyle.StandardPixmap, icon_name)
        icon = self.style().standardIcon(pixmapi)
        button.setIcon(icon)
        button.setIconSize(QSize(icon_size, icon_size))
        button.setFixedSize(icon_size + 3, icon_size + 3)
        button.setFlat(True)
        button.clicked.connect(menu_to_show)
        # button.setStyleSheet(
        #     """
        #     QPushButton {
        #         background-color: #232426;
        #         border: none;
        #         border-radius: 4px;
        #         padding: 2px;
        #         margin: 0px;
        #     }
        #
        #     QPushButton:hover {
        #         background-color: #3c3c3c;
        #     }
        #
        #     QPushButton:pressed {
        #         background-color: #1f1f1f;
        #     }
        #
        #     QPushButton:focus {
        #         outline: none;
        #     }
        #     """
        # )
        return button

    def _action_menu(self):
        # First-level menu
        self.action_menu = QMenu()

        # Second-level submenu
        for action_name, action_func in zip(
            ["Color by", "Group by", "Sort by"], ["color_by", "group_by", "sort_by"]
        ):
            action = self.action_menu.addAction(action_name)
            action.setObjectName(action_name)
            action.triggered.connect(self._popup_menu)
            action.setObjectName(action_func)

    def _popup_menu(self):
        action = self.sender()
        popup_name = action.objectName()

        if popup_name == "color_by":
            cmap_list = sorted(plt.colormaps())
            cmap = getattr(self.plot, "cmap", None)
            idx = bisect.bisect_left(cmap_list, cmap) if cmap else 0
            dialog = DropdownDialog(
                "Color by", self.metadata.columns, cmap_list, idx, parent=self
            )
            dialog.setEnabled(True)
            dialog.exec()
        #
        #
        # # Example metadata and other_combo data
        # meta_columns = ["Column1", "Column2", "Column3"]
        # other_combo = {"Option1": "Value1", "Option2": "Value2"}
        #
        #
        # if dialog.exec():
        #     selection1, selection2 = dialog.get_selections()
        #     print(f"Selected: {selection1}, {selection2}")

    def show_action_menu(self):
        # Show menu below the button
        pos = self.action_button.mapToGlobal(QPoint(0, self.action_button.height()))
        self.action_menu.exec(pos)

    def show_select_menu(self):
        # model = TsdFrameColumnListModel(my_tsdframe)
        dialog = ChannelList(self.channel_model, parent=self)
        dialog.exec()




class TsGroupWidget(QWidget):

    def __init__(self, data, index=None, size=(640, 480), set_parent=True):
        super().__init__(None)
        self.resize(*size)

        # The main layout
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)  # Remove default margins
        layout.setSpacing(0)
        self.setLayout(layout)

        # Canvas
        parent = self if set_parent else None
        self.plot = PlotTsGroup(data, index=index, parent=parent)

        # Top level menu container
        self.button_container = MenuWidget(
            metadata=data.metadata, plot=self.plot
        )

        # Add overlay and canvas to layout
        layout.addWidget(self.button_container)
        layout.addWidget(self.plot.canvas)


class TsdWidget(QWidget):

    def __init__(self, data, index=None, set_parent=False):
        super().__init__(None)
        parent = self if set_parent else None
        self.plot = PlotTsd(data, index=index, parent=parent)


class TsdFrameWidget(QWidget):

    def __init__(self, data, index=None, size=(640, 480), set_parent=True):
        super().__init__(None)
        self.resize(*size)

        # The main layout
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)  # Remove default margins
        layout.setSpacing(0)
        self.setLayout(layout)

        # Canvas
        parent = self if set_parent else None
        self.plot = PlotTsdFrame(data, index=index, parent=parent)

        # Top level menu container
        self.button_container = MenuWidget(
            metadata=data.metadata, plot=self.plot
        )

        # Add overlay and canvas to layout
        layout.addWidget(self.button_container)
        layout.addWidget(self.plot.canvas)


class TsdTensorWidget(QWidget):

    def __init__(self, data, index=None, set_parent=True):
        super().__init__(None)
        parent = self if set_parent else None
        self.plot = PlotTsdTensor(data, index=index, parent=parent)


class TsWidget(QWidget):

    def __init__(self, data, index=None, set_parent=False):
        super().__init__(None)
        parent = self if set_parent else None
        self.plot = PlotTs(data, index=index, parent=parent)
