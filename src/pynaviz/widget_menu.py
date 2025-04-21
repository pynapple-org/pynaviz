"""
Action and context classes
Create a unique Qt widget menu for each plot.
Classes hold the specific interactive methods for each pynapple object.
"""

import bisect
from collections import OrderedDict
from typing import Callable

import matplotlib.pyplot as plt
from PyQt6.QtCore import QPoint, QSize, Qt
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QListView,
    QMenu,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpacerItem,
    QStyle,
    QVBoxLayout,
    QWidget,
)

from pynaviz.qt_item_models import ChannelListModel

from .utils import GRADED_COLOR_LIST

WIDGET_PARAMS = {
    QComboBox: {
        "name": "setObjectName",
        "items": "addItems",
        "values": "setItemData",  # needs to iterate over items
        "current_index": "setCurrentIndex",
    },
    QDoubleSpinBox: {
        "name": "setObjectNAme",
        "value": "setValue",
    },
}


def widget_factory(parameters):
    widget_type = parameters.pop("type")
    if widget_type == QComboBox:
        widget = QComboBox()
        for arg_name, attr_name in WIDGET_PARAMS[QComboBox].items():
            meth = getattr(widget, attr_name, None)
            val = parameters.get(arg_name, None)
            if (meth is not None) and (val is not None):
                if arg_name == "values":
                    for i, v in enumerate(val):
                        meth(i, v)
                else:
                    meth(val)
    elif widget_type == QDoubleSpinBox:
        widget = QDoubleSpinBox()
        for arg_name, attr_name in WIDGET_PARAMS[QDoubleSpinBox].items():
            meth = getattr(widget, attr_name, None)
            val = parameters.get(arg_name, None)
            if meth and val:
                meth(val)
    else:
        raise ValueError("Unknown widget type.")
    return widget


class DropdownDialog(QDialog):
    def __init__(
        self,
        title: str,
        widgets: OrderedDict[dict],
        func: Callable,
        ok_cancel_button: bool = False,
        parent=None,
    ):
        """
        Parameters
        ----------
        meta_names:
            The metadata column names.
        other_widgets:
            Parameter for constructing other widgets.
        parent:
            The parent widget.
        """
        super().__init__()
        self.setWindowTitle(title)
        self.setWindowModality(Qt.WindowModality.ApplicationModal)

        # Determine grid arrangement
        num_cols = min(len(widgets), 3)  # max 3 per row
        num_rows = (len(widgets)) // num_cols
        self.setFixedWidth(180 * num_cols)
        self.setFixedHeight(min(150 * num_rows, 400))
        self._func = func
        self.widgets = {}

        main_layout = QVBoxLayout(self)

        # Scroll area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        scroll_content = QWidget()
        scroll_content.setSizePolicy(
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum
        )

        grid_layout = QGridLayout()

        outer_layout = QHBoxLayout()
        inner_layout = QVBoxLayout()
        inner_layout.addLayout(grid_layout)

        # Expanding spacer to push everything to the top
        spacer = QSpacerItem(
            0, 0, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding
        )
        h_spacer = QSpacerItem(
            0, 0, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum
        )
        inner_layout.addItem(spacer)
        outer_layout.addLayout(inner_layout)
        outer_layout.addItem(h_spacer)

        # Set inner_layout on the scroll_content widget
        scroll_content.setLayout(outer_layout)

        scroll.setWidget(scroll_content)
        main_layout.addWidget(scroll)

        def make_labeled_widget(label_text, widget):
            label = QLabel(label_text)
            label.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
            widget.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed)

            wrapper = QWidget()
            wrapper_layout = QHBoxLayout()
            wrapper_layout.setContentsMargins(1, 0, 1, 0)
            wrapper_layout.setSpacing(2)
            wrapper_layout.addWidget(label)
            wrapper_layout.addWidget(widget)
            wrapper.setLayout(wrapper_layout)

            return wrapper

        # Other widgets
        for i, (label, params) in enumerate(widgets.items()):
            widget = widget_factory(params)
            if hasattr(widget, "currentIndexChanged"):
                widget.currentIndexChanged.connect(self.item_changed)
            if hasattr(widget, "valueChanged"):
                widget.valueChanged.connect(self.item_changed)

            labeled_widget = make_labeled_widget(label, widget)
            row, col = divmod(i, num_cols)
            grid_layout.addWidget(labeled_widget, row, col)

            self.widgets[i] = widget

        if ok_cancel_button:
            self._update_on_selection = False
            button_layout = QHBoxLayout()
            button_layout.addStretch()  # This is the spacer

            ok_button = QPushButton("OK")
            ok_button.setDefault(True)
            cancel_button = QPushButton("Cancel")

            ok_button.clicked.connect(self.accept)
            cancel_button.clicked.connect(self.reject)

            button_layout.addWidget(cancel_button)
            button_layout.addWidget(ok_button)

            main_layout.addLayout(button_layout)
        else:
            self._update_on_selection = True

        self.adjustSize()

    def accept(self):
        self.update_plot()
        return super().accept()

    def get_selections(self):
        out = []
        for widget in self.widgets.values():
            if isinstance(widget, QComboBox):
                out += [widget.currentText()]
            elif isinstance(widget, QDoubleSpinBox):
                out += [widget.value()]
        return out

    def update_plot(self):
        out = self.get_selections()
        self._func(*out)

    def item_changed(self):
        if self._update_on_selection:
            self.update_plot()


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
            action.triggered.connect(self._popup_menu)
            action.setObjectName(action_func)

        self.action_menu.addSeparator()
        action = self.action_menu.addAction("Plot x vs y")
        action.triggered.connect(self._popup_menu)
        action.setObjectName("x_vs_y")

    def show_action_menu(self):
        # Show menu below the button
        pos = self.action_button.mapToGlobal(QPoint(0, self.action_button.height()))
        self.action_menu.exec(pos)

    def show_select_menu(self):
        # model = TsdFrameColumnListModel(my_tsdframe)
        dialog = ChannelList(self.channel_model, parent=self)
        dialog.exec()

    def _popup_menu(self):
        action = self.sender()
        popup_name = action.objectName()

        if popup_name == "color_by":
            meta = {
                "type": QComboBox,
                "name": "metadata",
                "items": self.metadata.columns,
                "current_index": 0,
            }
            cmap_list = sorted(plt.colormaps())
            cmap = getattr(self.plot, "cmap", None)
            idx = bisect.bisect_left(cmap_list, cmap) if cmap else 0
            parameters = {
                "type": QComboBox,
                "name": "colormap",
                "items": cmap_list,
                "current_index": idx,
            }
            dialog = DropdownDialog(
                "Color by",
                OrderedDict(Metadata=meta, Colormap=parameters),
                self.plot.color_by,
                parent=self,
            )

            dialog.setEnabled(True)
            dialog.exec()

        if popup_name == "x_vs_y":
            cols = {}
            for i, x in enumerate(['x', 'y']):
                cols[x] ={
                "type": QComboBox,
                "name": f"{x} data",
                "items": self.plot.data.columns.astype("str"),
                "current_index": 0 if self.plot.data.shape[1] == 1 else i,
            }
            cols['Color'] = {
                "type": QComboBox,
                "name": "colors",
                "items": GRADED_COLOR_LIST,
                "current_index": 0,
            }
            dialog = DropdownDialog(
                "Plot x vs y",
                cols,
                lambda *args, **kwargs: print("yo"),
                ok_cancel_button=True,
                parent=self,
            )

            dialog.setEnabled(True)
            dialog.exec()
