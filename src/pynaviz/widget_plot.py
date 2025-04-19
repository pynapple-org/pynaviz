"""
Plotting class for each pynapple object using Qt Widget.
Create a unique Qt widget for each class.
Classes hold the specific interactive methods for each pynapple object.
"""

import bisect
from typing import Callable, List

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

from pynaviz import PlotTs, PlotTsd, PlotTsdFrame, PlotTsdTensor, PlotTsGroup
from pynaviz.qt_item_models import ChannelListModel

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
            if meth and val:
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
        meta_names: List[str],
        other_widgets: dict[dict],
        func: Callable,
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
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setWindowModality(Qt.WindowModality.ApplicationModal)

        # Determine grid arrangement
        num_cols = min(len(other_widgets) + 1, 3)  # max 3 per row
        num_rows = (len(other_widgets) + 1) // num_cols
        self.setFixedWidth(180 * num_cols)
        self.setFixedHeight(min(150 * num_rows, 400))
        self._func = func
        self.other_widgets = {}

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
            widget.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)

            wrapper = QWidget()
            wrapper_layout = QVBoxLayout()
            wrapper_layout.setContentsMargins(0, 0, 0, 0)
            wrapper_layout.setSpacing(2)
            wrapper_layout.addWidget(label)
            wrapper_layout.addWidget(widget)
            wrapper.setLayout(wrapper_layout)

            return wrapper

        # Metadata combobox
        self.combo_meta = QComboBox()
        self.combo_meta.addItems(meta_names)
        meta_widget = make_labeled_widget("Metadata", self.combo_meta)
        self.combo_meta.setMinimumWidth(120)
        grid_layout.addWidget(meta_widget, 0, 0)

        # Other widgets
        for i, (label, params) in enumerate(other_widgets.items(), start=1):
            widget = widget_factory(params)
            if hasattr(widget, "currentIndexChanged"):
                widget.currentIndexChanged.connect(self.item_changed)
            if hasattr(widget, "valueChanged"):
                widget.valueChanged.connect(self.item_changed)

            labeled_widget = make_labeled_widget(label, widget)
            row, col = divmod(i, num_cols)
            grid_layout.addWidget(labeled_widget, row, col)

            self.other_widgets[i] = widget

    def get_selections(self):
        out = [self.combo_meta.currentText()]
        for widget in self.other_widgets.values():
            if isinstance(widget, QComboBox):
                out += [widget.currentText()]
            elif isinstance(widget, QDoubleSpinBox):
                out += [widget.value()]
        return out

    def item_changed(self):
        out = self.get_selections()
        self._func(*out)


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


class MenuWidget(QHBoxLayout):

    def __init__(self, metadata, plot, parent):
        """

        Parameters
        ----------
        metadata: pd.DataFrame or dict
            The list of metadata column names
        plot: _BasePlot
            All the possible base plot
        parent: QWidget
            The parent widget
        """
        super().__init__(None)
        self.metadata = metadata
        self.plot = plot
        self.channel_model = ChannelListModel(self.plot)

        self.parent = parent

        self.setSpacing(0)
        self.setContentsMargins(0, 0, 0, 0)

        # Select button
        icon_size = 15
        self.select_button = self._make_button(
            menu_to_show=self.show_select_menu,
            icon_name="SP_DialogApplyButton",
            icon_size=icon_size,
        )
        self.addWidget(self.select_button)

        # Action button
        self.action_button = self._make_button(
            menu_to_show=self.show_action_menu,
            icon_name="SP_FileDialogDetailedView",
            icon_size=icon_size,
        )
        self.addWidget(self.action_button)

        # Action menu
        self._action_menu()

        # Set layout to the container widget
        self.addStretch()

    def _action_menu(self):
        # First-level menu
        self.action_menu = QMenu()

        # Second-level submenu
        for action_name, action_func in zip(
            ["Color by", "Group by", "Sort by"], ["color_by", "group_by", "sort_by"]
        ):
            action = self.action_menu.addAction(action_name)
            action.setObjectName(action_name)
            action.triggered.connect(self.parent.popup_menu)
            action.setObjectName(action_func)

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
        dialog = ChannelList(self.channel_model, parent=self.parent)
        dialog.exec()

    def handle_action(self):
        action = self.sender()
        event = action.data()
        self.plot.update(event)

    def _make_button(self, menu_to_show, icon_name, icon_size=20):
        button = QPushButton()
        pixmapi = getattr(QStyle.StandardPixmap, icon_name)
        icon = self.parent.style().standardIcon(pixmapi)
        button.setIcon(icon)
        button.setIconSize(QSize(icon_size, icon_size))
        button.setFixedSize(icon_size + 8, icon_size + 8)
        button.setFlat(True)
        button.clicked.connect(menu_to_show)
        button.setStyleSheet(
            """
            QPushButton {
                background-color: #3a3a3a;
                border: none;
                padding: 0px;
                margin: 0px;
            }
            QPushButton:hover {
                background-color: #505050;
            }
        """
        )
        return button


class TsGroupWidget(QWidget):

    def __init__(self, data, index=None, size=(640, 480), set_parent=True):
        super().__init__(None)
        self.resize(*size)

        # The main layout
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)  # Remove default margins
        self.setLayout(layout)

        # Canvas
        parent = self if set_parent else None
        self.plot = PlotTsGroup(data, index=index, parent=parent)

        # Top level menu container
        self.button_container = MenuWidget(
            metadata=data.metadata, plot=self.plot, parent=self
        )

        # Add overlay and canvas to layout
        layout.addLayout(self.button_container)
        layout.addWidget(self.plot.canvas)

    def popup_menu(self):
        action = self.sender()
        popup_name = action.objectName()

        if popup_name == "color_by":
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
                self.button_container.metadata.columns,
                dict(Colormap=parameters),
                self.button_container.plot.color_by,
                parent=self,
            )

            dialog.setEnabled(True)
            dialog.exec()


class TsdWidget(QWidget):

    def __init__(self, data, index=None, set_parent=False):
        super().__init__(None)
        parent = self if set_parent else None
        self.plot = PlotTsd(data, index=index, parent=parent)


class TsdFrameWidget(QWidget):

    def __init__(self, data, index=None, size=(640, 480), set_parent=True):
        super().__init__(None)
        self.resize(*size)

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)  # Remove default margins
        self.setLayout(layout)

        # Canvas
        # super().__init__(None)
        parent = self if set_parent else None
        self.plot = PlotTsdFrame(data, index=index, parent=parent)

        # Add overlay and canvas to layout
        layout.addWidget(self.plot.canvas)

        button_container = QWidget()
        button_container.setFixedHeight(50)
        button_container.setStyleSheet(
            "background-color: rgba(0, 0, 0, 150); color: white; padding: 10px;"
        )
        button_layout = QHBoxLayout()  # Arrange buttons horizontally

        # Create three buttons
        button1 = QPushButton("Colrby")
        button2 = QPushButton("Sortby")
        button3 = QPushButton("Button 3")

        # connect button
        button1.clicked.connect(self.button1_clicked)

        # Add buttons to the layout
        button_layout.addWidget(button1)
        button_layout.addWidget(button2)
        button_layout.addWidget(button3)

        button1.clicked.connect(lambda: self.plot.update("color_by"))
        button2.clicked.connect(lambda: self.plot.update("sort_by"))

        # Set layout to the container widget
        button_container.setLayout(button_layout)

        # # Add to the canvas
        button_container.setParent(self.plot.canvas)
        self.button_container = button_container


class TsdTensorWidget(QWidget):

    def __init__(self, data, index=None, set_parent=False):
        super().__init__(None)
        parent = self if set_parent else None
        self.plot = PlotTsdTensor(data, index=index, parent=parent)


class TsWidget(QWidget):

    def __init__(self, data, index=None, set_parent=False):
        super().__init__(None)
        parent = self if set_parent else None
        self.plot = PlotTs(data, index=index, parent=parent)
