"""
Plotting class for each pynapple object using Qt Widget.
Create a unique Qt widget for each class.
Classes hold the specific interactive methods for each pynapple object.
"""
from importlib.metadata import metadata
from typing import List, Callable
import bisect
import numpy as np
from PyQt6.QtGui import QIcon

from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton, QHBoxLayout, QStyle, QMenu, QListWidget, QDialog, \
    QComboBox, QScrollArea, QGridLayout, QDoubleSpinBox
from PyQt6.QtCore import Qt, QSize, QPoint
from pynaviz import PlotTs, PlotTsd, PlotTsdFrame, PlotTsdTensor, PlotTsGroup
import matplotlib.pyplot as plt

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
    }
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
            parent=None
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
        self.setFixedSize(300, 150)
        self._func = func

        main_layout = QVBoxLayout()
        # Scroll area setup
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_content = QWidget()
        grid_layout = QGridLayout(scroll_content)

        # ComboBoxes from metadata columns
        num_cols = min(len(other_widgets) + 1, 3) # max per row
        container = QVBoxLayout()
        container.addWidget(QLabel("Metadata"))
        self.combo_meta = QComboBox()
        self.combo_meta.addItems(list(meta_names))
        container.addWidget(self.combo_meta)

        self.other_widgets = {}
        grid_layout.addLayout(container, 0, 0)
        num_widget = 1
        for key, params in other_widgets.items():
            widget = widget_factory(params)
            container = QVBoxLayout()
            container.addWidget(QLabel(key))
            container.addWidget(widget)
            if hasattr(widget, "currentIndexChanged"):
                widget.currentIndexChanged.connect(self.item_changed)
            if hasattr(widget, "valueChanged"):
                widget.valueChanged.connect(self.item_changed)
            row, col = num_widget // num_cols, num_widget % num_cols
            grid_layout.addLayout(container, row, col)
            self.other_widgets[num_widget] = widget
            num_widget += 1

        scroll.setWidget(scroll_content)
        main_layout.addWidget(scroll)

        self.setLayout(main_layout)
        # self.combo_other = QComboBox()
        # self.combo_other.addItems(list(other_combo))
        # self.combo_other.setCurrentIndex(initial_idx_other)

        # # Layout setup
        # layout = QHBoxLayout()
        # layout.addWidget(self.combo_meta)
        # layout.addWidget(self.combo_other)
        # self.setLayout(layout)
        #
        # self.combo_meta.currentIndexChanged.connect(self.item_changed)
        # self.combo_other.currentIndexChanged.connect(self.item_changed)
        # self._parent = parent

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




class MenuWidget(QWidget):

    def __init__(self, metadata, plot, index=None):
        """

        Parameters
        ----------
        metadata: pd.DataFrame or dict
            The list of metadata column names
        plot: _BasePlot
            All the possible base plot
        index
            The item index
        """
        super().__init__(None)
        self.metadata = metadata
        self.plot = plot
        self.setFixedHeight(20)
        self.setContentsMargins(0, 0, 0, 0)

        self.button_layout = QHBoxLayout()  # Arrange buttons horizontally
        self.button_layout.setContentsMargins(0, 0, 0, 0)
        self.button_layout.setSpacing(0)

        # Select button
        icon_size = 20
        self.select_button = self._make_button(
            menu_to_show=self.show_select_menu,
            icon_name="SP_DialogApplyButton",
            icon_size=icon_size
        )
        self.button_layout.addWidget(self.select_button)

        # Action button
        self.action_button = self._make_button(
            menu_to_show=self.show_action_menu,
            icon_name="SP_FileDialogDetailedView",
            icon_size=icon_size
        )
        self.button_layout.addWidget(self.action_button)

        # Action menu
        self._action_menu()

        # Set layout to the container widget
        self.button_layout.addStretch()
        self.setLayout(self.button_layout)


    def _action_menu(self):
        # First-level menu
        self.action_menu = QMenu(self)

        # Second-level submenu
        for action_name, action_func in zip(
                ["Color by", "Group by", "Sort by"],
                ["color_by", "group_by", "sort_by"]
        ):
            action = self.action_menu.addAction(action_name)
            action.setObjectName(action_name)
            action.triggered.connect(self.popup_menu)
            action.setObjectName(action_func)

            # self.action_menu.addAction(action_name)
            # menu = QMenu(action_name, self.action_menu)
            # for name in self.metadata.columns:
            #     action = menu.addAction(name)
            #     action.setObjectName(action_func+"|"+name)
            #     action.triggered.connect(self.handle_action)
            #
            # self.action_menu.addMenu(menu)

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
            dialog = DropdownDialog("Color by", self.metadata.columns, dict(Colormap=parameters), self.plot.color_by, parent=self)
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
        pass

    def handle_action(self):
        action = self.sender()
        event = action.data()
        self.plot.update(event)

    def _make_button(self, menu_to_show, icon_name, icon_size=20):
        button = QPushButton()
        pixmapi = getattr(QStyle.StandardPixmap, icon_name)
        icon = self.style().standardIcon(pixmapi)
        button.setIcon(icon)
        button.setIconSize(QSize(icon_size,icon_size))
        button.setFixedSize(icon_size + 8, icon_size + 8)
        button.setFlat(True)
        button.clicked.connect(menu_to_show)
        button.setStyleSheet("""
            QPushButton {
                background-color: #3a3a3a;
                border: none;
                padding: 0px;
                margin: 0px;
            }
            QPushButton:hover {
                background-color: #505050;
            }
        """)
        return button


class TsGroupWidget(QWidget):

    def __init__(self, data, index=None, size = (640, 480), set_parent=True):
        super().__init__(None)
        self.resize(*size)

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)  # Remove default margins
        self.setLayout(layout)

        # Canvas
        parent = self if set_parent else None
        self.plot = PlotTsGroup(data, index=index, parent=parent)

        # Top level menu container
        self.button_container = MenuWidget(metadata=data.metadata, plot=self.plot)

        # Add overlay and canvas to layout
        layout.addWidget(self.button_container)
        layout.addWidget(self.plot.canvas)

        # # Top level menu container
        # self.button_container = MenuWidget(metadata=data.metadata, plot=self.plot)
        # self.button_container.setParent(self.plot.canvas)

class TsdWidget(QWidget):

    def __init__(self, data, index=None, set_parent=False):
        super().__init__(None)
        parent = self if set_parent else None
        self.plot = PlotTsd(data, index=index, parent=parent)


class TsdFrameWidget(QWidget):

    def __init__(self, data, index=None, size = (640, 480), set_parent=True):
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
        button_container.setStyleSheet("background-color: rgba(0, 0, 0, 150); color: white; padding: 10px;")
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

        button1.clicked.connect(lambda : self.plot.update("color_by"))
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
