"""
Plotting class for each pynapple object using Qt Widget.
Create a unique Qt widget for each class.
Classes hold the specific interactive methods for each pynapple object.
"""
from importlib.metadata import metadata
from typing import List
from PyQt6.QtGui import QIcon

from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton, QHBoxLayout, QStyle, QMenu, QListWidget, QDialog, QComboBox
from PyQt6.QtCore import Qt, QSize, QPoint
from pynaviz import PlotTs, PlotTsd, PlotTsdFrame, PlotTsdTensor, PlotTsGroup
import matplotlib.pyplot as plt


class DropdownDialog(QDialog):
    def __init__(
            self,
            title: str,
            meta_names: List[str],
            other_combo: List[str],
            parent=None
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

        # Layout setup
        layout = QHBoxLayout()
        layout.addWidget(self.combo_meta)
        layout.addWidget(self.combo_other)
        self.setLayout(layout)

        self.combo_meta.currentIndexChanged.connect(self.combo_changed)
        self.combo_other.currentIndexChanged.connect(self.combo_changed)
        self._parent = parent

    def get_selections(self):
        return self.combo_meta.currentText(), self.combo_other.currentText()

    def combo_changed(self):
        plot = getattr(self._parent, "plot", None)
        if plot:
            meta, cmap = self.get_selections()
            plot.color_by(meta, cmap)




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

        self.setFixedHeight(15)
        # self.setStyleSheet("background-color: rgba(100, 100, 100, 100); color: white; padding: 10px;")
        # self.setStyleSheet("background-color: white; color: white; padding: 10px;")
        self.setStyleSheet("background-color: white; color: white; padding: 10px; margin: 0px;")

        self.button_layout = QHBoxLayout()  # Arrange buttons horizontally
        self.button_layout.setContentsMargins(2, 2, 2, 2)
        self.button_layout.setSpacing(0)

        # Select button
        self.select_button = QPushButton()
        pixmapi = getattr(QStyle.StandardPixmap, "SP_DialogApplyButton")
        icon = self.style().standardIcon(pixmapi)
        self.select_button.setIcon(icon)
        self.select_button.setIconSize(QSize(15,15))
        self.select_button.setFlat(True)
        self.select_button.setFixedSize(15,15)
        self.select_button.clicked.connect(self.show_select_menu)
        self.button_layout.addWidget(self.select_button)

        # Action button
        self.action_button = QPushButton()
        pixmapi = getattr(QStyle.StandardPixmap, "SP_FileDialogDetailedView")
        icon = self.style().standardIcon(pixmapi)
        self.action_button.setIcon(icon)
        # self.action_button.setIconSize(QSize(32,32))
        self.action_button.setFlat(True)
        # self.action_button.setFixedSize(40,40)
        self.action_button.clicked.connect(self.show_action_menu)
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
            dialog = DropdownDialog("Color by", self.metadata.columns, sorted(plt.colormaps()), parent=self)
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
