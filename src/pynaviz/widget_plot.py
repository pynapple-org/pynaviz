"""
Plotting class for each pynapple object using Qt Widget.
Create a unique Qt widget for each class.
Classes hold the specific interactive methods for each pynapple object.
"""
from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton, QHBoxLayout, QStyle, QMenu
from PyQt6.QtCore import Qt, QSize, QPoint
from pynaviz import PlotTs, PlotTsd, PlotTsdFrame, PlotTsdTensor, PlotTsGroup

class MenuWidget(QWidget):

    def __init__(self, metadata, index=None):
        """

        Parameters
        ----------
        metadata: pd.DataFrame or dict
            The list of metadata column names
        index
            The item index
        """
        super().__init__(None)
        self.metadata = metadata
        self.setFixedHeight(50)
        # self.setStyleSheet("background-color: rgba(100, 100, 100, 100); color: white; padding: 10px;")
        self.setStyleSheet("background-color: white; color: white; padding: 10px;")
        button_layout = QHBoxLayout()  # Arrange buttons horizontally

        # Select button
        self.select_button = QPushButton()
        pixmapi = getattr(QStyle.StandardPixmap, "SP_DialogApplyButton")
        icon = self.style().standardIcon(pixmapi)
        self.select_button.setIcon(icon)
        self.select_button.setIconSize(QSize(32,32))
        self.select_button.setFlat(True)
        self.select_button.setFixedSize(40,40)
        self.select_button.clicked.connect(self.show_select_menu)
        button_layout.addWidget(self.select_button)

        # Action button
        self.action_button = QPushButton()
        pixmapi = getattr(QStyle.StandardPixmap, "SP_FileDialogDetailedView")
        icon = self.style().standardIcon(pixmapi)
        self.action_button.setIcon(icon)
        self.action_button.setIconSize(QSize(32,32))
        self.action_button.setFlat(True)
        self.action_button.setFixedSize(40,40)
        self.action_button.clicked.connect(self.show_action_menu)
        button_layout.addWidget(self.action_button)

        # Action menu
        self._action_menu()

        # Set layout to the container widget
        button_layout.addStretch()
        self.setLayout(button_layout)
        self.hide()

    def _action_menu(self):
        # First-level menu
        self.action_menu = QMenu()

        # Second-level submenu
        for action_name in ["Color by", "Group by", "Sort by"]:
            menu = QMenu(action_name, self.action_menu)
            for name in self.metadata.columns:
                menu.addAction(name, lambda: print("yo"))

            self.action_menu.addMenu(menu)

    def show_action_menu(self):
        # Show menu below the button
        pos = self.action_button.mapToGlobal(QPoint(0, self.action_button.height()))
        self.action_menu.exec(pos)

    def show_select_menu(self):
        pass



class TsGroupWidget(QWidget):

    def __init__(self, data, index=None, size = (640, 480), set_parent=True):
        super().__init__(None)
        self.resize(*size)

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)  # Remove default margins
        self.setLayout(layout)

        # Canvas
        # super().__init__(None)
        parent = self if set_parent else None
        self.plot = PlotTsGroup(data, index=index, parent=parent)

        # Add overlay and canvas to layout
        layout.addWidget(self.plot.canvas)

        # Top level menu container
        self.button_container = MenuWidget(metadata=data.metadata)
        self.button_container.setParent(self.plot.canvas)

    def enterEvent(self, event):
        """Show the widget when the mouse enters."""
        self.button_container.show()
        super().enterEvent(event)

    def leaveEvent(self, event):
        """Hide the widget when the mouse leaves."""
        self.button_container.hide()
        super().leaveEvent(event)


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
        self.button_container.hide()

    def button1_clicked(self):
        print("clicked button1")

    def enterEvent(self, event):
        """Show the widget when the mouse enters."""
        self.button_container.show()
        super().enterEvent(event)

    def leaveEvent(self, event):
        """Hide the widget when the mouse leaves."""
        self.button_container.hide()
        super().leaveEvent(event)


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
