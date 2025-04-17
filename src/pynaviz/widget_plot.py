"""
Plotting class for each pynapple object using Qt Widget.
Create a unique Qt widget for each class.
Classes hold the specific interactive methods for each pynapple object.
"""

from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton, QHBoxLayout
from PyQt6.QtCore import Qt
from pynaviz import PlotTs, PlotTsd, PlotTsdFrame, PlotTsdTensor, PlotTsGroup


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

        button_container = QWidget()
        button_container.setFixedHeight(50)
        button_container.setStyleSheet("background-color: rgba(0, 0, 0, 150); color: white; padding: 10px;")
        button_layout = QHBoxLayout()  # Arrange buttons horizontally

        # Create three buttons
        button1 = QPushButton("Colorby")
        button2 = QPushButton("Sortby")
        button3 = QPushButton("Button 3")

        # Add buttons to the layout
        button_layout.addWidget(button1)
        button_layout.addWidget(button2)
        button_layout.addWidget(button3)

        button1.clicked.connect(lambda: self.plot.update("color_by"))
        button2.clicked.connect(lambda: self.plot.update("sort_by"))

        # Set layout to the container widget
        button_container.setLayout(button_layout)

        # Add to the canvas
        button_container.setParent(self.plot.canvas)
        self.button_container = button_container
        self.button_container.hide()
        # layout.addWidget(self.plot.canvas)

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
