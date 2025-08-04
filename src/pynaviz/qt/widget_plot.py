"""
Plotting class for each pynapple object using Qt Widget.
Create a unique Qt widget for each class.
"""
import pathlib
import sys
from typing import Optional, Tuple

from numpy._typing import NDArray
from PyQt6.QtWidgets import QApplication, QVBoxLayout, QWidget

from ..base_plot import (
    PlotIntervalSet,
    PlotTs,
    PlotTsd,
    PlotTsdFrame,
    PlotTsGroup,
)
from ..audiovideo import PlotTsdTensor, PlotVideo
from .widget_menu import MenuWidget


class BaseWidget(QWidget):
    def __init__(self, size: Tuple[int, int] = (800, 600)) -> None:
        # Ensure a QApplication instance exists.
        app: Optional[QApplication] = QApplication.instance()
        if app is None:
            # Create and store a QApplication if it doesn't already exist
            self._own_app: Optional[QApplication] = QApplication(sys.argv)
        else:
            # If one already exists, we don't need to manage it
            self._own_app = None

        # Initialize the QWidget superclass
        super().__init__(None)

        # Set initial window size
        self.resize(*size)

        # Set up a vertical layout with no margins or spacing
        self.layout: QVBoxLayout = QVBoxLayout()
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(0)
        self.setLayout(self.layout)

    def show(self) -> None:
        # Show the widget window
        super().show()

        # Start the event loop only if we created our own QApplication
        if self._own_app:
            self._own_app.exec()


class TsGroupWidget(BaseWidget):

    def __init__(self, data, index=None, size=(640, 480), set_parent=True):
        super().__init__(size=size)

        # Canvas
        parent = self if set_parent else None
        self.plot = PlotTsGroup(data, index=index, parent=parent)

        # Top level menu container
        self.button_container = MenuWidget(metadata=data.metadata, plot=self.plot)

        # Add overlay and canvas to layout
        self.layout.addWidget(self.button_container)
        self.layout.addWidget(self.plot.canvas)


class TsdWidget(BaseWidget):

    def __init__(self, data, index=None, size=(640, 480), set_parent=True):
        super().__init__(size=size)

        # Canvas
        parent = self if set_parent else None
        self.plot = PlotTsd(data, index=index, parent=parent)

        # Top level menu container
        self.button_container = MenuWidget(metadata=None, plot=self.plot)

        # Add overlay and canvas to layout
        self.layout.addWidget(self.button_container)
        self.layout.addWidget(self.plot.canvas)


class TsdFrameWidget(BaseWidget):

    def __init__(self, data, index=None, size=(640, 480), set_parent=True):
        super().__init__(size=size)

        # Canvas
        parent = self if set_parent else None
        self.plot = PlotTsdFrame(data, index=index, parent=parent)

        # Top level menu container
        self.button_container = MenuWidget(metadata=data.metadata, plot=self.plot)

        # Add custom menu items
        self.button_container.action_menu.addSeparator()
        xvy_action = self.button_container.action_menu.addAction("Plot x vs y")
        xvy_action.setObjectName("x_vs_y")
        xvy_action.triggered.connect(self.button_container._popup_menu)

        # Add overlay and canvas to layout
        self.layout.addWidget(self.button_container)
        self.layout.addWidget(self.plot.canvas)


class TsdTensorWidget(BaseWidget):

    def __init__(self, data, index=None, size=(640, 480), set_parent=True):
        super().__init__(size=size)

        # Canvas
        parent = self if set_parent else None
        self.plot = PlotTsdTensor(data, index=index, parent=parent)

        # Top level menu container
        self.button_container = MenuWidget(metadata=None, plot=self.plot)

        # Add overlay and canvas to layout
        self.layout.addWidget(self.button_container)
        self.layout.addWidget(self.plot.canvas)


class TsWidget(BaseWidget):

    def __init__(self, data, index=None, size=(640, 480), set_parent=True):
        super().__init__(size=size)

        # Canvas
        parent = self if set_parent else None
        self.plot = PlotTs(data, index=index, parent=parent)

        # Top level menu container
        self.button_container = MenuWidget(metadata=None, plot=self.plot)

        # Add overlay and canvas to layout
        self.layout.addWidget(self.button_container)
        self.layout.addWidget(self.plot.canvas)

class IntervalSetWidget(BaseWidget):

    def __init__(self, data, index=None, size=(640, 480), set_parent=True):
        super().__init__(size=size)

        # Canvas
        parent = self if set_parent else None
        self.plot = PlotIntervalSet(data, index=index, parent=parent)

        # Top level menu container
        self.button_container = MenuWidget(metadata=data.metadata, plot=self.plot)

         # Add overlay and canvas to layout
        self.layout.addWidget(self.button_container)
        self.layout.addWidget(self.plot.canvas)


class VideoWidget(BaseWidget):

    def __init__(self, video_path: str | pathlib.Path, t: Optional[NDArray] = None, stream_index: int=0, index=None, size=(640, 480), set_parent=True):
        super().__init__(size=size)

        # Canvas
        parent = self if set_parent else None
        self.plot = PlotVideo(video_path=video_path, t=t, stream_index=stream_index, index=index, parent=parent)

        # Top level menu container
        self.button_container = MenuWidget(metadata=None, plot=self.plot)

        # Add overlay and canvas to layout
        self.layout.addWidget(self.button_container)
        self.layout.addWidget(self.plot.canvas)




