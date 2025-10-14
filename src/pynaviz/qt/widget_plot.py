"""
Plotting class for each pynapple object using Qt Widget.
Create a unique Qt widget for each class.
"""
import pathlib
import sys
from typing import Optional, Tuple

import pynapple as nap
from numpy._typing import NDArray
from PyQt6.QtCore import QTimer
from PyQt6.QtGui import QAction, QKeySequence
from PyQt6.QtWidgets import QApplication, QVBoxLayout, QWidget

from .. import VideoHandler
from ..audiovideo import PlotTsdTensor, PlotVideo
from ..base_plot import (
    PlotIntervalSet,
    PlotTs,
    PlotTsd,
    PlotTsdFrame,
    PlotTsGroup,
)
from .widget_menu import MenuWidget


def expand_with_time_support(time_support, interval_sets):
    if isinstance(interval_sets, dict):
        interval_sets["Time Support"] = time_support
        return interval_sets
    else:
        return {"Time Support": time_support}


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

        # Add play shortcut (only if we created the app ourselves)
        if self._own_app is not None:
            action = QAction(self)
            action.setShortcut(QKeySequence("Space"))
            action.triggered.connect(self._toggle_play)
            self.addAction(action)
            self._playing = False
            self._play_timer = QTimer()
            self._play_timer.timeout.connect(self._play)
            self._play_timer.start(25)  # 40 FPS

    def show(self) -> None:
        # Show the widget window
        super().show()

        # Start the event loop only if we created our own QApplication
        if self._own_app:
            self._own_app.exec()

    def close(self) -> None:
        # Close the base plot
        if hasattr(self, "plot"):
            self.plot.close()

        # Close the widget window
        super().close()

        # Quit the application if we created it ourselves
        if self._own_app is not None:
            self._own_app.quit()

    def _toggle_play(self):
        if self._own_app is not None:
            self._playing = not self._playing

    def _play(self) -> None:
        if self._playing and hasattr(self, "plot"):
            self.plot.controller.advance(0.025)


class TsGroupWidget(BaseWidget):

    def __init__(self, data, index=None, size=(640, 480), set_parent=True, interval_sets=None):
        super().__init__(size=size)

        # Canvas
        parent = self if set_parent else None
        self.plot = PlotTsGroup(data, index=index, parent=parent)

        # Top level menu container
        interval_sets = expand_with_time_support(data.time_support, interval_sets)
        self.button_container = MenuWidget(metadata=data.metadata, plot=self.plot, interval_sets=interval_sets)

        # Add overlay and canvas to layout
        self.layout.addWidget(self.button_container)
        self.layout.addWidget(self.plot.canvas)


class TsdWidget(BaseWidget):

    def __init__(self, data, index=None, size=(640, 480), set_parent=True, interval_sets=None):
        super().__init__(size=size)

        # Canvas
        parent = self if set_parent else None
        self.plot = PlotTsd(data, index=index, parent=parent)

        # Top level menu container
        interval_sets = expand_with_time_support(data.time_support, interval_sets)
        self.button_container = MenuWidget(metadata=None, plot=self.plot, interval_sets=interval_sets)

        # Add overlay and canvas to layout
        self.layout.addWidget(self.button_container)
        self.layout.addWidget(self.plot.canvas)


class TsdFrameWidget(BaseWidget):

    def __init__(self, data, index=None, size=(640, 480), set_parent=True, interval_sets=None):
        super().__init__(size=size)

        # Canvas
        parent = self if set_parent else None
        self.plot = PlotTsdFrame(data, index=index, parent=parent)

        # Top level menu container
        interval_sets = expand_with_time_support(data.time_support, interval_sets)
        self.button_container = MenuWidget(metadata=data.metadata, plot=self.plot, interval_sets=interval_sets)

        # Add custom menu items
        self.button_container.action_menu.addSeparator()
        xvy_action = self.button_container.action_menu.addAction("Plot x vs y")
        xvy_action.setObjectName("x_vs_y")
        xvy_action.triggered.connect(self.button_container._popup_menu)

        # Add overlay and canvas to layout
        self.layout.addWidget(self.button_container, 0)
        self.layout.addWidget(self.plot.canvas)


class TsWidget(BaseWidget):

    def __init__(self, data, index=None, size=(640, 480), set_parent=True, interval_sets=None):
        super().__init__(size=size)

        # Canvas
        parent = self if set_parent else None
        self.plot = PlotTs(data, index=index, parent=parent)

        # Top level menu container
        interval_sets = expand_with_time_support(data.time_support, interval_sets)
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


class TsdTensorWidget(BaseWidget):

    def __init__(self, data: nap.TsdTensor,
                 index: int =None,
                 size: tuple =(640, 480),
                 set_parent: bool=True,
                 tsdframes: dict = None):
        """
        Widget for visualizing TsdTensor data with optional overlay of TsdFrame data.

        Parameters
        ----------
        data : TsdTensor
            The TsdTensor data to visualize.
        index : int, optional
            Used when included in a ControllerGroup to identify the widget.
        size : tuple, optional
            Initial size of the widget (width, height).
        set_parent : bool, optional
            Whether to set the widget as the parent of the plot (default is True).
        tsdframes : dict of TsdFrame, optional
            TsdFrame object to overlay on the TsdTensor plot.

        """
        super().__init__(size=size)

        # Canvas
        parent = self if set_parent else None
        self.plot = PlotTsdTensor(data, index=index, parent=parent)

        # Top level menu container
        self.button_container = MenuWidget(metadata=None, plot=self.plot, tsdframes=tsdframes)

        # Add overlay and canvas to layout
        self.layout.addWidget(self.button_container)
        self.layout.addWidget(self.plot.canvas)


class VideoWidget(BaseWidget):

    def __init__(self, video: str | pathlib.Path | VideoHandler,
                 t: Optional[NDArray] = None,
                 stream_index: int=0,
                 index=None,
                 size=(640, 480),
                 set_parent=True,
                 tsdframes: dict = None):
        """
        Widget for visualizing video data with optional overlay of TsdFrame data.
        Parameters
        ----------
        video : str or pathlib.Path or VideoHandler
            Path to the video file or a VideoHandler object.
        t : array-like, optional
            Array of timestamps corresponding to video frames.
        stream_index : int, optional
            Index of the video stream to use (default is 0).
        index : int, optional
            Used when included in a ControllerGroup to identify the widget.
        size : tuple, optional
            Initial size of the widget (width, height).
        set_parent : bool, optional
            Whether to set the widget as the parent of the plot (default is True).
        tsdframes : dict of TsdFrame, optional
            TsdFrame object to overlay on the video plot.
        """
        super().__init__(size=size)

        # Canvas
        parent = self if set_parent else None
        self.plot = PlotVideo(video=video, t=t, stream_index=stream_index, index=index, parent=parent)

        # Top level menu container
        self.button_container = MenuWidget(metadata=None, plot=self.plot, tsdframes=tsdframes)

        # Add overlay and canvas to layout
        self.layout.addWidget(self.button_container)
        self.layout.addWidget(self.plot.canvas)




