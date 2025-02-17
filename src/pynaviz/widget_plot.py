"""
Plotting class for each pynapple object using Qt Widget.
Create a unique Qt widget for each class.
Classes hold the specific interactive methods for each pynapple object.
"""

from PyQt6.QtWidgets import QWidget

from pynaviz import PlotTs, PlotTsd, PlotTsdFrame, PlotTsdTensor, PlotTsGroup


class TsGroupWidget(QWidget):

    def __init__(self, data, parent=None, index=0):
        super().__init__(parent=parent)
        if parent is None:
            parent = self
        self.plot = PlotTsGroup(data, parent=parent, index=index)


class TsdWidget(QWidget):

    def __init__(self, data, parent=None, index=0):
        super().__init__(parent=parent)
        if parent is None:
            parent = self
        self.plot = PlotTsd(data, parent=parent, index=index)


class TsdFrameWidget(QWidget):

    def __init__(self, data, parent=None, index=0):
        super().__init__(parent=parent)
        if parent is None:
            parent = self
        self.plot = PlotTsdFrame(data, parent=parent, index=index)


class TsdTensorWidget(QWidget):

    def __init__(self, data, parent=None, index=0):
        super().__init__(parent=parent)
        if parent is None:
            parent = self
        self.plot = PlotTsdTensor(data, parent=parent, index=index)


class TsWidget(QWidget):

    def __init__(self, data, parent=None, index=0):
        super().__init__(parent=parent)
        if parent is None:
            parent = self
        self.plot = PlotTs(data, parent=parent, index=index)
