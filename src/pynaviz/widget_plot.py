"""
Plotting class for each pynapple object using Qt Widget.
Create a unique Qt widget for each class.
Classes hold the specific interactive methods for each pynapple object.
"""

from PyQt6.QtWidgets import QWidget

from pynaviz import PlotTs, PlotTsd, PlotTsdFrame, PlotTsdTensor, PlotTsGroup


class TsGroupWidget(QWidget):

    def __init__(self, data, index=0, set_parent=False):
        super().__init__(None)
        parent = self if set_parent else None
        self.plot = PlotTsGroup(data, index=index, parent=parent)


class TsdWidget(QWidget):

    def __init__(self, data, index=0, set_parent=False):
        super().__init__(None)
        parent = self if set_parent else None
        self.plot = PlotTsd(data, index=index, parent=parent)


class TsdFrameWidget(QWidget):

    def __init__(self, data, index=0, set_parent=False):
        super().__init__(None)
        parent = self if set_parent else None
        self.plot = PlotTsdFrame(data, index=index, parent=parent)


class TsdTensorWidget(QWidget):

    def __init__(self, data, index=0, set_parent=False):
        super().__init__(None)
        parent = self if set_parent else None
        self.plot = PlotTsdTensor(data, index=index, parent=parent)


class TsWidget(QWidget):

    def __init__(self, data, index=0, set_parent=False):
        super().__init__(None)
        parent = self if set_parent else None
        self.plot = PlotTs(data, index=index, parent=parent)
