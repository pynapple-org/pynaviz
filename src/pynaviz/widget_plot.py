"""
Plotting class for each pynapple object using Qt Widget.
Create a unique Qt widget for each class.
Classes hold the specific interactive methods for each pynapple object.
"""

from PyQt6.QtWidgets import QWidget

from pynaviz import PlotTs, PlotTsd, PlotTsdFrame, PlotTsdTensor, PlotTsGroup


class TsGroupWidget(QWidget):

    def __init__(self, data, index=0):
        super().__init__(None)
        self.plot = PlotTsGroup(data, index=index)


class TsdWidget(QWidget):

    def __init__(self, data, index=0):
        super().__init__(None)
        self.plot = PlotTsd(data, index=index)


class TsdFrameWidget(QWidget):

    def __init__(self, data, index=0):
        super().__init__(None)
        self.plot = PlotTsdFrame(data, index=index)


class TsdTensorWidget(QWidget):

    def __init__(self, data, index=0):
        super().__init__(None)
        self.plot = PlotTsdTensor(data, index=index)


class TsWidget(QWidget):

    def __init__(self, data, index=0):
        super().__init__(None)
        self.plot = PlotTs(data, index=index)
