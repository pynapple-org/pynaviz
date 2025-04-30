from .base_plot import PlotTs, PlotTsd, PlotTsdFrame, PlotTsdTensor, PlotTsGroup
from .gui import scope
from .widget_plot import (
    TsdFrameWidget,
    TsdTensorWidget,
    TsdWidget,
    TsGroupWidget,
    TsWidget,
)

__version__ = "0.0.1"
__all__ = [
    "PlotTsd", "PlotTsdFrame", "PlotTsdTensor", "PlotTsGroup", "PlotTs",
    "TsdFrameWidget", "TsdTensorWidget", "TsdWidget", "TsGroupWidget", "TsWidget",
    "scope"
]


# from .neurowidget import NeuroWidget
# from .store_models import *
# from .store_items import *
