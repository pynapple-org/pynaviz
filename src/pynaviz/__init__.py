from .base_plot import PlotTs, PlotTsd, PlotTsdFrame, PlotTsdTensor, PlotTsGroup

try:
    from .gui import scope
except ImportError:
    pass # Check if cleaner way of doing this
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
