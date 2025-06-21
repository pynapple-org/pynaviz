from .base_plot import (
    PlotTs,
    PlotTsd,
    PlotTsdFrame,
    PlotTsdTensor,
    PlotTsGroup,
    PlotVideo,
)

__all__ = ["PlotTsd", "PlotTsdFrame", "PlotTsdTensor", "PlotTsGroup", "PlotTs", "PlotVideo"]

try:
    from .gui import scope
    from .widget_plot import (
        TsdFrameWidget,
        TsdTensorWidget,
        TsdWidget,
        TsGroupWidget,
        TsWidget,
        VideoWidget,
    )
    __all__ += ["TsdFrameWidget", "TsdTensorWidget", "TsdWidget", "TsGroupWidget", "TsWidget", "scope", "VideoWidget"]
except ImportError:
    pass # Check if cleaner way of doing this

__version__ = "0.0.1"


