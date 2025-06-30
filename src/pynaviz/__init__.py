from .base_plot import (
    PlotIntervalSet,
    PlotTs,
    PlotTsd,
    PlotTsdFrame,
    PlotTsdTensor,
    PlotTsGroup,
    PlotVideo,
)

__all__ = [
    "PlotIntervalSet",
    "PlotTsd",
    "PlotTsdFrame",
    "PlotTsdTensor",
    "PlotTsGroup",
    "PlotTs",
    "PlotVideo"
]

try:
    from .gui import scope
    from .widget_plot import (
        IntervalSetWidget,
        TsdFrameWidget,
        TsdTensorWidget,
        TsdWidget,
        TsGroupWidget,
        TsWidget,
        VideoWidget,
    )

    __all__ += [
        "IntervalSetWidget",
        "TsdFrameWidget",
        "TsdTensorWidget",
        "TsdWidget",
        "TsGroupWidget",
        "TsWidget",
        "scope",
        "VideoWidget"
    ]
    
except ImportError:
    pass  # Check if cleaner way of doing this

__version__ = "0.0.1"
