from .base_plot import (
    PlotTs,
    PlotTsd,
    PlotTsdFrame,
    PlotTsdTensor,
    PlotTsGroup,
    PlotIntervalSet,
)

__all__ = [
    "PlotTsd",
    "PlotTsdFrame",
    "PlotTsdTensor",
    "PlotTsGroup",
    "PlotTs",
    "PlotIntervalSet",
]

try:
    from .gui import scope
    from .widget_plot import (
        TsdFrameWidget,
        TsdTensorWidget,
        TsdWidget,
        TsGroupWidget,
        TsWidget,
        IntervalSetWidget,
    )

    __all__ += [
        "TsdFrameWidget",
        "TsdTensorWidget",
        "TsdWidget",
        "TsGroupWidget",
        "TsWidget",
        "IntervalSetWidget",
        "scope",
    ]
except ImportError:
    pass  # Check if cleaner way of doing this

__version__ = "0.0.1"
