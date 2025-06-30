from .base_plot import (
    PlotIntervalSet,
    PlotTs,
    PlotTsd,
    PlotTsdFrame,
    PlotTsdTensor,
    PlotTsGroup,
)

__all__ = [
    "PlotIntervalSet",
    "PlotTsd",
    "PlotTsdFrame",
    "PlotTsdTensor",
    "PlotTsGroup",
    "PlotTs",
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
    )

    __all__ += [
        "IntervalSetWidget",
        "TsdFrameWidget",
        "TsdTensorWidget",
        "TsdWidget",
        "TsGroupWidget",
        "TsWidget",
        "scope",
    ]
except ImportError:
    pass  # Check if cleaner way of doing this

__version__ = "0.0.1"
