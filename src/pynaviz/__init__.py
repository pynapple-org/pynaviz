from .audiovideo import (
    PlotTsdTensor,
    PlotVideo,
)
from .base_plot import (
    PlotIntervalSet,
    PlotTs,
    PlotTsd,
    PlotTsdFrame,
    PlotTsGroup,
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
    from .qt import (
        IntervalSetWidget,
        TsdFrameWidget,
        TsdTensorWidget,
        TsdWidget,
        TsGroupWidget,
        TsWidget,
        VideoWidget,
        scope,
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

except ImportError as e:
    print(f"An error occurred when importing: {e}")


__version__ = "0.0.1"
