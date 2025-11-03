from importlib.metadata import PackageNotFoundError as _PackageNotFoundError
from importlib.metadata import version as _get_version

from .audiovideo import AudioHandler, PlotTsdTensor, PlotVideo, VideoHandler
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
    "PlotVideo",
    "AudioHandler",
    "VideoHandler",
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
    print(f"An error occurred when importing: {e}. Try installing with the [qt] extra. `pip install pynaviz[qt]`")


try:
    __version__ = _get_version("pynaviz")
except _PackageNotFoundError:
    # package is not installed
    pass
