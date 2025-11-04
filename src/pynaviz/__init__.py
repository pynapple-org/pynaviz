from importlib.metadata import PackageNotFoundError as _PackageNotFoundError
from importlib.metadata import version as _get_version

try:
    __version__ = _get_version("pynaviz")
except _PackageNotFoundError:
    # package is not installed
    pass

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

# ----------------------------------------------------------------------
# Optional Qt imports
# ----------------------------------------------------------------------
def _load_qt():
    """Lazy loader for the Qt widgets. Called only when the user needs Qt."""
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
        return {
            "IntervalSetWidget": IntervalSetWidget,
            "TsdFrameWidget": TsdFrameWidget,
            "TsdTensorWidget": TsdTensorWidget,
            "TsdWidget": TsdWidget,
            "TsGroupWidget": TsGroupWidget,
            "TsWidget": TsWidget,
            "VideoWidget": VideoWidget,
            "scope": scope,
        }
    except ImportError as e:
        raise ImportError(
            "Qt support is not installed.\n"
            "Install optional Qt dependencies with:\n\n"
            "    pip install pynaviz[qt]\n"
        ) from e


# expose Qt attributes lazily
def __getattr__(name):
    qt_objects = _load_qt()
    if name in qt_objects:
        return qt_objects[name]
    raise AttributeError(f"module 'pynaviz' has no attribute '{name}'")


# Add Qt names to __all__ for discoverability
__all__ += [
    "IntervalSetWidget",
    "TsdFrameWidget",
    "TsdTensorWidget",
    "TsdWidget",
    "TsGroupWidget",
    "TsWidget",
    "VideoWidget",
    "scope",
]


