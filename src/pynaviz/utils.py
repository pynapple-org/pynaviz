import inspect
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .base_plot import _BasePlot

GRADED_COLOR_LIST = [
    "midnightblue",
    "navy",
    "blue",
    "royalblue",
    "cornflowerblue",
    "skyblue",
    "lightblue",
    "aquamarine",
    "mediumseagreen",
    "limegreen",
    "yellowgreen",
    "gold",
    "orange",
    "darkorange",
    "tomato",
    "orangered",
    "red",
    "crimson",
    "deeppink",
    "magenta",
]


def get_plot_attribute(
    plot: "_BasePlot", attr_name, filter_graphic: dict[bool] = None
) -> dict | None:
    """Auxiliary safe function for debugging."""
    graphic = getattr(plot, "graphic", None)
    if graphic is None:
        print(f"{plot} doesn't have a graphic.")
        return None
    filter_graphic = filter_graphic or {c: True for c in graphic}
    dict_attr: dict = {
        c: getattr(graphic[c], attr_name)
        for c in graphic
        if hasattr(graphic[c], attr_name) and filter_graphic[c]
    }
    return dict_attr


def trim_kwargs(func, kwargs):
    params = inspect.signature(func).parameters
    return {k: v for k, v in kwargs.items() if k in params}
