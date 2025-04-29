from numbers import Number
from typing import TYPE_CHECKING, Optional

import numpy as np
import pandas as pd
from numpy.typing import NDArray
import pygfx
from matplotlib import colormaps
from matplotlib.colors import Colormap

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


def to_mappable_color(vals):
    unq_vals = np.uniuque(vals)
    try:
        [pygfx.Color(c) for c in unq_vals]
    except ValueError:
        return False
    return True


def check_mappable_object_items(values):
    return all(isinstance(v, Number) or isinstance(v, str) for v in values)

def is_mappable_to_floats(values: NDArray):
    if values.ndim != 1:
        return False
    valid = np.issubdtype(values.dtype, np.str_) or np.issubdtype(values.dtype, np.number)
    if valid:
        return True
    return check_mappable_object_items(values)


def map_numeric_arrays(
        values: NDArray | pd.Series,
        vmin: float = 0.,
        vmax: float = 100.,
        cmap: Colormap = colormaps["rainbow"]):
    """
    Map numerical array to colors.

    Parameters
    ----------
    values:
        A numeric one dimensional array or pandas series.
    vmin:
        Min percentile, between 0 and 100.
    vmax:
        Max percentile, between 0 and 100.
    cmap:
        A colormap.

    Returns
    -------
        A dictionary containing the color maps, keys are metadata entries, values are colors.

    """
    # truncate between percentiles
    values = np.clip(
        values,
        np.nanpercentile(values, vmin, method="closest_observation"),
        np.nanpercentile(values, vmax, method="closest_observation")
    )
    unq_vals = np.unique(values)
    col_val = np.linspace(0, 1, unq_vals.shape[0])
    return {v: pygfx.Color(cmap(c)) for v, c in zip(unq_vals, col_val)}


def map_non_color_string_array(values, cmap=colormaps["rainbow"]):
    """
    Map string/categorical array to colors.

    Parameters
    ----------
    values:
        A numeric one dimensional array or pandas series.
    cmap:
        A colormap.

    Returns
    -------
        A dictionary containing the color maps, keys are metadata entries, values are colors.

    """
    unq_vals, index = np.unique(values, return_index=True)
    # keep the ordering of the metadata array
    unq_vals = unq_vals[np.argsort(index)]
    col_val = np.linspace(0, 1, unq_vals.shape[0])
    return {v: pygfx.Color(cmap(c)) for v, c in zip(unq_vals, col_val)}

def map_color_string_array(values):
    """
    Map arrays of color strings to pygfx colors.

    Parameters
    ----------
    values:
        Array of strings with valid pygfx named colors.

    Returns
    -------
     A dictionary containing the color maps, keys are metadata entries, values are colors.
    """
    return {v: pygfx.Color(v) for v in np.unique(values)}

def map_metadata_to_zero_one(values, vmin=0, vmax=1, map_rescale=False):
    """

    Parameters
    ----------
    values:
        One dimensional array of metadata.
    vmin:
        vmin for the cmap, as a percentile.
    vmax:
        vmax for the cmap, as a percentile.
    map_rescale:
        If True, map to a 0, 1 float, keep as is otherwise.
    """
    if not is_mappable_to_floats(values):
        return
    if np.issubdtype(values.dtype, np.number):
        values = np.clip(values, np.nanpercentile(values, vmin*100), np.nanpercentile(values, vmax*100))
        unq_vals = np.unique(values)

    else:
        unq_vals, index = np.unique(values, return_index=True)
        # keep the ordering of the metadata array
        unq_vals = unq_vals[index]

    col_val = unq_vals if not map_rescale else np.linspace(0, 1, unq_vals.shape[0])
    return {v: c for v, c in zip(unq_vals, col_val)}


## TODO: Implement a filter on what can be mapped to colors.