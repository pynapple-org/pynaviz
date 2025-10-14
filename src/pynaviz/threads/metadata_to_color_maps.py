import concurrent.futures
import threading
from numbers import Number

import numpy as np
import pandas as pd
import pygfx
from matplotlib import colormaps
from matplotlib.colors import Colormap
from numpy.typing import NDArray


def map_numeric_arrays(
    values: NDArray | pd.Series,
    vmin: float = 0.0,
    vmax: float = 100.0,
    cmap: Colormap = colormaps["rainbow"],
):
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
    :
        A dictionary containing the color maps, keys are metadata entries, values are colors.

    """
    mn = np.nanpercentile(values, vmin, method="closest_observation")
    mx = np.nanpercentile(values, vmax, method="closest_observation")

    # Get unique values and their colors in one go
    unq_vals = np.unique(values)
    unq_clipped = np.clip(unq_vals, mn, mx)
    unq_clipped_sorted = np.unique(unq_clipped)  # for colormap range

    # Map clipped values to [0,1] range
    if len(unq_clipped_sorted) == 1:
        normalized = np.full(len(unq_vals), 0.5)
    else:
        # Find position of each clipped value in the sorted unique range
        indices = np.searchsorted(unq_clipped_sorted, unq_clipped)
        normalized = indices / (len(unq_clipped_sorted) - 1)

    # Create the mapping dictionary
    colors = cmap(normalized)
    map_dict = {val: pygfx.Color(color) for val, color in zip(unq_vals, colors)}

    return map_dict


def map_non_color_string_array(values, cmap=colormaps["rainbow"]):
    """
    Map string/categorical array to colors.

    Parameters
    ----------
    values:
        A categorical or string-like array or pandas Series.
    cmap:
        A colormap.

    Returns
    -------
    :
        A dictionary containing the color maps, keys are metadata entries, values are colors.

    """
    unq_vals, index = np.unique(values, return_index=True)
    # keep the ordering of the metadata array
    unq_vals = unq_vals[np.argsort(index)]
    col_val = np.linspace(0, 1, unq_vals.shape[0])
    return {v: pygfx.Color(cmap(c)) for v, c in zip(unq_vals, col_val)}


def map_color_array(values):
    """
    Map arrays of color strings to pygfx colors.

    Parameters
    ----------
    values:
        Array of strings with valid pygfx named colors.

    Returns
    -------
    :
        A dictionary containing the color maps, keys are metadata entries, values are colors.
    """
    return {v: pygfx.Color(v) for v in np.unique(values)}


def is_mappable_color(vals):
    """Check if values are mappable to pygfx colors."""
    unq_vals = np.unique(vals)
    try:
        [pygfx.Color(c) for c in unq_vals]
    except ValueError:
        return False
    return True

def is_hashable(v):
    """Check if a value is hashable."""
    try:
        hash(v)
    except TypeError:
        return False
    return True


class MetadataMappingThread:
    def __init__(self, time_series):
        self.map_lock = threading.Lock()
        self._meta = getattr(time_series, "metadata", None)
        self.color_maps = {}
        # event that stop the loop on metadata columns
        self._stop_event = threading.Event()
        # create the worker
        self.worker = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self.future = None
        self.colormap_ready = threading.Event()
        self.compute_map()

    def update_maps(self, time_series):
        self.request_stop()
        self.wait_until_done()
        with self.map_lock:
            self.color_maps = {}
        self._meta = getattr(time_series, "metadata", None)
        self.compute_map()

    def is_running(self):
        return self.future is not None and self.future.running()

    def compute_map(self):
        if self.is_running():
            self.request_stop()
            self.wait_until_done()
        self._stop_event.clear()
        self.future = self.worker.submit(self._compute_mapping)

    def request_stop(self):
        """Request the current computation to stop."""
        self._stop_event.set()

    def wait_until_done(self, timeout=None):
        """Wait for the current future to complete."""
        if self.future:
            try:
                self.future.result(timeout=timeout)
            except concurrent.futures.TimeoutError:
                pass
            except Exception as e:
                print(f"Error during mapping: {e}")

    def shutdown(self):
        self.request_stop()
        self.wait_until_done()
        self.worker.shutdown(wait=False)

    def _compute_mapping(self):
        if self._meta is None:
            return

        for col in self._meta.columns:
            if self._stop_event.is_set():
                return
            values = self._meta[col]

            # try to see if it is an rgb or other pygfx supported format
            if values.ndim != 1 and is_mappable_color(values):
                with self.map_lock:
                    self.color_maps[col] = map_color_array
            #  string subtype or object array containing strings
            elif np.issubdtype(values.dtype, np.str_) or all(
                isinstance(v, str) for v in values
            ):
                if is_mappable_color(values):
                    with self.map_lock:
                        self.color_maps[col] = map_color_array
                else:
                    with self.map_lock:
                        self.color_maps[col] = map_non_color_string_array
            # array of numbers or object array of numbers
            elif np.issubdtype(values.dtype, np.number) or all(
                isinstance(v, Number) for v in values
            ):
                with self.map_lock:
                    self.color_maps[col] = map_numeric_arrays
            # try any other pygfx supported format
            elif is_mappable_color(values) and all(is_hashable(v) for v in values):
                with self.map_lock:
                    self.color_maps[col] = map_color_array
            # array of objects
            else:
                with self.map_lock:
                    self.color_maps[col] = None

        self.colormap_ready.set()
