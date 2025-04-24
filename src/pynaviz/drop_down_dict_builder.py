import bisect
from collections import OrderedDict
from typing import Callable, Tuple

import matplotlib.pyplot as plt
from PyQt6.QtWidgets import QComboBox, QWidget

from .utils import GRADED_COLOR_LIST


def _get_meta_combo(widget):
    metadata = getattr(widget, "metadata", None)
    if metadata is None:
        return
    meta = {
        "type": QComboBox,
        "name": "metadata",
        "items": metadata.columns,
        "current_index": 0,
    }
    return meta


def get_popup_kwargs(popup_name: str, widget: QWidget) -> dict | None:

    plot = getattr(widget, "plot", None)
    if plot is None:
        return
    kwargs = None
    if popup_name == "color_by":
        metadata = getattr(widget, "metadata", None)
        cmap = getattr(plot, "cmap", None)
        # safety in case no metadata is available
        if metadata is None or cmap is None:
            return

        meta = _get_meta_combo(widget)
        if meta is None:
            return

        cmap_list = sorted(plt.colormaps())
        idx = bisect.bisect_left(cmap_list, cmap)
        parameters = {
            "type": QComboBox,
            "name": "colormap",
            "items": cmap_list,
            "current_index": idx,
        }
        kwargs = dict(
            widgets=OrderedDict(Metadata=meta, Colormap=parameters),
            title="Color by",
            func=plot.color_by,
            ok_cancel_button=False,
            parent=widget,
        )

    elif popup_name == "x_vs_y":
        cols = {}
        for i, x in enumerate(["x", "y"]):
            cols[x] = {
                "type": QComboBox,
                "name": f"{x} data",
                "items": plot.data.columns.astype("str"),
                "current_index": 0 if plot.data.shape[1] == 1 else i,
                "values":plot.data.columns

            }
        cols["Color"] = {
            "type": QComboBox,
            "name": "colors",
            "items": GRADED_COLOR_LIST,
            "current_index": 0,
        }
        kwargs = dict(
            widgets=cols,
            title="Plot x vs y",
            func=plot.plot_x_vs_y,
            ok_cancel_button=True,
            parent=widget,
        )

    elif popup_name == "sort_by":
        metadata = getattr(widget, "metadata", None)
        if metadata is None:
            return
        meta = _get_meta_combo(widget)
        order = {
            "type": QComboBox,
            "name": "order",
            "items": ["ascending", "descending"],
            "current_index": 0,
        }
        kwargs = dict(
            widgets=OrderedDict(Metadata=meta, Order=order),
            title="Sort by",
            func=plot.sort_by,
            ok_cancel_button=False,
            parent=widget,
        )
    return kwargs
