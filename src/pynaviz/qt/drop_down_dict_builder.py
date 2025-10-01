import bisect
from collections import OrderedDict

import matplotlib.pyplot as plt
from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import QComboBox, QWidget

from pynaviz.utils import GRADED_COLOR_LIST


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


def get_popup_kwargs(popup_name: str, widget: QWidget, action: QAction | None) -> dict | None:
    plot = getattr(widget, "plot", None)
    if plot is None:
        return
    kwargs = None
    if popup_name == "color_by":
        metadata = getattr(widget, "metadata", None)
        cmap = getattr(plot, "cmap", None)
        cmap = "viridis" if cmap is None else cmap
        # safety in case no metadata is available
        if metadata is None:
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
            ok_cancel_button=True,
            parent=widget,
        )
    elif popup_name == "group_by":
        metadata = getattr(widget, "metadata", None)
        if metadata is None:
            return
        meta = _get_meta_combo(widget)
        kwargs = dict(
            widgets=OrderedDict(Metadata=meta),
            title="Group by",
            func=plot.group_by,
            ok_cancel_button=False,
            parent=widget,
        )
    elif popup_name == "add_interval_set":
        keys = [bytes(k).decode() for k in action.dynamicPropertyNames()]
        cols = {
            "type": QComboBox,
            "name": "add_interval_set",
            "items": keys,
            "values": [action.property(k) for k in keys],
            "current_index": 0}
        kwargs = dict(
            widgets=OrderedDict(IntervalSet=cols),
            title="Add interval_set",
            func=plot.add_interval_sets,
            ok_cancel_button=True,
            parent=widget,
        )
    return kwargs
