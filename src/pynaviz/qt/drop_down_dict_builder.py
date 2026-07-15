from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
from PySide6.QtCore import QSize
from PySide6.QtGui import QAction, QColor, QImage, QPixmap
from PySide6.QtWidgets import QComboBox, QWidget

from pynaviz.utils import GRADED_COLOR_LIST

_CMAP_GROUPS = OrderedDict([
    ("Perceptually Uniform", [
        "viridis", "plasma", "inferno", "magma", "cividis",
    ]),
    ("Sequential", [
        "Greys", "Purples", "Blues", "Greens", "Oranges", "Reds",
        "YlOrBr", "YlOrRd", "OrRd", "PuRd", "RdPu", "BuPu",
        "GnBu", "PuBu", "YlGnBu", "PuBuGn", "BuGn", "YlGn",
        "binary", "gist_yarg", "gist_gray", "gray", "bone", "pink",
        "spring", "summer", "autumn", "winter", "cool", "Wistia",
        "hot", "afmhot", "gist_heat", "copper",
    ]),
    ("Diverging", [
        "PiYG", "PRGn", "BrBG", "PuOr", "RdGy", "RdBu",
        "RdYlBu", "RdYlGn", "Spectral", "coolwarm", "bwr", "seismic",
        "berlin", "managua", "vanimo",
    ]),
    ("Cyclic", ["twilight", "twilight_shifted", "hsv"]),
    ("Qualitative", [
        "Pastel1", "Pastel2", "Paired", "Accent", "Dark2",
        "Set1", "Set2", "Set3", "tab10", "tab20", "tab20b", "tab20c",
    ]),
    ("Miscellaneous", [
        "flag", "prism", "ocean", "gist_earth", "terrain", "gist_stern",
        "gnuplot", "gnuplot2", "CMRmap", "cubehelix", "brg", "gist_rainbow",
        "rainbow", "jet", "turbo", "nipy_spectral", "gist_ncar",
    ]),
])


def _color_icon(name: str) -> QPixmap:
    px = QPixmap(32, 16)
    px.fill(QColor(name))
    return px


def _cmap_icon(name: str, width: int = 64, height: int = 12) -> QPixmap:
    try:
        cmap = plt.colormaps[name]
    except KeyError:
        px = QPixmap(width, height)
        px.fill(QColor("gray"))
        return px
    rgba = (cmap(np.linspace(0, 1, width)) * 255).astype(np.uint8)
    rgba_2d = np.ascontiguousarray(np.tile(rgba, (height, 1, 1)))
    img = QImage(rgba_2d.tobytes(), width, height, width * 4, QImage.Format.Format_RGBA8888)
    return QPixmap.fromImage(img)


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

        # Build groups, appending any installed cmaps not in the known lists to Miscellaneous
        registered = set(plt.colormaps())
        known = {name for names in _CMAP_GROUPS.values() for name in names}
        groups = OrderedDict(
            (cat, [n for n in names if n in registered])
            for cat, names in _CMAP_GROUPS.items()
        )
        groups["Miscellaneous"] += sorted(registered - known - {n + "_r" for n in known})
        parameters = {
            "type": QComboBox,
            "name": "colormap",
            "groups": groups,
            "current_value": cmap,
            "icon_factory": _cmap_icon,
            "icon_size": QSize(64, 12),
            "clear_text": True,
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
            "icon_factory": _color_icon,
            "icon_size": QSize(32, 16),
            "clear_text": True,
        }
        kwargs = dict(
            widgets=cols,
            title="Plot x vs y",
            func=plot.plot_x_vs_y,
            ok_cancel_button=True,
            parent=widget,
        )

    elif popup_name == "skeleton":
        color = {
            "type": QComboBox,
            "name": "colors",
            "items": GRADED_COLOR_LIST,
            "current_index": 0,
            "icon_factory": _color_icon,
            "icon_size": QSize(32, 16),
            "clear_text": True,
        }
        kwargs = dict(
            widgets=OrderedDict(Color=color),
            title="Plot Skeleton",
            # `edges` has no widget yet (falls back to the complete-graph
            # default); only `color` is exposed here.
            func=lambda color: plot.plot_skeleton(color=color),
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
