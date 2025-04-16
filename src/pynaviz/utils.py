import pygfx
import pygfx as gfx
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from .base_plot import _BasePlot


def get_plot_attribute(plot: "_BasePlot", attr_name) -> dict | None:
    """Auxiliary safe function for debugging."""
    graphic = getattr(plot, "graphic", None)
    if graphic is None:
        print(f"{plot} doesn't have a graphic.")
        return None

    dict_attr: dict = {c: getattr(graphic[c], attr_name)  for c in graphic if hasattr(graphic[c], attr_name)}
    return dict_attr


def action_caller(plot: "_BasePlot", action_name: str, metadata: Optional[dict], **kwargs):
    metadata = metadata or {}
    if action_name == "color_by":
        materials = get_plot_attribute(plot, "material")
        if materials:
            plot.color_by(materials, metadata, **kwargs)
    elif action_name == "sort_by":
        geom = get_plot_attribute(plot, "geometry")
        plot.sort_by(geom, metadata, **kwargs)