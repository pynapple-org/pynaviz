from typing import List, Tuple, Dict
import numpy as np

import fastplotlib as fpl
import pynapple as nap

from .store_items import LineItem
from .store_models import TimeStore
from .store_items import StoreModelItem


class NeuroWidget:
    _viz_types: List[str] = [
        "line"
    ]

    def __init__(
            self,
            data: Dict[str, List[nap.TsdTensor | nap.TsdFrame | nap.Tsd] | nap.TsdTensor | nap.TsdFrame | nap.Tsd],
    ):
        """
        Creates an interactive visual from pynapple objects using fastplotlib.

        Parameters
        ----------
        data: List[nap.TsdTensor | nap.TsdFrame | nap.Tsd] | nap.TsdTensor | nap.TsdFrame | nap.Tsd
            Dictionary that maps the data to the desired visual type. For each type of desired visual, give
            a pynapple object or list of pynapple objects. See below for details on possible visualizations.
        names: List[str]
            List of names for each item in `data`.


        +-------------+----------------------------------------------------------------------------------------------+
        | Visual Type | Pynapple Object                                                                              |
        +=============+==============================================================================================+
        | "line"      | nap.Tsd or nap.TsdFrame                                                                      |
        +-------------+----------------------------------------------------------------------------------------------+

        """
        # time store
        self._time_store = TimeStore()

        # generate a visual for each pynapple array passed in
        self._visuals = list()
        for key, item in data.items():
            if key not in self._viz_types:
                raise KeyError(f"The list of available visuals are: {self._viz_types} You have passed {key}.")
            # can have more than one item per type of visual
            if isinstance(item, list):
                for obj in item:
                    # generate visual based on key, item pairing
                    visual = self._make_visual(visual_type=key, data=obj)
                    self._visuals.append(visual)
            else:
                # generate visual based on key, item pairing
                visual = self._make_visual(visual_type=key, data=item)
                self._visuals.append(visual)

        # parse data to create figure shape
        # without including any metadata, assumes that len(data) = # of subplots
        # will reshape into best square fit
        shape = fpl.utils.calculate_figure_shape(len(self.visuals))

        self._figure = fpl.Figure(
            shape=shape
        )

        # for visual in visual, add graphics to visual, subscribe to stores
        for (viz, subplot) in zip(self.visuals, self.figure):
            subplot.add_graphic(viz.graphic)
            # can't add a selector for a graphic until after it has been added to a plot
            if isinstance(viz, LineItem):
                ls = viz.graphic.add_linear_selector()
                viz.selector = ls

            # register visuals to stores
            if hasattr(viz, "set_time"):
                self._time_store.subscribe(viz)

        self._output = None

    @property
    def figure(self) -> fpl.Figure:
        """The underlying fastplotlib.Figure object."""
        return self._figure

    @property
    def visuals(self) -> List[StoreModelItem]:
        """List of visuals in the figure."""
        return self._visuals

    @property
    def names(self) -> List[str]:
        """List of visual names in the figure."""
        return self._names

    def _make_visual(self, visual_type: str, data: nap.TsdTensor | nap.TsdFrame | nap.Tsd) -> StoreModelItem:
        """Returns a visual based on the specified type and data."""
        match visual_type:
            case "line":
                visual = LineItem(data=data)
                return visual

    def show(self):
        """Shows the visualization."""
        if self._output is None:
            self._output = self.figure.show()

        return self._output
