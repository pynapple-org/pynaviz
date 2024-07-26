from typing import List, Dict
from itertools import product

import fastplotlib as fpl
import pynapple as nap

from .store_items import LineItem, HeatmapItem, MovieItem
from .store_models import TimeStore
from .store_items import StoreModelItem


class NeuroWidget:
    _viz_types: List[str] = [
        "line",
        "heatmap",
        "movie"
    ]

    def __init__(
            self,
            data: Dict[str, List[nap.TsdTensor | nap.TsdFrame | nap.Tsd] | nap.TsdTensor | nap.TsdFrame | nap.Tsd],
    ):
        """
        Creates an interactive visual from pynapple objects using fastplotlib.

        +-------------+----------------------------------------------------------------------------------------------+
        | Visual Type | Pynapple Object                                                                              |
        +=============+==============================================================================================+
        | "line"      | nap.Tsd or nap.TsdFrame                                                                      |
        +-------------+----------------------------------------------------------------------------------------------+
        | "heatmap"   | nap.Tsd or nap.TsdFrame                                                                      |
        +-------------+----------------------------------------------------------------------------------------------+
        | "movie"     | nap.TsdTensor                                                                                |
        +-------------+----------------------------------------------------------------------------------------------+

        Parameters
        ----------
        data: Dict[str, List[nap.TsdTensor | nap.TsdFrame | nap.Tsd] | nap.TsdTensor | nap.TsdFrame | nap.Tsd]
            Dictionary that maps the data to the desired visual type. For each type of desired visual, give
            a pynapple object or list of pynapple objects. See above for details on possible visualizations.

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
        # without including any metadata, assumes that len(visuals) = # of subplots
        # will reshape into best square fit
        shape = fpl.utils.calculate_figure_shape(len(self.visuals))

        self._figure = fpl.Figure(
            shape=shape
        )

        # TODO: what happens if there is not an even number of visuals, zip won't work
        # for visual in visual, add graphics to visual, subscribe to stores
        for (viz, subplot) in zip(self.visuals, self.figure):
            subplot.add_graphic(viz.graphic)
            if hasattr(viz, "_time_selector"):
                subplot.add_graphic(viz.time_selector)
            if hasattr(viz, "_component_selector"):
                subplot.add_graphic(viz.component_selector)

            # register visuals to stores
            if hasattr(viz, "_set_time"):
                self.time_store.subscribe(viz)

        # initial figure output is None
        self._output = None

    @property
    def figure(self) -> fpl.Figure:
        """The underlying fastplotlib.Figure object."""
        return self._figure

    @property
    def time_store(self) -> TimeStore:
        """The associated TimeStore object for the widget."""
        return self._time_store

    @property
    def visuals(self) -> List[StoreModelItem]:
        """List of visuals in the figure."""
        return self._visuals

    @staticmethod
    def _make_visual(visual_type: str, data: nap.TsdTensor | nap.TsdFrame | nap.Tsd) -> StoreModelItem:
        """Returns a visual based on the specified type and data."""
        match visual_type:
            case "line":
                visual = LineItem(data=data)
                return visual
            case "heatmap":
                visual = HeatmapItem(data=data)
                return visual
            case "movie":
                visual = MovieItem(data=data)
                return visual

    def _sync_plots(self):
        """Synchronize the cameras/controllers of each subplot."""
        # for every subplot, add every other subplot camera to the controller
        # TODO: fastplotlib only allows one iterable for a figure to exist at one time,
        #  might want to change that to simplify this code
        # TODO: need to be more selective about which subplots get synced, should depend on the type of graphic I think
        ixs = list(product(range(self.figure.shape[0]), range(self.figure.shape[1])))
        for ix in ixs:
            for __ in ixs:
                if ix == __:
                    continue
                self.figure[ix].controller.add_camera(self.figure[__].camera, include_state={"x", "width"})

    def show(self, sync_plots: bool = True):
        """Shows the visualization."""
        if self._output is None:
            # sync cameras/controllers in x, width
            self._output = self.figure.show(maintain_aspect=False)

        if sync_plots:
            self._sync_plots()

        return self._output

    def close(self):
        """Close the widget."""
        self.figure.close()
