from typing import List, Dict
from itertools import product

import fastplotlib as fpl
import pynapple as nap
import numpy as np

from .store_items import LineItem, HeatmapItem, MovieItem, ROIsItem
from .store_models import TimeStore, ComponentStore
from .store_items import StoreModelItem


class NeuroWidget:
    _viz_types: List[str] = [
        "line",
        "heatmap",
        "movie",
        "rois",
        "raster"
    ]

    def __init__(
            self,
            line: List[nap.Tsd | nap.TsdFrame] | nap.Tsd | nap.TsdFrame = None,
            heatmap: List[nap.Tsd | nap.TsdFrame] | nap.TsdFrame | nap.Tsd = None,
            movie: List[nap.Tsd | nap.TsdTensor] | nap.Tsd | nap.TsdTensor = None,
            rois: List[nap.TsdTensor] | nap.TsdTensor = None,
            raster: List[nap.TsGroup] | nap.TsGroup = None,
            names: List[str] = None,
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
        | "rois"      | nap.TsdTensor                                                                                |
        +-------------+----------------------------------------------------------------------------------------------+
        | "raster"    | nap.TsdGroup                                                                                 |
        +-------------+----------------------------------------------------------------------------------------------+

        Parameters
        ----------
        line: List[nap.Tsd | nap.TsdFrame] | nap.Tsd | nap.TsdFrame
            Optional list of pynapple objects to be made into a line visual.
        heatmap: List[nap.Tsd | nap.TsdFrame] | nap.TsdFrame | nap.Tsd
            Optional list of pynapple objects to be made into a heatmap visual.
        movie: List[nap.Tsd | nap.TsdTensor] | nap.Tsd | nap.TsdTensor
            Optional list of pynapple objects to be made into a movie visual.
        rois: List[nap.TsdTensor] | nap.TsdTensor
            Optional list of pynapple objects to be made into a rois visual.
        raster: List[nap.TsGroup] | nap.TsGroup
            Optional list of pynapple objects to be made into a raster visual.
        names: List[str], optional
            Flat list that is reshaped based on the number of visuals. The number of names in the list should match
            how many items are in data.

        """
        # time store
        self._time_store = TimeStore()
        self._component_store = ComponentStore()

        # generate a visual for each pynapple array passed in
        self._visuals = list()
        for viz_type in self._viz_types:
            if eval(viz_type) is not None:
                if isinstance(eval(viz_type), list):
                    for obj in eval(viz_type):
                        # generate visual based on key, item pairing
                        visual = self._make_visual(visual_type=viz_type, data=obj)
                        self._visuals.append(visual)
                else:
                    # generate visual based on key, item pairing
                    visual = self._make_visual(visual_type=viz_type, data=eval(viz_type))
                    self._visuals.append(visual)

        # parse data to create figure shape
        # without including any metadata, assumes that len(visuals) = # of subplots
        # will reshape into best square fit
        shape = fpl.utils.calculate_figure_shape(len(self.visuals))

        self._names = names

        if self.names is not None:
            if len(self.visuals) != len(names):
                raise ValueError(f"Each visual requires a unique name. There are {len(self.visuals)} visual and you have "
                                 f"given {len(names)} names.")

            # if odd # of visuals
            while len(self.names) < len(list(product(range(shape[0]), range(shape[1])))):
                self.names.append(None)
            # hacky way to get names in correct shape until #541 done
            self._names = list(np.array(names).reshape(shape))



        self._figure = fpl.Figure(
            shape=shape,
            names=self.names
        )

        # TODO: what happens if there is not an even number of visuals, zip won't work
        # for visual in visual, add graphics to visual, subscribe to stores
        for (viz, subplot) in zip(self.visuals, self.figure):
            subplot.add_graphic(viz.graphic)
            # add any selectors as well
            if hasattr(viz, "_time_selector"):
                subplot.add_graphic(viz.time_selector)
            if hasattr(viz, "_component_selector"):
                subplot.add_graphic(viz.component_selector)

            # register visuals to stores
            if hasattr(viz, "_set_time"):
                self.time_store.subscribe(viz)
            if hasattr(viz, "_set_component"):
                self.component_store.subscribe(viz)

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
    def component_store(self):
        """The associated ComponentStore object for the widget."""
        return self._component_store

    @property
    def visuals(self) -> List[StoreModelItem]:
        """List of visuals in the figure."""
        return self._visuals

    @property
    def names(self) -> List[str]:
        """Names of visuals in the widget."""
        return self._names

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
            case "rois":
                visual = ROIsItem(data=data)
                return visual
            case "default":
                raise ValueError("Could not create visual!")

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
