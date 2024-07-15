from typing import List, Tuple
import numpy as np

import fastplotlib as fpl
import pynapple as nap

from .store_models import TimeStore
from .store_items import StoreModelItem


class Visual:
    def __init__(
            self,
            data: List[nap.TsdTensor | nap.TsdTensor | nap.Tsd],
            zoomed_plot: bool = False
    ):
        """
        Creates an interactive visual from pynapple objects using fastplotlib.

        Parameters
        ----------
        data: List[nap.TsdTensor | nap.TsdTensor | nap.Tsd]
            List of pynapple objects to create visualizations from.
        zoomed_plot: bool, default False
            Will auto-generate a fastplotlib.LinearRegionSelector for 2D line data accompanied by a plot of the
            data under the selector. Useful when wanting to inspect a subsample of the data.
        """
        self._data = data

        # generate a visual for each pynapple array passed in
        self._visuals = list()
        for data in self._data:
            visual = self._make_visual(data=data)
            self._visuals.append(visual)

        self._time_store = TimeStore()

        # parse data to create figure shape
        # without including any metadata, assumes that len(data) = # of subplots
        # will reshape into best square fit
        shape = fpl.utils.calculate_figure_shape(len(data))

        self._figure = fpl.Figure(
            shape=shape
        )

    @property
    def figure(self) -> fpl.Figure:
        """The underlying fastplotlib.Figure object."""
        return self._figure

    @property
    def data(self) -> List[nap.TsdTensor | nap.TsdTensor | nap.Tsd]:
        """A list of the pynapple objects in the visualization."""
        return self._data

    @property
    def visuals(self) -> List[StoreModelItem]:
        """List of visuals in the figure."""
        return self._visuals

    def _make_visual(self, data: nap.TsdTensor | nap.TsdTensor | nap.Tsd) -> StoreModelItem:
        # requires some kind of parsing
        pass

    def show(self):
        return self.figure.show()
