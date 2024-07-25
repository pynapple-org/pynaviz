import pynapple as nap
import fastplotlib as fpl
import numpy as np

from ._base_item import StoreModelItem


class HeatmapItem(StoreModelItem):
    def __init__(
            self,
            data: nap.TsdFrame,
            name: str = None
    ):
        """
        A visual for heatmap data.

        Parameters
        ----------
        data : nap.TsdFrame
            The data component of the object.
        name : str, optional
            Name of the item. Default None.
        """
        # check data
        if not isinstance(data, nap.TsdFrame):
            raise ValueError(f"The data passed to create a heatmap visual must be a pynapple TsdFrame object "
                             f"You have passed an object of type {type(data.__class__.__name__)}.")
        super().__init__(data=data, name=name)

        # try to make a heatmap from the data
        self._graphic = fpl.ImageGraphic(data=data.d.T)

        # create a linear selector for time and component
        self._time_selector = None
        self._component_selector = None

    @property
    def time_selector(self) -> fpl.LinearSelector:
        return self._time_selector

    @time_selector.setter
    def time_selector(self, selector: fpl.LinearSelector):
        self._time_selector = selector

    @property
    def component_selector(self) -> fpl.LinearSelector:
        return self._component_selector

    @component_selector.setter
    def component_selector(self, selector: fpl.LinearSelector):
        self._component_selector = selector

    def set_time(self, time: int | float):
        """Update the position of the selector in the time axis."""
        if self.time_selector.selection == time:
            return
        self.time_selector.selection = time

