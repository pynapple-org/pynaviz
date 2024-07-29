import pynapple as nap
import fastplotlib as fpl
import numpy as np

from ._base_item import StoreModelItem


class HeatmapItem(StoreModelItem):
    def __init__(
            self,
            data: nap.TsdFrame,
    ):
        """
        A visual for heatmap data.

        Parameters
        ----------
        data : nap.TsdFrame
            The data component of the object.
        """
        # check data
        if not isinstance(data, nap.TsdFrame):
            raise ValueError(f"The data passed to create a heatmap visual must be a pynapple TsdFrame object "
                             f"You have passed an object of type {type(data.__class__.__name__)}.")
        super().__init__(data=data)

        # try to make a heatmap from the data
        self._graphic = fpl.ImageGraphic(data=data.d.T)

        # TODO: need to decide how to decide whether a line visual should get a LinearSelector vs a
        #  LinearRegionSelector
        # create a linear selector for time
        size = self.graphic.data.value.shape[0]
        center = size / 2
        limits = (0, self.graphic.data.value.shape[1])
        # for padding purposes
        size *= 1.25
        selection = limits[0]

        self._time_selector = fpl.LinearSelector(
            selection=selection,
            limits=limits,
            size=size,
            center=center,
            axis="x",
            parent=self.graphic,
        )

        # place selector above this graphic
        self._time_selector.offset = self._time_selector.offset + (0.0, 0.0, self.graphic.offset[-1] + 1)

        # create a linear selector for component
        size = self.graphic.data.value.shape[1]
        center = size / 2
        limits = (0, self.graphic.data.value.shape[0])
        # for padding purposes
        size *= 1.25
        selection = limits[0]

        self._component_selector = fpl.LinearSelector(
            selection=selection,
            limits=limits,
            size=size,
            center=center,
            axis="y",
            parent=self.graphic,
        )

        self._component_selector.offset = self._component_selector.offset + (0.0, 0.0, self.graphic.offset[-1] + 1)

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

    def _set_time(self, time: int | float):
        """Update the position of the selector in the time axis."""
        if self.time_selector.selection == time:
            return
        self.time_selector.selection = time

