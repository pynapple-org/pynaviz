import pynapple as nap
import fastplotlib as fpl
import numpy as np

from ._base_item import StoreModelItem


class HeatmapItem(StoreModelItem):
    def __init__(
            self,
            data: nap.Tsd | nap.TsdFrame,
            name: str = None
    ):
        """
        A visual for heatmap data.

        Parameters
        ----------
        data : nap.Tsd | nap.TsdFrame
            Data can be a pynapple Tsd object or TsdFrame object. The data component of the object
            should be 1D or 2D.
        name : str, optional
            Name of the item. Default None.
        """
        super().__init__(data=data, name=name)

        # try to make a heatmap from the data
        if isinstance(data, nap.Tsd) or (isinstance(data, nap.TsdFrame) and data.d.shape[1] == 1):
            data = np.column_stack((data.t, data.d))
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

