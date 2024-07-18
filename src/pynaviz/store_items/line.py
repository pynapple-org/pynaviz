import pynapple as nap
import fastplotlib as fpl
import numpy as np

from ._base_item import StoreModelItem


class LineItem(StoreModelItem):
    def __init__(
            self,
            data: nap.TsdTensor,

    ):
        super().__init__(data=data)

        # try to make a line graphic from the data
        if isinstance(data, nap.Tsd):
            data = np.column_stack((data.t, data.d))
        self._graphic = fpl.LineGraphic(data=data)

        # create a linear selector
        self._selector = None

    @property
    def selector(self) -> fpl.LinearSelector:
        return self._selector

    @selector.setter
    def selector(self, selector: fpl.LinearSelector):
        self._selector = selector

    def set_time(self, time: int | float):
        """Update the position of the selector in the time axis."""
        if self.selector.selection == time:
            return
        self.selector.selection = time
