import pynapple as nap
import fastplotlib as fpl
import numpy as np

from ._base_item import StoreModelItem


class LineItem(StoreModelItem):
    def __init__(
            self,
            data: nap.Tsd | nap.TsdFrame,
            name: str = None
    ):
        """
        A visual for single line data.

        Parameters
        ----------
        data : nap.Tsd | nap.TsdFrame
            Data can be a pynapple Tsd object or TsdFrame object. The data component of the object
            should be 1D, 2D, or 3D.
        name : str, optional
            Name of the item. Default None.
        """
        super().__init__(data=data, name=name)

        # try to make a line graphic from the data
        if isinstance(data, nap.Tsd) or (isinstance(data, nap.TsdFrame) and data.d.shape[1] == 1):
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
