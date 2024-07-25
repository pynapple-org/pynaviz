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
        A visual for single line or multiple line data.

        Parameters
        ----------
        data : nap.Tsd | nap.TsdFrame
            Data can be a pynapple Tsd object or TsdFrame object. If data is of type nap.Tsd, then a single line
            will be created. If data is of type nap.TsdFrame, then multiple lines will be created with a fixed offset.
        name : str, optional
            Name of the item. Default None.
        """
        # check data
        if not isinstance(data, (nap.Tsd, nap.TsdFrame)):
            raise ValueError(f"The data passed to create a line visual must be a pynapple Tsd object or pynapple "
                             f"TsdFrame object. You have passed an object of type {type(data.__class__.__name__)}.")

        super().__init__(data=data, name=name)

        # try to make a line graphic from the data
        if isinstance(data, nap.Tsd):
            data = np.column_stack((data.t, data.d))
            self._graphic = fpl.LineGraphic(data=data)
        elif isinstance(data, nap.TsdFrame):
            data = [np.column_stack((data.t, data.d.T[i])) for i in range(data.shape[1])]
            self._graphic = fpl.LineStack(data=data)

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
