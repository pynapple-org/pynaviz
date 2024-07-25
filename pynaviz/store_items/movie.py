import pynapple as nap
import fastplotlib as fpl
import numpy as np

from ._base_item import StoreModelItem


class MovieItem(StoreModelItem):
    def __init__(
            self,
            data: nap.Tsd | nap.TsdTensor,
            name: str = None
    ):
        """
        A visual for movie data.

        Parameters
        ----------
        data : nap.TsdFrame
            Data can be a pynapple
        name : str, optional
            Name of the item. Default None.
        """
        super().__init__(data=data, name=name)

        # parse data

        # first frame to start
        self._graphic = fpl.ImageGraphic(data.get(0))

    def set_time(self, time: int | float):
        """Update the data of the ImageGraphic in time."""
        self.graphic.data = self.data.get(time)
