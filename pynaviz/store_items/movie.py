import pynapple as nap
import fastplotlib as fpl
import numpy as np

from ._base_item import StoreModelItem


class MovieItem(StoreModelItem):
    def __init__(
            self,
            data: nap.TsdTensor,
            name: str = None
    ):
        """
        A visual for movie data.

        Parameters
        ----------
        data : nap.TsdTensor
            Pynapple object for movie data.
        name : str, optional
            Name of the item. Default None.
        """
        # check data
        if not isinstance(data, nap.TsdTensor):
            raise ValueError(f"The data passed to create a movie visual must be a pynapple TsdTensor object "
                             f"You have passed an object of type {type(data.__class__.__name__)}.")
        super().__init__(data=data, name=name)

        # parse data

        # first frame to start
        self._graphic = fpl.ImageGraphic(data.get(0), cmap="gnuplot2")

    def _set_time(self, time: int | float):
        """Update the data of the ImageGraphic in time."""
        self.graphic.data = self.data.get(time)
