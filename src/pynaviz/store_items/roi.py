import fastplotlib as fpl
import numpy as np
import pynapple as nap

from ._base_item import StoreModelItem


class ROIsItem(StoreModelItem):
    def __init__(
        self,
        data: nap.TsdTensor,
    ):
        """
        A visual for ROI data.

        Parameters
        ----------
        data: nap.TsdFrame
            Data of the object.
        """
        # check data
        if not isinstance(data, nap.TsdTensor):
            raise ValueError(
                f"The data passed to create an ROI visual must be a pynapple TsdTensor object "
                f"You have passed an object of type {type(data.__class__.__name__)}."
            )
        super().__init__(data=data)

        # try to make line collection of ROI
        data = list(data.d)
        self._graphic = fpl.LineCollection(data=data)

    def _set_component(self, index: int):
        """Update the selected component of the ROI."""
        # reset the ROI colors
        self.graphic.colors = "w"
        # update the selected
        self.graphic[index].colors = "magenta"
