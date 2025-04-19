import fastplotlib as fpl
import numpy as np
import pynapple as nap

from ._base_item import StoreModelItem


class LineItem(StoreModelItem):
    def __init__(
        self,
        data: nap.Tsd | nap.TsdFrame,
        time_interval: nap.IntervalSet = None,
    ):
        """
        A visual for single line or multiple line data.

        Parameters
        ----------
        data : nap.Tsd | nap.TsdFrame
            Data can be a pynapple Tsd object or TsdFrame object. If data is of type nap.Tsd, then a single line
            will be created. If data is of type nap.TsdFrame, then multiple lines will be created with a fixed offset.
        """
        # check data
        if not isinstance(data, (nap.Tsd, nap.TsdFrame)):
            raise ValueError(
                f"The data passed to create a line visual must be a pynapple Tsd object or pynapple "
                f"TsdFrame object. You have passed an object of type {type(data.__class__.__name__)}."
            )

        super().__init__(data=data, time_interval=time_interval)

        # try to make a line graphic from the data
        if isinstance(data, nap.Tsd):
            data = np.column_stack((self.data.t, self.data.d))
            self._graphic = fpl.LineGraphic(data=data)
        elif isinstance(data, nap.TsdFrame):
            data = [
                np.column_stack((self.data.t, self.data.d.T[i]))
                for i in range(self.data.shape[1])
            ]

            self._graphic = fpl.LineCollection(data=data, thickness=0.5)

            for i, g in enumerate(self._graphic):
                offset = g.offset.copy()
                offset[1] = i * 1800
                g.offset = offset

            # if has attr set colors based on group
            if hasattr(data, "group"):
                self.graphic.cmap_transform = np.array(data["group"])
                self.graphic.cmap = "tab20"

        # TODO: need to decide how to decide whether a line visual should get a LinearSelector vs a
        #  LinearRegionSelector
        # create a linear selector
        bounds_init, limits, size, center = self.graphic._get_linear_selector_init_args(
            "x", 0.0
        )
        selection = bounds_init[0]

        self._time_selector = fpl.LinearSelector(
            selection=selection,
            limits=limits,
            size=size,
            center=center,
            axis="x",
            parent=self.graphic,
        )

    @property
    def time_selector(self) -> fpl.LinearSelector:
        return self._time_selector

    def _set_time(self, time: int | float):
        """Update the position of the selector in the time axis."""
        if self.time_selector.selection == time:
            return
        self.time_selector.selection = time
