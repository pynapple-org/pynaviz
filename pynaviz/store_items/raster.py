import pynapple as nap
import fastplotlib as fpl
import numpy as np

from ._base_item import StoreModelItem

COLORS = ["hotpink", "lightpink", "cyan", "orange", "lightcoral", "lightsteelblue",
          "lime", "lightgreen", "magenta", "pink", "aliceblue"]


class RasterItem(StoreModelItem):
    def __init__(
            self,
            data: nap.TsGroup,
            time_interval: nap.IntervalSet = None,
    ):
        """
        A visual for raster data.

        Parameters
        ----------
        data: nap.TsdFrame
            Data of the object.
        """
        # check data
        if not isinstance(data, nap.TsGroup):
            raise ValueError(f"The data passed to create an Raster visual must be a pynapple TsGroup object "
                             f"You have passed an object of type {type(data.__class__.__name__)}.")
        super().__init__(data=data, time_interval=time_interval)

        scatter_graphics = list()

        if self.time_interval is not None:
            min_time = self.time_interval.start[0]
            max_time = self.time_interval.end[0]
        else:
            min_time, max_time = data.t[0], data.t[-1]

        # format raster data and make scatter graphics
        for i in range(data.index.shape[0]):
            xs = data[i].get(min_time, max_time).t
            # dummy data if there is no spiking in the time interval in order to keep index ordering
            if len(xs) == 0:
                xs = np.linspace(min_time, max_time, 3)
                ys = np.zeros(xs.shape)
            else:
                ys = np.ones((xs.shape[0]))
            scatter_data = np.column_stack((xs, ys))

            # set offset by order
            if hasattr(data, "order"):
                scatter = fpl.ScatterGraphic(data=scatter_data, offset=(0, data["order"][i], 0), sizes=5)
            # otherwise go in order
            else:
                scatter = fpl.ScatterGraphic(data=scatter_data, offset=(0, i, 0), sizes=5)

            scatter_graphics.append(scatter)

        # set graphics for visual
        self._graphic = scatter_graphics

        # color by group if possible
        if hasattr(data, "group"):
            num_groups = data["group"].max()
            for i in range(num_groups + 1):
                ixs = data["group"].index[data["group"] == i].tolist()

                subset = np.array(self.graphic)[ixs]

                for s in subset:
                    s.colors = COLORS[i]

        #add linear selector for time
        self._time_selector = fpl.LinearSelector(
            selection=0,
            limits=(min_time, max_time),
            size=data.index.shape[0] + 2,
            center=int(data.index.shape[0] / 2),
            parent=self.graphic[0]
        )

        # add a legend

    @property
    def time_selector(self) -> fpl.LinearSelector:
        return self._time_selector

    def _set_time(self, time: int | float):
        """Update the position of the selector in the time axis."""
        if self.time_selector.selection == time:
            return
        self.time_selector.selection = time

    def _set_component(self, index: int):
        pass
