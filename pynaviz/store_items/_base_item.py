import abc
from typing import Any, List

from fastplotlib.graphics._base import Graphic
import pynapple as nap


class StoreModelItem:
    """
    Base class for an object that can be subscribed to a store model.

    Parameters
    ----------
    data: Any
        The data of the object to be stored.
    time_interval: nap.IntervalSet, default None
        Restricts the data displayed to the specified time interval.
    """
    def __init__(
            self,
            data: Any,
            time_interval: nap.IntervalSet = None,
    ):
        self._data = data
        self._time_interval = time_interval

        # initialize in subclass
        self._graphic = None

    @property
    def graphic(self) -> Graphic | List[Graphic]:
        """The underlying fastplotlib.Graphic object associated with item."""
        return self._graphic

    @property
    def data(self) -> Any:
        """The underlying pynapple object associated with item."""
        return self._data

    @property
    def time_interval(self) -> nap.IntervalSet:
        return self._time_interval

    @abc.abstractmethod
    def _set_time(self, time: float | int):
        """Abstract method implemented in subclass StoreModelItems that have a time axes that can be set."""
        pass

    @abc.abstractmethod
    def _set_component(self, index: int):
        """Abstract method implemented in subclass StoreModelItems that have an index that can be set."""
        pass


