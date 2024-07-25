import abc
from typing import Any

from fastplotlib.graphics._base import Graphic


class StoreModelItem:
    """
    Base class for an object that can be subscribed to a store model.

    Parameters
    ----------
    data: Any
        The data of the object to be stored.
    name: str
        Optional string that gives the name of the object to be stored.
    """
    def __init__(
            self,
            data: Any,
            name: str = None,
    ):
        self._data = data
        self._name = name

        # initialize in subclass
        self._graphic = None

    @property
    def graphic(self) -> Graphic:
        """The underlying fastplotlib.Graphic object associated with item."""
        return self._graphic

    @property
    def data(self) -> Any:
        """The underlying pynapple object associated with item."""
        return self._data

    @property
    def name(self) -> str:
        """The name of the item."""
        return self._name

    @abc.abstractmethod
    def set_time(self, time: float | int):
        """Abstract method implemented in subclass StoreModelItems that have a time axes that can be set."""
        pass


