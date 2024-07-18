from typing import Any

from fastplotlib.graphics._base import Graphic


class StoreModelItem:
    """Base class for an object that can be subscribed to a store model."""
    def __init__(
            self,
            data: Any,
    ):
        self._data = data

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

    def set_time(self, time: float | int):
        pass


