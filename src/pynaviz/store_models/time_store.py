from ._base_store import StoreModel
from ..store_items import StoreModelItem


class TimeStore(StoreModel):
    target = "time"

    def __init__(
            self,
            time: float | int = 0,

    ):
        """
        TimeStore for synchronizes components of a visual in the time axis.
        :param time:
        """
        super().__init__()
        self.time = time

    @property
    def time(self) -> float | int:
        """Current t value that items in the store are set at."""
        return self._time

    @time.setter
    def time(self, time: float | int) -> None:
        # if new time == current time, don't update
        if time == self.time:
            return

        # should parse whether working with float or int and cast

        self._time = time
        self.update_store(
            ev={
                "new": time
            }
        )
        self._time = time

    def subscribe(self, item: StoreModelItem):
        # parse item
        super().subscribe(item=item)
        # time store relevant subscription stuff?

    def unsubscribe(self, subscriber):
        pass

    def update_store(self, ev):
        pass
