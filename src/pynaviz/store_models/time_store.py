from fastplotlib.graphics._features import FeatureEvent


from ._base_store import StoreModel
from ..store_items import *


class TimeStore(StoreModel):
    target = "time"

    def __init__(
            self,
            time: float | int = 0,

    ):
        """
        TimeStore for synchronizes components of a visual in the time axis.

        Parameters
        ----------
        time : float | int
            Time value that items in the store are currently set at.
        """
        super().__init__()
        self._time = time

    @property
    def time(self) -> float | int:
        """Current t value that items in the store are set at."""
        return self._time

    @time.setter
    def time(self, time: float | int) -> None:
        # cast passed time to the type of time being used in the store
        # float -> int or int -> float
        if not isinstance(time, type(self.time)):
            time = type(self.time)(time)

        # if new time == current time, don't update
        if time == self.time:
            return

        # update time
        self._time = time
        # update store
        self.update_store(
                ev=FeatureEvent(
                    type="",
                    info={
                        "value": self.time
                    }
                )

        )

    def subscribe(self, item: StoreModelItem):
        """Add an item to the time store."""
        # parse item
        super().subscribe(item=item)
        # time store relevant subscription stuff?
        if isinstance(item, LineItem):
            item.selector.add_event_handler(self.update_store, "selection")

    def unsubscribe(self, item: StoreModelItem):
        """Remove an item from the time store."""
        super().unsubscribe(item)

        # unhook events
        if isinstance(item, LineItem):
            item.selector.remove_event_handler(self.update_store, "selection")

    def update_store(self, ev):
        self.time = ev.info["value"]

        for item in self.store:
            item.set_time(self.time)
