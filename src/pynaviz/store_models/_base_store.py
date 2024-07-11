from typing import List

from ..store_items import StoreModelItem


class StoreModel:
    """
    Base class for all store models to inherit from.

    Attributes
    ----------
    subscribers: List[StoreModelItem] = None
        List of StoreModelItem, default None
        Items subscribed to the store will notify the store of changes to the item that then be propagated to other
        items in the store.
    """

    def __init__(self):
        # initialize empty list for subscribers to be added
        self._subscribers: List[StoreModelItem] = list()

    @property
    def subscribers(self) -> List[StoreModelItem]:
        """Returns a list of StoreModelItems subscribed to the store."""
        return self._subscribers

    def subscribe(self, subscriber):
        pass

    def unsubscribe(self, subscriber):
        pass

    def update_store(self, ev):
        pass
