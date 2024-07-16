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
    target: str = None

    def __init__(self):
        # initialize empty list for subscribers to be added
        self._store: List[StoreModelItem] = list()

    @property
    def store(self) -> List[StoreModelItem]:
        """Returns a list of StoreModelItems subscribed to the store."""
        return self._store

    def subscribe(self, item: StoreModelItem):
        """
        Add an item to the store to be updated.

        Parameters
        ----------
        item : StoreModelItem
            Item to be added to the store.
        """
        if not isinstance(item, StoreModelItem):
            raise TypeError(f"Items subscribed to store must be of type `StoreModelItem`. You have passed an item of "
                            f"type: {type(item)}")
        if not hasattr(item, f"_viz_{self.target}"):
            raise TypeError(f"Items subscribed to store must have a {self.target} attribute to be used when the store "
                            f"is updated.")

    def unsubscribe(self, item: StoreModelItem):
        """
        Remove an item from the store.

        Parameters
        ----------
        item : StoreModelItem
            Item to be removed from the store.
        """
        if not isinstance(item, StoreModelItem):
            raise TypeError(f"All items in the store are of type `StoreModelItem`. Must pass a `StoreModelItem` to "
                            f"remove an item. You have passed an item of type {type(item)}")
        for i in self.store:
            if i == item:
                # TODO: unhook events, delete graphic
                self.store.remove(i)

    def update_store(self, ev):
        pass
