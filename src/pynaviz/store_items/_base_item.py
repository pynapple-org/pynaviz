from typing import Any


class StoreModelItem:
    """Base class for an object that can be subscribed to a store model."""
    def __init__(
            self,
            data: Any
    ):
        pass

    def set_time(self):
        pass

    def set_component(self):
        pass
